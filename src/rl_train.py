"""
GRPO-подобное контекстное RL для рекомендательной системы.

Ключевые идеи:
    - Для каждого контекста сэмплируем G действий из π_θ.
    - Advantage = (reward − mean по группе) / (std по группе + eps).
    - Policy gradient: −E[A · log π_θ(a|h)].
    - KL штраф: β · KL(π_θ ‖ π_ref), точно по top-M кандидатам.
    - Действия ограничены top-M по π_ref + маскировка seen.
"""

import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup



#  Вспомогательные функции

def prepare_input_ids(contexts, item2idx, max_len, device):
    """
    Превращает список историй в (B, max_len) LongTensor с PAD слева.
    """
    B = len(contexts)
    input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, ctx in enumerate(contexts):
        recent = ctx[-max_len:] if ctx else []
        indexed = [item2idx.get(int(x), 1) for x in recent]
        input_ids[i, -len(indexed):] = torch.tensor(indexed, device=device)
    return input_ids


def apply_masking(logits, seen_batch, item2idx, item_pad_idx=0, item_unk_idx=1):
    """
    Маскирует PAD, UNK, seen. logits: (B, V) -> in-place модификация.

    Возвращает копию — не портит исходный тензор.
    """
    logits = logits.clone()
    logits[:, item_pad_idx] = -1e9
    logits[:, item_unk_idx] = -1e9
    B = logits.shape[0]
    for b in range(B):
        if seen_batch is not None and seen_batch[b]:
            idxs = [item2idx.get(int(x), -1) for x in seen_batch[b] if x in item2idx]
            idxs = [i for i in idxs if i >= 0]
            if idxs:
                logits[b, idxs] = -1e9
    return logits


#  GRPO loss для одного батча

def compute_grpo_loss(
    policy,
    ref_policy,
    reward_fn,
    contexts,
    seen_batch,
    item2idx,
    idx2item,
    device,
    max_len=50,
    top_m=200,
    G=8,
    beta_kl=0.03,
):
    """
    Один шаг GRPO.

    Возвращает:
        loss  — torch.Tensor (скаляр, с графом)
        info  — dict метрик (без графа)
    """
    B = len(contexts)

    # 1. Input tensors
    input_ids = prepare_input_ids(contexts, item2idx, max_len, device)

    # 2. Logits π_θ (с графом) и π_ref (без)
    logits_theta = policy(input_ids).logits[:, -1, :]  # (B, V)
    with torch.no_grad():
        logits_ref = ref_policy(input_ids).logits[:, -1, :]  # (B, V)

    # 3. Маскировка PAD/UNK/seen
    logits_theta = apply_masking(logits_theta, seen_batch, item2idx)
    with torch.no_grad():
        logits_ref = apply_masking(logits_ref, seen_batch, item2idx)

    # 4. Ограничение top-M по π_ref
    with torch.no_grad():
        if top_m is not None:
            _, top_m_idx = torch.topk(logits_ref, top_m, dim=-1)  # (B, M)
        else:
            V = logits_ref.shape[1]
            top_m_idx = torch.arange(V, device=device).unsqueeze(0).expand(B, -1)

    # 5. Логиты и log-softmax по top-M
    logits_theta_m = torch.gather(logits_theta, 1, top_m_idx)
    logits_ref_m = torch.gather(logits_ref, 1, top_m_idx)

    log_pi_theta = torch.log_softmax(logits_theta_m, dim=-1)   # (B, M)
    with torch.no_grad():
        log_pi_ref = torch.log_softmax(logits_ref_m, dim=-1)   # (B, M)
    pi_theta = log_pi_theta.exp()                              # (B, M)

    # 6. Сэмплирование G действий на контекст из π_θ (по top-M)
    with torch.no_grad():
        samples_pos = torch.multinomial(pi_theta, num_samples=G, replacement=True)  # (B, G) — позиции в top-M
        samples_idx = torch.gather(top_m_idx, 1, samples_pos)  # (B, G) — реальные индексы в vocab
        samples_item_ids = [[idx2item[int(s)] for s in row] for row in samples_idx]

    # 7. Reward для всех (B * G) пар
    flat_contexts = [contexts[i] for i in range(B) for _ in range(G)]
    flat_actions = [a for row in samples_item_ids for a in row]
    rewards = reward_fn(flat_contexts, flat_actions).view(B, G).to(device)  # (B, G)

    # 8. Advantage = (r − mean(r)) / (std(r) + ε)
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True) + 1e-6
    advantage = (rewards - mean_r) / std_r  # (B, G)
    advantage = torch.clamp(advantage, -2.0, 2.0) # Смягчение

    # 9. Policy gradient: −mean(A · log π_θ(a|h))
    log_pi_theta_sampled = torch.gather(log_pi_theta, 1, samples_pos)  # (B, G)
    pg_loss = -(advantage.detach() * log_pi_theta_sampled).mean()

    # 10. KL(π_θ ‖ π_ref), точно по top-M
    kl = (pi_theta * (log_pi_theta - log_pi_ref)).sum(dim=-1).mean()

    # 11. Total loss
    loss = pg_loss + beta_kl * kl

    info = {
        'loss': loss.item(),
        'pg_loss': pg_loss.item(),
        'kl': kl.item(),
        'mean_reward': rewards.mean().item(),
        'mean_advantage_abs': advantage.abs().mean().item(),
        'max_pi_theta': pi_theta.max(dim=-1).values.mean().item(),
    }
    return loss, info


#  Главная функция обучения

def train_rl(
    policy,
    ref_policy,
    reward_fn,
    rm_index,
    idx2item,
    item2idx,
    n_steps=500,
    batch_size=64,
    lr=5e-6,
    beta_kl=0.03,
    G=8,
    top_m=200,
    max_len=50,
    device='cuda',
    save_dir='../results/checkpoints_rl',
    seed=1,
    log_every=50,
    warmup_steps=50,
):
    """
    GRPO-подобное обучение политики π_θ.

    Параметры:
        policy       — GPT2LMHeadModel (обучаемая)
        ref_policy   — GPT2LMHeadModel (заморожена, π_ref)
        reward_fn    — callable(contexts, actions) -> Tensor(B,)
        rm_index     — RMIndex для сэмплирования контекстов из train
        n_steps      — количество шагов оптимизации (не эпох!)
        batch_size   — контекстов на шаг
        lr           — learning rate (маленький: 5e-6 или 1e-5)
        beta_kl      — сила KL-штрафа
        G            — действий на контекст (8–16)
        top_m        — ограничение на действия по π_ref (200)
        log_every    — каждые N шагов печатать метрики
    """
    # Сид
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device) if isinstance(device, str) else device
    policy = policy.to(device)
    ref_policy = ref_policy.to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = AdamW(policy.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_steps,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {'loss': [], 'pg_loss': [], 'kl': [], 'mean_reward': []}

    # Индексы train-событий для сэмплирования контекстов
    train_rows_all = np.flatnonzero(rm_index.train_mask)
    print(f"Train contexts available: {len(train_rows_all):,}")

    def sample_batch():
        """Возвращает (contexts, seen_batch) для batch_size контекстов из train."""
        rows = np.random.choice(train_rows_all, size=batch_size, replace=False)
        contexts, seen_batch = [], []
        for i in rows:
            hi = rm_index.day_start[i]
            lo = max(rm_index.user_start[i], hi - (max_len - 1))
            ctx = [idx2item[int(rm_index.items[j])] for j in range(lo, hi)]
            seen = set(ctx)
            contexts.append(ctx)
            seen_batch.append(list(seen))
        return contexts, seen_batch

    print(f"\n{'='*60}")
    print(f"Начинаем RL: {n_steps} шагов, batch={batch_size}, lr={lr}, "
          f"β={beta_kl}, G={G}, M={top_m}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        policy.train()

        contexts, seen_batch = sample_batch()

        loss, info = compute_grpo_loss(
            policy=policy,
            ref_policy=ref_policy,
            reward_fn=reward_fn,
            contexts=contexts,
            seen_batch=seen_batch,
            item2idx=item2idx,
            idx2item=idx2item,
            device=device,
            max_len=max_len,
            top_m=top_m,
            G=G,
            beta_kl=beta_kl,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        for k in history:
            if k in info:
                history[k].append(info[k])

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"Step {step:>4}/{n_steps} | "
                f"pg_loss={info['pg_loss']:+.4f} | "
                f"kl={info['kl']:.4f} | "
                f"mean_r={info['mean_reward']:.4f} | "
                f"|A|={info['mean_advantage_abs']:.3f} | "
                f"max π_θ={info['max_pi_theta']:.3f} | "
                f"{elapsed:.0f}s"
            )

        if step % 100 == 0:
            torch.save(policy.state_dict(), save_dir / f'step_{step}.pt')

    # Финальный чекпоинт
    torch.save(policy.state_dict(), save_dir / 'final_model.pt')

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RL завершено: {n_steps} шагов за {time.time()-t0:.0f}s")
    print(f"Финальные метрики (последние 20 шагов):")
    print(f"  mean_reward = {np.mean(history['mean_reward'][-20:]):.4f}")
    print(f"  kl          = {np.mean(history['kl'][-20:]):.4f}")
    print(f"Чекпоинты в: {save_dir}")
    print(f"{'='*60}")

    return history
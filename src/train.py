"""
Цикл обучения GPT-2 для next-item prediction.

Поддерживает два режима early stopping:
    - 'val_loss': классический (loss на val-выборке)
    - 'ndcg':     по NDCG@10 на val_targets (использует полный протокол оценки)

Для режима 'ndcg' нужно передать val_targets, val_context, idx2item.
В этом случае val_sequences можно оставить пустым ({}) — loss-валидация
пропускается.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dataset import build_item_vocab, build_user_sequences, NextItemDataset
from model import create_model


# ============================================================
# Вспомогательные функции
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, verbose=True):
    """
    Одна эпоха обучения.

    Возвращает:
        средний train loss за эпоху.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if verbose and n_batches % 200 == 0:
            print(f"    batch {n_batches}: loss = {loss.item():.4f}")

    return total_loss / n_batches if n_batches > 0 else float('nan')


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Средний loss на val-выборке."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float('nan')


def evaluate_ndcg_on_val(model, val_targets, val_context, item2idx, idx2item,
                          k=10, batch_size=512):
    """
    Считает NDCG@k на val_targets по полному протоколу (с маскировкой).

    Используется для early stopping и выбора чекпоинта.

    Важно: переводит model в eval() и обратно в train() в конце.
    """
    from my_evaluate import GPT2Predictor, evaluate_gpt2_fast
    from metrics import ndcg_at_k

    was_training = model.training
    model.eval()

    try:
        predictor = GPT2Predictor(model, item2idx, idx2item, max_len=50)
        preds_dict = evaluate_gpt2_fast(
            predictor, val_targets, val_context,
            k=max(k, 100), batch_size=batch_size, exclude_seen=True,
        )

        scores = []
        for _, row in val_targets.iterrows():
            preds = preds_dict.get(row['user_id'], [])[:k]
            scores.append(ndcg_at_k(preds, row['item_id'], k))
        return float(np.mean(scores))
    finally:
        if was_training:
            model.train()


# ============================================================
# Главная функция обучения
# ============================================================

def train(
    train_sequences,
    val_sequences,
    item2idx,
    vocab_size,
    max_len=50,
    batch_size=256,
    n_epochs=3,
    lr=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=3,
    device=None,
    save_dir='../results/checkpoints',
    seed=1,
    val_targets=None,
    val_context=None,
    idx2item=None,
    early_stop_metric='val_loss',  # 'val_loss' или 'ndcg'
):
    """
    Полный цикл обучения с валидацией, scheduler и early stopping.

    Параметры:
        train_sequences, val_sequences — dict {user_id: [item_id, ...]}.
            val_sequences можно оставить {} если early_stop_metric='ndcg'.
        item2idx     — словарь из build_item_vocab
        vocab_size   — размер словаря
        max_len      — длина последовательности
        batch_size   — размер батча
        n_epochs     — максимум эпох
        lr           — learning rate (пиковый)
        weight_decay — L2-регуляризация в AdamW
        warmup_ratio — доля шагов на warmup
        patience     — early stopping
        device       — torch.device или None (авто)
        save_dir     — куда сохранять чекпоинты
        seed         — сид для воспроизводимости
        val_targets  — DataFrame с колонками [user_id, item_id].
            Нужно для early_stop_metric='ndcg'
        val_context  — dict {user_id: {'items': [...], 'last_date': ...}}
            Нужно для early_stop_metric='ndcg'
        idx2item     — dict {index: item_id}
            Нужно для early_stop_metric='ndcg'
        early_stop_metric — 'val_loss' или 'ndcg'

    Возвращает:
        dict с историей: {'train_loss': [...], 'val_loss': [...],
                          'val_ndcg': [...], 'lr': [...]}
    """
    # -------- Валидация аргументов --------
    if early_stop_metric == 'ndcg':
        if val_targets is None:
            raise ValueError("early_stop_metric='ndcg', но val_targets=None")
        if val_context is None:
            raise ValueError("early_stop_metric='ndcg', но val_context=None")
        if idx2item is None:
            raise ValueError("early_stop_metric='ndcg', но idx2item=None")
    if early_stop_metric not in ('val_loss', 'ndcg'):
        raise ValueError(f"early_stop_metric должен быть 'val_loss' или 'ndcg', "
                         f"получено: {early_stop_metric!r}")

    # -------- Сид --------
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # -------- Датасеты и лоадеры --------
    train_ds = NextItemDataset(train_sequences, item2idx, max_len=max_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True,
    )

    val_ds = None
    val_loader = None
    if val_sequences:
        val_ds = NextItemDataset(val_sequences, item2idx, max_len=max_len)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
            )

    print(f"Train examples: {len(train_ds):,}")
    print(f"Val examples:   {len(val_ds) if val_ds is not None else 0:,}")
    print(f"Train batches/epoch: {len(train_loader):,}")
    print(f"Early stop metric: {early_stop_metric}")

    # -------- Модель --------
    model = create_model(vocab_size=vocab_size).to(device)

    # -------- Оптимизатор + scheduler --------
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # -------- Папка и история --------
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ndcg': [],
        'lr': [],
    }

    # best_metric всегда "чем больше — тем лучше":
    # - ndcg: напрямую
    # - val_loss: инвертируем (-val_loss)
    best_metric = -float('inf')
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Начало обучение: {n_epochs} эпох, batch={batch_size}, lr={lr}")
    print(f"Warmup: {warmup_steps} шагов из {total_steps}")
    print(f"{'='*60}\n")

    # =============
    # Основной цикл
    # =============
    for epoch in range(1, n_epochs + 1):
        print(f"--- Эпоха {epoch}/{n_epochs} ---")
        t0 = time.time()

        # 1. Обучение
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            verbose=True,
        )

        # 2. Val loss (если есть val_loader)
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, device)

        # 3. Val NDCG@10 (если early_stop_metric='ndcg')
        val_ndcg = None
        if early_stop_metric == 'ndcg':
            print("  Считаем NDCG@10 на val...")
            val_ndcg = evaluate_ndcg_on_val(
                model, val_targets, val_context, item2idx, idx2item,
            )

        dt = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # 4. История
        history['train_loss'].append(train_loss)
        if val_loss is not None:
            history['val_loss'].append(val_loss)
        if val_ndcg is not None:
            history['val_ndcg'].append(val_ndcg)
        history['lr'].append(current_lr)

        # 5. Логирование
        print(f"  train_loss = {train_loss:.4f}")
        if val_loss is not None:
            print(f"  val_loss   = {val_loss:.4f}")
        if val_ndcg is not None:
            print(f"  val_NDCG@10 = {val_ndcg:.4f}")
        print(f"  lr         = {current_lr:.2e}")
        print(f"  время      = {dt:.1f} сек\n")

        # 6. Сохранение эпохи
        torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pt')

        # 7. Текущая метрика для сравнения
        if early_stop_metric == 'ndcg':
            current_metric = val_ndcg
        else:
            current_metric = -val_loss  # -inf если val_loss = inf/None — не должно случаться

        # 8. Early stopping
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            print(f"  Новый best {early_stop_metric}: "
                  f"{current_metric if early_stop_metric == 'ndcg' else -current_metric:.4f}\n")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch}")
                break

    # ============================================================
    # Финал
    # ============================================================
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    if early_stop_metric == 'ndcg':
        print(f"Обучение завершено. Лучший val NDCG@10: {best_metric:.4f}")
    else:
        print(f"Обучение завершено. Лучший val_loss: {-best_metric:.4f}")
    print(f"Чекпоинты сохранены в: {save_dir}")
    print(f"{'='*60}")

    return history
"""
Reward Model v2: Two-tower с shared backbone.

Вход: [история..., candidate] — LongTensor (B, max_len)
Выход: логит скаляра reward — FloatTensor (B,)
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class RewardModel(nn.Module):
    def __init__(self, vocab_size, n_layer=2, n_head=4, n_embd=128,
                 n_positions=50, dropout=0.2):
        super().__init__()

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
        )

        self.backbone = GPT2Model(config)

        # Отдельный эмбеддинг для кандидата
        self.candidate_emb = nn.Embedding(vocab_size, n_embd, padding_idx=0)

        # Голова: (user_state и candidate_emb) -> score
        self.head = nn.Sequential(
            nn.Linear(2 * n_embd, n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd, n_embd // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd // 2, 1),
        )

    def forward(self, input_ids):
        """
        input_ids: (B, L) — последний токен = candidate, остальное = история.
        """
        # 1. Разделения входа на историю и кандидата
        history_ids = input_ids[:, :-1]       # (B, L-1)
        candidate_ids = input_ids[:, -1]       # (B,)

        # 2. Attention mask для истории (0 = padding)
        history_mask = (history_ids != 0).long()  # (B, L-1)

        # 3. Backbone forward
        hidden = self.backbone(
            input_ids=history_ids,
            attention_mask=history_mask,
        ).last_hidden_state  # (B, L-1, n_embd)

        # 4. User state — последняя позиция истории
        # (так как история паддится СЛЕВА, последняя позиция — реальный айтем)
        user_state = hidden[:, -1, :]  # (B, n_embd)

        # 5. Candidate embedding
        cand_emb = self.candidate_emb(candidate_ids)  # (B, n_embd)

        # 6. Concat + head
        combined = torch.cat([user_state, cand_emb], dim=-1)  # (B, 2*n_embd)
        logit = self.head(combined).squeeze(-1)  # (B,)
        return logit


def create_reward_model(vocab_size, init_from=None, **kwargs):
    """
    Создаёт RewardModel. Опционально инициализирует backbone и candidate_emb
    из обученной GPT-2.
    """
    model = RewardModel(vocab_size=vocab_size, **kwargs)

    if init_from is not None:
        state = torch.load(init_from, map_location='cpu')

        backbone_state = {}
        for k, v in state.items():
            if k.startswith('transformer.'):
                backbone_state[k[len('transformer.'):]] = v

        missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
        n_loaded = len(backbone_state) - len(missing)
        print(f"[create_reward_model] Backbone: загружено {n_loaded} тензоров")

        # Инициализация candidate_emb из backbone.wte
        with torch.no_grad():
            model.candidate_emb.weight.copy_(model.backbone.wte.weight)
        print(f"[create_reward_model] candidate_emb инициализирован из wte")

    return model


@torch.no_grad()
def predict_reward(model, input_ids, device='cpu'):
    """Инференс: возвращает reward ∈ [0, 1]."""
    was_training = model.training
    model.eval()
    model = model.to(device)

    try:
        logits = model(input_ids.to(device))
        rewards = torch.sigmoid(logits)
    finally:
        if was_training:
            model.train()

    return rewards.cpu()
# src/evaluate.py
"""
Оценка GPT-2 на val/test по протоколу из metrics.py.

Отличие от метрик бейзлайнов: GPT-2 работает в индексах словаря,
а метрики принимают item_id. Здесь делаем «обёртку»:
predict_fn(user_id, context_items, context_last_date) → list[item_id]
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class GPT2Predictor:
    """
    Обёртка над обученной GPT-2, совместимая с metrics.evaluate_model.

    Использование:
        predictor = GPT2Predictor(model, item2idx, idx2item, max_len=50)
        metrics = evaluate_model(predictor.predict, targets, context)
    """

    def __init__(self, model, item2idx, idx2item, max_len=50, device=None):
        self.model = model
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.max_len = max_len

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def predict(self, user_id, context_items, context_last_date, k=100):
        """
        Единичное предсказание. Совместимо с metrics.evaluate_model.

        Параметры:
            context_items — список item_id (по времени, старые → новые)
            k             — сколько рекомендаций вернуть

        Возвращает: список item_id (не индексов!), от лучшего к худшему.
        """
        if not context_items:
            return []

        # 1. Обрезаем до max_len и переводим в индексы
        recent = context_items[-self.max_len:]
        indexed = [self.item2idx.get(i, 1) for i in recent]  # 1 = UNK (не должно случаться)

        # 2. Паддинг слева до max_len
        pad_len = self.max_len - len(indexed)
        input_ids = [0] * pad_len + indexed
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        # 3. Прогон через модель
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
            last_logits = logits[0, -1, :].clone()

        # 4. Исключаем PAD
        last_logits[0] = float('-inf')

        # 5. Top-k индексов → top-k item_id
        top_idx = torch.topk(last_logits, k).indices.tolist()
        top_items = [self.idx2item[i] for i in top_idx if i in self.idx2item]
        return top_items

    def predict_batch(self, list_of_contexts, k=100, batch_size=256):
        """
        Батчевый inference. Возвращает список списков item_id.

        Параметры:
            list_of_contexts — список историй (каждая — список item_id)

        Возвращает: list[list[item_id]], по одному списку на каждый контекст.
        """
        results = []
        n = len(list_of_contexts)

        for start in tqdm(range(0, n, batch_size), desc="predict_batch"):
            batch_contexts = list_of_contexts[start:start + batch_size]
            batch_tensor = self._prepare_batch(batch_contexts)

            with torch.no_grad():
                logits = self.model(input_ids=batch_tensor).logits  # (B, L, V)
                last_logits = logits[:, -1, :].clone()  # (B, V)
                last_logits[:, 0] = float('-inf')  # исключить PAD

            # top-k для каждого примера
            topk = torch.topk(last_logits, k, dim=-1).indices.cpu().tolist()

            for row in topk:
                items = [self.idx2item[i] for i in row if i in self.idx2item]
                results.append(items)

        return results

    def _prepare_batch(self, batch_contexts):
        """
        Превращает список историй в тензор (B, max_len) с паддингом слева.
        """
        batch = []
        for ctx in batch_contexts:
            recent = ctx[-self.max_len:] if ctx else []
            indexed = [self.item2idx.get(i, 1) for i in recent]
            pad_len = self.max_len - len(indexed)
            row = [0] * pad_len + indexed
            batch.append(row)
        return torch.tensor(batch, dtype=torch.long).to(self.device)


def evaluate_gpt2_fast(predictor, targets_df, context, k=100, batch_size=256):
    """
    Быстрая оценка GPT-2 на всём множестве таргетов.

    Отличие от evaluate_model: строим батч по всем ЮНИКАЛЬНЫМ юзерам,
    гоняем модель один раз, потом считаем метрики.

    Параметры:
        predictor   — GPT2Predictor
        targets_df  — DataFrame с колонками user_id, item_id
        context     — dict из build_context
        k           — top-k рекомендаций
        batch_size  — размер батча для inference

    Возвращает:
        dict {user_id: [top-k item_id]}
    """
    unique_users = targets_df['user_id'].unique()
    print(f"Уникальных юзеров для оценки: {len(unique_users):,}")

    # Собираем контексты
    contexts = []
    for u in unique_users:
        ctx = context.get(u)
        items = ctx['items'] if ctx else []
        contexts.append(items)

    # Батчевый inference
    predictions = predictor.predict_batch(contexts, k=k, batch_size=batch_size)

    # Возвращаем как dict
    return dict(zip(unique_users, predictions))
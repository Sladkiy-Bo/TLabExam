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
            context_items - список item_id (по времени, старые -> новые)
            k             - сколько рекомендаций вернуть

        Возвращает: список item_id (не индексов!), от лучшего к худшему.
        """
        if not context_items:
            return []

        # 1. Обрез до max_len и переводим в индексы
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

        # 4. Исключtybt PAD
        last_logits[0] = float('-inf')

        # 5. Top-k индексов -> top-k item_id
        top_idx = torch.topk(last_logits, k).indices.tolist()
        top_items = [self.idx2item[i] for i in top_idx if i in self.idx2item]
        return top_items

    # Для каждого контекста - все его item_id -> индексы

    def predict_batch(self, list_of_contexts, k=100, batch_size=256, exclude_seen=True):
        """
        Батчевый inference с маскировкой просмотренного.

        exclude_seen=True - исключаем из top-k все айтемы, которые юзер уже видел.
        Это критично для честного сравнения: kNN маскирует, а GPT-2 должен маскировать тоже.
        """
        from tqdm.auto import tqdm
        results = []

        for start in tqdm(range(0, len(list_of_contexts), batch_size), desc="predict_batch"):
            batch_contexts = list_of_contexts[start:start + batch_size]
            batch_tensor = self._prepare_batch(batch_contexts)

            with torch.no_grad():
               logits = self.model(input_ids=batch_tensor).logits[:, -1, :].clone()

           # PAD и UNK -> -inf (никогда не рекомендуем)
            logits[:, 0] = float('-inf')
            logits[:, 1] = float('-inf')

            if exclude_seen:
                # Для каждого контекста - все его item_id → индексы
                # Собираем (row_idx, col_idx) пары и разом обнуляем
                rows, cols = [], []
                for r, ctx in enumerate(batch_contexts):
                    if not ctx:
                        continue
                    idx_seen = [self.item2idx[i] for i in ctx if i in self.item2idx]
                    if idx_seen:
                       rows.extend([r] * len(idx_seen))
                       cols.extend(idx_seen)
                if rows:
                    logits[torch.tensor(rows, device=logits.device),
                           torch.tensor(cols, device=logits.device)] = float('-inf')

            topk = torch.topk(logits, k, dim=-1).indices.cpu().tolist()
            results.extend([
                [self.idx2item[i] for i in row if i in self.idx2item]
                for row in topk
            ])

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


def evaluate_gpt2_fast(predictor, targets_df, context, k=100, batch_size=256, exclude_seen=True):
    """
    Быстрая оценка GPT-2 на всём множестве таргетов.

    Отличие от evaluate_model: строим батч по всем ЮНИКАЛЬНЫМ юзерам,
    гоняем модель один раз, потом считаем метрики.

    Параметры:
        predictor   - GPT2Predictor
        targets_df  - DataFrame с колонками user_id, item_id
        context     - dict из build_context
        k           - top-k рекомендаций
        batch_size  - размер батча для inference

    Возвращает:
        dict {user_id: [top-k item_id]}
    """
    unique_users = targets_df['user_id'].unique()
    print(f"Уникальных юзеров для оценки: {len(unique_users):,}")

    # Соборка контексты
    contexts = []
    for u in unique_users:
        ctx = context.get(u)
        items = ctx['items'] if ctx else []
        contexts.append(items)

    # Батчевый inference
    predictions = predictor.predict_batch(
        contexts, k=k, batch_size=batch_size, exclude_seen=exclude_seen
    )
    # Возвращаем как dict
    return dict(zip(unique_users, predictions))


# update
def evaluate_recent_popularity(train_df, targets_df, context, cutoff,
                                window_days=14, top_n=500, k_values=(10, 50, 100)):
    """
    Оценка Recent-Popularity от фиксированной границы сплита.

    Параметры:
        cutoff - T1 для val, T2 для test
    """
    from baselines import recent_popularity_at_cutoff
    from metrics import evaluate_model

    predict_fn = recent_popularity_at_cutoff(
        train_df, cutoff, window_days=window_days, top_n=top_n
    )
    return evaluate_model(predict_fn, targets_df, context, k_values)
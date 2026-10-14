"""
Метрики ранжирования для оценки рекомендательных моделей.
Все три метрики работают по одной схеме:
    preds  — отсортированный список item_id (от лучшего к худшему)
    target — истинный item_id, который пользователь реально посмотрел следующим
    k      — ограничение по длине списка
"""

import numpy as np
import pandas as pd


# ---------- Метрики одного пользователя ----------

def recall_at_k(preds, target, k):
    """
    Recall@K (для одного юзера) = 1.0, если истинный айтем попал в топ-K, иначе 0.0.
    Поскольку у каждого юзера ровно один правильный ответ, это просто hit@K.
    """
    return 1.0 if target in preds[:k] else 0.0


def ndcg_at_k(preds, target, k):
    """
    NDCG@K — как Recall, но с дисконтом по позиции.
    Попал на 1-е место → 1.0, на 2-е → 1/log2(3) ≈ 0.63, на 3-е → 1/log2(4) = 0.5 и т.д.
    Поощряет модели, которые ставят правильный айтем в начало списка.
    """
    for i, p in enumerate(preds[:k]):
        if p == target:
            return 1.0 / np.log2(i + 2)   # i начинается с 0 → i+2 даёт 2, 3, 4, ...
    return 0.0


def average_precision_at_k(preds, target, k):
    """
    AP@K (для одного юзера) — 1 / (позиция истинного айтема), если он попал в топ-K.
    Усреднённый AP по всем юзерам = MAP@K.
    Для нашего случая (один таргет на юзера) это просто 1/rank.
    """
    for i, p in enumerate(preds[:k]):
        if p == target:
            return 1.0 / (i + 1)
    return 0.0


# ---------- Подготовка таргетов ----------

def build_test_targets(test_df, one_per_user=True):
    """
    Строит таргеты для оценки: один (или все) item_id на пользователя.

    one_per_user=True (стандартный протокол next-item):
        для каждого юзера берём ПОСЛЕДНИЙ по времени просмотр в test
        → ровно один таргет на юзера.

    one_per_user=False:
        вернёт все взаимодействия из test → оценка по всем событиям
        (дороже, но информативнее).

    Важно: сортируем по last_watch_dt перед groupby, иначе .tail(1)
    вернёт случайную строку, а не последнюю по времени.
    """
    df = test_df.sort_values('last_watch_dt')
    if one_per_user:
        df = df.groupby('user_id').tail(1)
    return df[['user_id', 'item_id']].reset_index(drop=True)


# ---------- Главный цикл оценки ----------

def evaluate_model(predict_fn, targets_df, k_values=(10, 50, 100), verbose=True):
    """
    Оценивает модель на таргетах.

    Параметры:
        predict_fn — функция: user_id → отсортированный список item_id
                     (это универсальный интерфейс для всех моделей:
                      популярное, last-item, GPT-2 — все дают этот интерфейс)
        targets_df — DataFrame с колонками user_id, item_id (см. build_test_targets)
        k_values   — для каких K считать метрики
        verbose    — печатать прогресс каждые 10 000 юзеров

    Возвращает:
        dict {metric_name: value}, например {'Recall@10': 0.034, 'NDCG@10': 0.018, ...}
    """
    max_k = max(k_values)

    # Списки, куда будем складывать метрики каждого юзера
    accum = {}
    for k in k_values:
        accum[f'Recall@{k}'] = []
        accum[f'NDCG@{k}'] = []
        accum[f'MAP@{k}'] = []

    users = targets_df['user_id'].values
    items = targets_df['item_id'].values
    n = len(users)

    for i, (user_id, target) in enumerate(zip(users, items)):
        # Получаем рекомендации модели (только первые max_k нужны)
        preds = predict_fn(user_id)[:max_k]

        # Считаем метрики для каждого K
        for k in k_values:
            accum[f'Recall@{k}'].append(recall_at_k(preds, target, k))
            accum[f'NDCG@{k}'].append(ndcg_at_k(preds, target, k))
            accum[f'MAP@{k}'].append(average_precision_at_k(preds, target, k))

        if verbose and (i + 1) % 10000 == 0:
            print(f"  обработано {i + 1:,}/{n:,}")

    # Усредняем по всем юзерам
    return {k: float(np.mean(v)) for k, v in accum.items()}
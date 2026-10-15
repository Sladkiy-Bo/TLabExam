# src/metrics.py
"""
Метрики ранжирования для оценки рекомендательных моделей.

Протокол оценки:
    - Таргет = все взаимодействия пользователя в ПЕРВЫЙ активный день
      в соответствующем сплите (val или test).
    - Контекст для val  = история пользователя в train.
    - Контекст для test = история пользователя в train + val.
    - Метрики считаются per-target: если у юзера 3 таргета, каждый даёт вклад.
"""

import numpy as np
import pandas as pd


#  Метрики одного таргета

def recall_at_k(preds, target, k):
    """1.0, если истинный айтем попал в топ-K, иначе 0.0."""
    return 1.0 if target in preds[:k] else 0.0


def ndcg_at_k(preds, target, k):
    """NDCG@K для одного таргета: 1/log2(rank+1), 0 если мимо."""
    for i, p in enumerate(preds[:k]):
        if p == target:
            return 1.0 / np.log2(i + 2)   # i начинается с 0 → i+2 = позиция с 1
    return 0.0


def average_precision_at_k(preds, target, k):
    """AP@K для одного таргета: 1/(rank), 0 если мимо."""
    for i, p in enumerate(preds[:k]):
        if p == target:
            return 1.0 / (i + 1)
    return 0.0


#  Построение таргетов и контекста

def build_eval_targets(split_df):
    """
    Возвращает все взаимодействия пользователя в его ПЕРВЫЙ активный день
    в данном сплите.

    Параметры:
        split_df — DataFrame (val или test) с колонками user_id, item_id, last_watch_dt

    Возвращает:
        DataFrame с колонками [user_id, item_id, last_watch_dt] — все таргеты.

    Почему «первый активный день»:
        - Между концом предыдущего сплита и началом текущего проходит время.
        - «Последний таргет» может быть через 3 недели — модель не может его угадать.
        - «Первый активный день» — это первое действие пользователя после последнего
          действия в истории. Это ближе к сценарию реальной рекомендации.
    """
    df = split_df.copy()
    first_day = (df.groupby('user_id')['last_watch_dt']
                   .min()
                   .rename('first_day'))
    df = df.merge(first_day, on='user_id')
    targets = df[df['last_watch_dt'] == df['first_day']]
    targets = targets[['user_id', 'item_id', 'last_watch_dt']].reset_index(drop=True)
    return targets


def build_context(train_df, val_df=None):
    """
    Возвращает историю каждого пользователя.
    Для оценки на val:   build_context(train)
    Для оценки на test:  build_context(train, val)
    """
    if val_df is not None:
        df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        df = train_df.copy()

    df = df.sort_values(['user_id', 'last_watch_dt'])

    grouped = df.groupby('user_id')
    items_by_user = grouped['item_id'].apply(list).to_dict()
    last_date_by_user = grouped['last_watch_dt'].max().to_dict()

    context = {
        u: {'items': items_by_user[u], 'last_date': last_date_by_user[u]}
        for u in items_by_user
    }
    return context


#  Главный цикл оценки

def evaluate_model(predict_fn, targets_df, context, k_values=(10, 50, 100), verbose=True):
    """
    Оценивает модель на таргетах.

    Параметры:
        predict_fn  — функция(user_id, context_items, context_last_date) -> list[item_id]
        targets_df  — DataFrame с колонками [user_id, item_id]
        context     — dict, см. build_context
        k_values    — tuple K для метрик
        verbose     — печатать прогресс

    Возвращает:
        dict {metric_name: mean_value}, усреднённое по всем (user, target) парам.
    """
    max_k = max(k_values)

    accum = {}
    for k in k_values:
        accum[f'Recall@{k}'] = []
        accum[f'NDCG@{k}'] = []
        accum[f'MAP@{k}'] = []

    n = len(targets_df)
    users = targets_df['user_id'].values
    items = targets_df['item_id'].values

    for i, (user_id, target) in enumerate(zip(users, items)):
        user_ctx = context.get(user_id)
        if user_ctx is None:
            # Нет истории — модель получит пустой контекст
            ctx_items, ctx_date = [], None
        else:
            ctx_items, ctx_date = user_ctx['items'], user_ctx['last_date']

        preds = predict_fn(user_id, ctx_items, ctx_date)[:max_k]

        for k in k_values:
            accum[f'Recall@{k}'].append(recall_at_k(preds, target, k))
            accum[f'NDCG@{k}'].append(ndcg_at_k(preds, target, k))
            accum[f'MAP@{k}'].append(average_precision_at_k(preds, target, k))

        if verbose and (i + 1) % 50000 == 0:
            print(f"  обработано {i + 1:,}/{n:,}")

    return {k: float(np.mean(v)) for k, v in accum.items()}
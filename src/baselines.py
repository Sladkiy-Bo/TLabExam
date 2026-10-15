# src/baselines.py
"""
Классические бейзлайны для сравнения с GPT-2
(теперь 4: Popularity, Recent-Popularity, Item-kNN и Last-Item).

Все функции возвращают predict_fn(user_id, context_items, context_last_date) -> list[item_id],
где:
    context_items     — список item_id в истории юзера (старые → новые)
    context_last_date — pd.Timestamp последнего взаимодействия в истории
                        (или None, если истории нет)
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict


#  Popularity — глобальный топ-N

def popularity_baseline(train, top_n=200):
    """
    Топ-N самых популярных айтемов за весь train.
    Для всех пользователей возвращает один и тот же список.

    Самый простой бейзлайн — нижняя планка.
    """
    popular = (train.groupby('item_id')
                    .size()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index
                    .tolist())

    def predict(user_id, context_items, context_last_date):
        return popular

    return predict


#  Recent-Popularity — топ-N за окно W дней до last_date юзера

def recent_popularity_baseline(train, window_days=14, top_n=200):
    """
    Топ-N популярных айтемов за окно [context_last_date - W, context_last_date).

    Идея: у платформы быстро растёт аудитория и появляются «свежие хиты».
    Топ за последние 14 дней точнее отражает, что сейчас интересно,
    чем топ за весь train за 4 месяца.

    Кэширует popularity по уникальным датам, чтобы не считать дважды.
    """
    train_sorted = train[['item_id', 'last_watch_dt']].copy()
    train_sorted['date'] = train_sorted['last_watch_dt'].dt.normalize()

    # Fallback на глобальный топ (если окно пустое)
    global_top = (train_sorted.groupby('item_id')
                              .size()
                              .sort_values(ascending=False)
                              .head(top_n)
                              .index
                              .tolist())

    cache = {}

    def popularity_for_date(t):
        t = pd.Timestamp(t).normalize()
        if t in cache:
            return cache[t]

        start = t - pd.Timedelta(days=window_days)
        mask = (train_sorted['date'] >= start) & (train_sorted['date'] < t)
        subset = train_sorted[mask]

        if len(subset) == 0:
            top = global_top
        else:
            top = (subset.groupby('item_id')
                         .size()
                         .sort_values(ascending=False)
                         .head(top_n)
                         .index
                         .tolist())

        cache[t] = top
        return top

    def predict(user_id, context_items, context_last_date):
        if context_last_date is None:
            return global_top
        return popularity_for_date(context_last_date)

    return predict


#  Last-Item с популярным fallback

def last_item_baseline(train, top_n=200):
    """
    Рекомендует последний просмотренный айтем + популярные для заполнения.

    ВАЖНО: в KION repeat rate = 0%. Это значит, что «последний просмотренный»
    практически никогда не окажется в таргетах. Фактически это
    popularity baseline с бесполезным шумом в топ-1.
    Да, по сути, он бесполезен.
    """
    train_sorted = train.sort_values('last_watch_dt')
    last_item = (train_sorted.groupby('user_id')['item_id']
                             .last()
                             .to_dict())

    popular = (train.groupby('item_id')
                    .size()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index
                    .tolist())

    def predict(user_id, context_items, context_last_date):
        seen = last_item.get(user_id)
        if seen is None:
            return popular
        rest = [x for x in popular if x != seen]
        return [seen] + rest

    return predict


#  Чистый Last-Item — только последний просмотренный, без fallback

def pure_last_item_baseline(train):
    """
    Рекомендует РОВНО последний просмотренный айтем, без popular fallback.

    Ожидаемый результат: Recall@10 = 0. Демонстрирует, что при нулевом
    repeat rate «last item» бесполезен как рекомендация.
    """
    train_sorted = train.sort_values('last_watch_dt')
    last_item = (train_sorted.groupby('user_id')['item_id']
                             .last()
                             .to_dict())

    def predict(user_id, context_items, context_last_date):
        seen = last_item.get(user_id)
        return [seen] if seen is not None else []

    return predict


#  Item-kNN — «пользователи, которые смотрели X, часто смотрят Y»

def item_knn_baseline(train, top_k_similar=50, top_n=200, max_history=20):
    """
    Item-to-item collaborative filtering.

    Алгоритм:
      1. Ограничивает историю каждого юзера последними max_history айтемами
         (для скорости — не больше 20).
      2. Для каждой пары (a, b) внутри одной истории увеличивается счётчик
         co-occurrence[a][b].
      3. Для каждого айтема оставляются top_k_similar самых частых соседей.
      4. При predict:
         - берутся последние max_history айтемов из истории юзера;
         - для каждого айтема добавляются его соседи в скоринг с весом
           (новее — важнее);
         - ранжируется по сумме весов;
         - исключается уже просмотренное;
         - хвост добивается популярными.
    """
    # 1. История юзеров, отсортированная по времени
    train_sorted = train.sort_values(['user_id', 'last_watch_dt'])
    user_items = (train_sorted.groupby('user_id')['item_id']
                              .apply(list)
                              .to_dict())

    # 2. Co-occurrence matrix (только по последним max_history айтемам)
    cooc = defaultdict(Counter)
    for items in user_items.values():
        recent = items[-max_history:]
        n = len(recent)
        for i in range(n):
            for j in range(n):
                if i != j:
                    cooc[recent[i]][recent[j]] += 1

    # 3. Для каждого айтема — top_k_similar соседей
    top_similar = {
        item: [sim for sim, _ in counter.most_common(top_k_similar)]
        for item, counter in cooc.items()
    }

    # 4. Fallback-популярные для добивания хвоста
    popular_fallback = (train.groupby('item_id')
                             .size()
                             .sort_values(ascending=False)
                             .head(top_n * 2)
                             .index
                             .tolist())

    def predict(user_id, context_items, context_last_date):
        if not context_items:
            return popular_fallback[:top_n]

        recent = context_items[-max_history:]
        context_set = set(context_items)

        # Взвешенный скоринг: чем новее — тем больше вес
        scores = Counter()
        for i, item in enumerate(recent):
            weight = 1.0 / (len(recent) - i)
            for sim_item in top_similar.get(item, []):
                scores[sim_item] += weight

        # Ранжируется и исключается уже просмотренное
        ranked = [it for it, _ in scores.most_common() if it not in context_set]

        # Хвост добивается популярными
        if len(ranked) < top_n:
            ranked_set = set(ranked)
            for item in popular_fallback:
                if item not in context_set and item not in ranked_set:
                    ranked.append(item)
                    if len(ranked) >= top_n:
                        break

        return ranked[:top_n]

    return predict
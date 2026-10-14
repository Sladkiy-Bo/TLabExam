"""
Простые бейзлайны для сравнения с GPT-2.
Оба возвращают predict_fn(user_id) → list[item_id], совместимый с evaluate_model.
"""

import pandas as pd


def popularity_baseline(train, top_n=200):
    """
    Всем пользователям — один и тот же топ-N популярных айтемов.

    Логика: считаем, сколько раз каждый айтем встречается в train,
    сортируем по убыванию, берём первые top_n.

    Это "нижняя планка". Если GPT-2 не бьёт её — с GPT-2 что-то не так.
    """

    popular = (train.groupby('item_id')
                    .size()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index
                    .tolist())

    def predict(user_id):
        return popular

    return predict


def last_item_baseline(train, top_n=200):
    """
    Рекомендует пользователю то, что он смотрел последним (в train),
    плюс популярное как fallback для тех, у кого нет истории.
    """

    # Сортировка по времени, потом берём последний item_id на юзера
    train_sorted = train.sort_values('last_watch_dt')
    last_item = (train_sorted.groupby('user_id')['item_id']
                             .last()
                             .to_dict())

    # Fallback-список популярных
    popular = (train.groupby('item_id')
                    .size()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index
                    .tolist())

    def predict(user_id):
        seen = last_item.get(user_id)
        if seen is None:
            # Юзер без истории в train (на всякий случай)
            return popular
        # Последний просмотренный идёт на 1-е место, следом за ним - популярные (кроме него)
        rest = [x for x in popular if x != seen]
        return [seen] + rest

    return predict
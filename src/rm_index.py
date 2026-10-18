"""
Быстрый numpy-индекс для Reward Model.

Идея: одно большое numpy-представление всех событий,
отсортированных по (user_id, date). Тогда история события —
это непрерывный кусок массива.

Экономия памяти: ~60 МБ вместо ~10 ГБ.
Скорость: секунды вместо минут.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RMIndex:
    """
    Единый индекс для RM. Строится по объединённым данным (train + val + test),
    но каждое событие имеет маску — к какому сплиту относится.
    """

    def __init__(self, all_df, item2idx, t1, t2):
        """
        Параметры:
            all_df    — DataFrame с колонками user_id, item_id, last_watch_dt, watched_pct
                        (должен быть уже отфильтрован по холодным)
            item2idx  — dict {item_id: index}
            t1, t2    — границы сплитов (pd.Timestamp)
        """
        # Сортировка по (user_id, last_watch_dt). kind='stable' важен:
        # внутри одного дня порядок сохраняется как в исходных данных.
        df = all_df.sort_values(
            ['user_id', 'last_watch_dt'],
            kind='stable',
        ).reset_index(drop=True)

        # Позиции (0..N-1)
        pos = np.arange(len(df))

        # user_start: индекс первого события юзера
        # groupby(...).cumcount() даёт 0,1,2,... внутри группы.
        # pos - cumcount = индекс начала группы для каждой строки.
        user_cumcount = df.groupby('user_id').cumcount().to_numpy()
        self.user_start = pos - user_cumcount

        # day_start: индекс первого события текущего дня юзера
        # (группировка по user_id + normalized date)
        df['_date'] = df['last_watch_dt'].dt.normalize()
        day_cumcount = df.groupby(['user_id', '_date']).cumcount().to_numpy()
        self.day_start = pos - day_cumcount

        # Основные массивы
        self.items = df['item_id'].map(item2idx).to_numpy(dtype=np.int32)
        assert not pd.isna(self.items).any(), "В items есть NaN — какой-то item_id не в словаре"

        self.rewards = (df['watched_pct'].to_numpy() / 100.0).astype(np.float32)
        self.dates = df['last_watch_dt'].to_numpy()
        self.users = df['user_id'].to_numpy()

        # has_history: True, если у события есть хотя бы одно событие до начала дня
        # (т.е. day_start > user_start)
        self.has_history = self.day_start > self.user_start

        # Сплиты
        t1 = pd.Timestamp(t1)
        t2 = pd.Timestamp(t2)
        self.train_mask = self.has_history & (df['last_watch_dt'] < t1).to_numpy()
        self.val_mask   = self.has_history & (df['last_watch_dt'] >= t1).to_numpy() & (df['last_watch_dt'] < t2).to_numpy()
        self.test_mask  = self.has_history & (df['last_watch_dt'] >= t2).to_numpy()

        self.n_total = len(df)
        print(f"RMIndex: {self.n_total:,} событий")
        print(f"  train: {self.train_mask.sum():,}")
        print(f"  val:   {self.val_mask.sum():,}")
        print(f"  test:  {self.test_mask.sum():,}")


class RMDataset(Dataset):
    """
    Dataset поверх RMIndex. Один пример = одно событие.

    __getitem__(k) берёт k-е событие (по маске) и строит:
        input_ids — история (обрезанная до max_len-1) + [candidate]
        reward    — watched_pct этого события
    """

    def __init__(self, index, mask, max_len=50):
        self.index = index
        self.rows = np.flatnonzero(mask)  # индексы строк, попадающих под маску
        self.max_len = max_len
        print(f"RMDataset: {len(self.rows):,} примеров")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i = self.rows[k]

        # История — от user_start до day_start
        hi = self.index.day_start[i]
        lo = max(self.index.user_start[i], hi - (self.max_len - 1))

        # seq = [история] + [candidate]
        seq = np.concatenate([
            self.index.items[lo:hi],
            [self.index.items[i]],
        ])

        # Паддинг слева
        input_ids = np.zeros(self.max_len, dtype=np.int64)
        input_ids[-len(seq):] = seq

        return (
            torch.from_numpy(input_ids),
            torch.tensor(self.index.rewards[i], dtype=torch.float32),
        )
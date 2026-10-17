# src/dataset.py
"""
Подготовка данных для обучения GPT-2 на задаче next-item prediction.

Ключевые концепции:
    - item_id -> contiguous index через словарь
    - Последовательности паддятся СЛЕВА до max_len
    - labels = input_ids, но PAD -> -100 (ignore_index)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Специальные индексы словаря
PAD_IDX = 0
UNK_IDX = 1


def build_item_vocab(train_df):
    """
    Строит словарь item_id -> contiguous index.

    Схема словаря:
        0       → [PAD]
        1       → [UNK]
        2..N+1  → реальные айтемы из train (в отсортированном порядке)

    Параметры:
        train_df — DataFrame с колонкой item_id

    Возвращает:
        item2idx: dict {item_id: index}
        idx2item: dict {index: item_id}
    """
    unique_items = sorted(train_df['item_id'].unique())
    item2idx = {item: idx + 2 for idx, item in enumerate(unique_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}
    return item2idx, idx2item


def build_user_sequences(df, max_len=50):
    """
    Для каждого пользователя строит его историю — список item_id
    в хронологическом порядке, обрезанный до последних max_len айтемов.

    Параметры:
        df      — DataFrame с колонками user_id, item_id, last_watch_dt
        max_len — максимум айтемов в истории

    Возвращает:
        dict {user_id: [item_id, item_id, ...]}

    Вместо сортировки всего DataFrame используется groupby, т. к. он
        сохраняет порядок строк внутри группы, и можно
        применить list() к каждой группе отдельно. Это быстрее,
        чем сортировать миллионы строк и потом резать.
    """
    df = df.sort_values(['user_id', 'last_watch_dt'])

    sequences = {}
    for user_id, group in df.groupby('user_id', sort=False):
        items = group['item_id'].tolist()
        sequences[user_id] = items[-max_len:]

    return sequences


class NextItemDataset(Dataset):
    """
    PyTorch Dataset для задачи next-item prediction.

    Каждый элемент — одна последовательность пользователя,
    выровненная до max_len паддингом слева.

    Возвращает dict:
        input_ids : LongTensor формы (max_len,)
        labels    : LongTensor формы (max_len,), где PAD → -100
    """

    def __init__(self, sequences, item2idx, max_len=50):
        """
        Параметры:
            sequences — dict {user_id: [item_id, ...]} из build_user_sequences
            item2idx  — dict из build_item_vocab
            max_len   — длина последовательности после паддинга
        """
        self.max_len = max_len
        self.item2idx = item2idx

        # Если последовательность < 2 - предсказывать нечего
        self.samples = []
        for seq in sequences.values():
            if len(seq) < 2:
                continue
            indexed = [item2idx[item] for item in seq]
            self.samples.append(indexed)

    def __len__(self):
        # Сколько всего обучающих примеров (последовательностей).
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Возвращает idx-й пример: input_ids и labels.

        Пример:
            seq = [5, 7, 9]  (в индексах)
            pad_len = 50 - 3 = 47
            input_ids = [0]*47 + [5, 7, 9]
            labels    = [-100]*47 + [5, 7, 9]
        """
        seq = self.samples[idx]
        pad_len = self.max_len - len(seq)

        # Паддинг слева
        input_ids = [PAD_IDX] * pad_len + seq

        # labels = input_ids, но PAD -> -100
        labels = [-100 if x == PAD_IDX else x for x in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels':    torch.tensor(labels,    dtype=torch.long),
        }
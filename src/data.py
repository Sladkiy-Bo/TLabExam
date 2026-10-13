# src/data.py
"""
Загрузка KION, временной сплит, фильтрация холодных юзеров/айтемов.
Запуск: python data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------- Пути ----------
DATA_RAW = Path('../data/raw')
DATA_PROCESSED = Path('../data/processed')
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------- Загрузка ----------
def load_raw():
    interactions = pd.read_csv(DATA_RAW / 'interactions.csv', parse_dates=['last_watch_dt'])
    users = pd.read_csv(DATA_RAW / 'users_en.csv')
    items = pd.read_csv(DATA_RAW / 'items_en.csv')

    # Чистим: 828 строк с пропуском в watched_pct (0.015%) — выкидываем
    n_before = len(interactions)
    interactions = interactions.dropna(subset=['watched_pct']).reset_index(drop=True)
    print(f"[load_raw] Удалено {n_before - len(interactions)} строк с NaN в watched_pct")
    print(f"[load_raw] Осталось: {len(interactions):,}")

    return interactions, users, items


# ---------- Диагностика сплита ----------
def diagnose_split(interactions, targets=(0.70, 0.80, 0.90)):
    """
    Строит график daily counts + cumulative share и возвращает даты,
    на которых накопленная доля достигает target.
    """
    df = interactions.sort_values('last_watch_dt').reset_index(drop=True)
    df['cum_share'] = (df.index + 1) / len(df)

    # Даты для каждого target
    result = {}
    for t in targets:
        row = df[df['cum_share'] >= t]
        if len(row) == 0:
            result[t] = df['last_watch_dt'].max()
        else:
            result[t] = row.iloc[0]['last_watch_dt']
        print(f"[diagnose] {int(t*100)}% накоплено к: {result[t]}")

    # График
    daily = df.groupby(df['last_watch_dt'].dt.date).size()
    cum_pct = daily.cumsum() / daily.sum() * 100

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(daily.index, daily.values, alpha=0.4, color='tab:blue', label='Daily count')
    ax1.set_ylabel('Daily interactions', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(cum_pct.index, cum_pct.values, color='red', linewidth=2, label='Cumulative %')
    for t in targets:
        ax2.axhline(t * 100, color='red', linestyle='--', alpha=0.4)
    ax2.set_ylabel('Cumulative %', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Daily interactions + cumulative share')
    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'split_diagnostic.png', dpi=100)
    plt.show()

    return result


# ---------- Сплит ----------
def make_temporal_split(interactions, t1, t2):
    """
    Глобальный сплит по времени:
    train: last_watch_dt <  t1
    val:   t1 <= last_watch_dt < t2
    test:  last_watch_dt >= t2
    """
    t1 = pd.Timestamp(t1)
    t2 = pd.Timestamp(t2)

    train = interactions[interactions['last_watch_dt'] < t1].copy()
    val = interactions[(interactions['last_watch_dt'] >= t1) &
                       (interactions['last_watch_dt'] < t2)].copy()
    test = interactions[interactions['last_watch_dt'] >= t2].copy()
    return train, val, test


def filter_cold(train, val, test, filter_users=True):
    """
    Оставляем только айтемы (и опционально юзеров), которые есть в train.
    filter_users=True  — строгий вариант (метрики честные, но теряем много данных).
    filter_users=False — мягкий вариант (меньше потерь, но у части юзеров пустой контекст).
    """
    train_items = set(train['item_id'].unique())
    train_users = set(train['user_id'].unique())

    if filter_users:
        val = val[val['user_id'].isin(train_users) & val['item_id'].isin(train_items)]
        test = test[test['user_id'].isin(train_users) & test['item_id'].isin(train_items)]
    else:
        val = val[val['item_id'].isin(train_items)]
        test = test[test['item_id'].isin(train_items)]

    return train, val, test


# ---------- Main ----------
def main():
    interactions, users, items = load_raw()

    print("\n" + "=" * 60)
    print("Диагностика сплита")
    print("=" * 60)
    dates = diagnose_split(interactions, targets=(0.70, 0.80))

    # Автоматически выбранные границы
    t1 = dates[0.70]
    t2 = dates[0.80]
    print(f"\nАвтосплит: T1 = {t1.date()}, T2 = {t2.date()}")

    # Если хочешь задать вручную — раскомментируй:
    # t1 = pd.Timestamp('2021-07-01')
    # t2 = pd.Timestamp('2021-07-20')

    print("\n" + "=" * 60)
    print("Сплит до фильтрации")
    print("=" * 60)
    train, val, test = make_temporal_split(interactions, t1, t2)
    _print_split(train, val, test)

    print("\n" + "=" * 60)
    print("Сплит после фильтрации холодных (filter_users=True)")
    print("=" * 60)
    train_f, val_f, test_f = filter_cold(train, val, test, filter_users=True)
    _print_split(train_f, val_f, test_f)

    # Сохраняем отфильтрованные
    train_f.to_parquet(DATA_PROCESSED / 'train.parquet', index=False)
    val_f.to_parquet(DATA_PROCESSED / 'val.parquet', index=False)
    test_f.to_parquet(DATA_PROCESSED / 'test.parquet', index=False)
    print(f"\nСохранено в {DATA_PROCESSED}/")


def _print_split(train, val, test):
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        if len(df) == 0:
            print(f"{name}: EMPTY")
            continue
        print(f"{name}: {len(df):>10,} | "
              f"{df['last_watch_dt'].min().date()} → {df['last_watch_dt'].max().date()}")


if __name__ == '__main__':
    main()
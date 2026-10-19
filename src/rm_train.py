"""
Обучение Reward Model.

Loss: BCEWithLogitsLoss на мягких метках r ∈ [0, 1] = watched_pct / 100.
Метрики: MAE, RMSE, R^2 на val и test.

Использует RMIndex + RMDataset из rm_index.py.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from rm_index import RMIndex, RMDataset
from reward_model import create_reward_model


# ==================
#  Метрики регрессии
# ==================

@torch.no_grad()
def evaluate_regression(model, loader, device):
    """
    Оценивает RM на отложенной выборке.

    Возвращает:
        dict с метриками: {'mae': ..., 'rmse': ..., 'r2': ...}
    """
    model.eval()
    all_preds = []
    all_targets = []

    for input_ids, rewards in loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(rewards.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}


# =====================
#  Обучение: одна эпоха
# =====================

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, verbose=True):
    """
    Одна эпоха обучения RM.

    Возвращает:
        средний train loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for input_ids, rewards in loader:
        input_ids = input_ids.to(device)
        rewards = rewards.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)              # (B,)
        loss = criterion(logits, rewards)      # BCEWithLogitsLoss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if verbose and n_batches % 200 == 0:
            print(f"    batch {n_batches}: loss = {loss.item():.4f}")

    return total_loss / n_batches if n_batches > 0 else float('nan')


# =========================
#  Главная функция обучения
# =========================

def train_rm(
    rm_index,
    vocab_size,
    init_from=None,
    batch_size=256,
    n_epochs=5,
    lr=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    patience=2,
    device=None,
    save_dir='../results/checkpoints_rm',
    seed=1,
    limit_batches=None,
):
    """
    Полный цикл обучения Reward Model.

    Параметры:
        rm_index    — RMIndex из rm_index.py (построенный по train+val+test)
        vocab_size  — размер словаря (14 937)
        init_from   — путь к чекпоинту GPT-2 (best_model.pt). Если None —
                      backbone обучается с нуля (плохая идея, но возможно).
        batch_size  — размер батча
        n_epochs    — максимум эпох
        lr          — learning rate (5e-4 — backbone уже обучен)
        weight_decay — L2-регуляризация
        warmup_ratio — доля warmup-шагов
        patience    — early stopping
        device      — 'cpu' или 'cuda' (None = авто)
        save_dir    — куда сохранять чекпоинты
        seed        — сид
        limit_batches — для отладки: обрезает эпоху до N батчей

    Возвращает:
        dict с историей: {'train_loss': [...], 'val_mae': [...], ...}
    """
    # Сид
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Датасеты
    train_ds = RMDataset(rm_index, rm_index.train_mask, max_len=50)
    val_ds = RMDataset(rm_index, rm_index.val_mask, max_len=50)
    test_ds = RMDataset(rm_index, rm_index.test_mask, max_len=50)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    print(f"Train examples: {len(train_ds):,}")
    print(f"Val examples:   {len(val_ds):,}")
    print(f"Test examples:  {len(test_ds):,}")
    print(f"Train batches/epoch: {len(train_loader):,}")

    # Модель
    model = create_reward_model(
        vocab_size=vocab_size,
        init_from=init_from,
    ).to(device)

    # Оптимизатор
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # Scheduler
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Папка
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # История
    history = {
        'train_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'lr': [],
    }

    best_val_mae = float('inf')
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Начало обучения RM: {n_epochs} эпох, batch={batch_size}, lr={lr}")
    print(f"Warmup: {warmup_steps} шагов из {total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(1, n_epochs + 1):
        print(f"--- Эпоха {epoch}/{n_epochs} ---")
        t0 = time.time()

        # 1. Обучение
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            verbose=True,
        )

        # 2. Валидация
        val_metrics = evaluate_regression(model, val_loader, device)
        dt = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # 3. История
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(current_lr)

        # 4. Логирование
        print(f"  train_loss = {train_loss:.4f}")
        print(f"  val_mae    = {val_metrics['mae']:.4f}")
        print(f"  val_rmse   = {val_metrics['rmse']:.4f}")
        print(f"  val_r2     = {val_metrics['r2']:.4f}")
        print(f"  lr         = {current_lr:.2e}")
        print(f"  время      = {dt:.1f} сек\n")

        # 5. Сохранение
        torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pt')

        # 6. Early stopping по val_mae
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            print(f"  Новый best val_mae: {best_val_mae:.4f}\n")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch}")
                break

    # Финальная оценка на test
    print(f"\n{'='*60}")
    print(f"Финальная оценка на test (лучший чекпоинт)")
    print(f"{'='*60}")

    best_state = torch.load(save_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(best_state)

    test_metrics = evaluate_regression(model, test_loader, device)
    print(f"  test_mae  = {test_metrics['mae']:.4f}")
    print(f"  test_rmse = {test_metrics['rmse']:.4f}")
    print(f"  test_r2   = {test_metrics['r2']:.4f}")

    # Сохранение истории
    with open(save_dir / 'history.json', 'w') as f:
        json.dump({
            'history': history,
            'best_val_mae': best_val_mae,
            'test_metrics': test_metrics,
        }, f, indent=2)

    print(f"\nЧекпоинты сохранены в: {save_dir}")
    print(f"{'='*60}")

    return history, test_metrics
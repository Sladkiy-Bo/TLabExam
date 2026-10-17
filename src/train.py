# src/train.py
"""
Цикл обучения GPT-2 для next-item prediction.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dataset import build_item_vocab, build_user_sequences, NextItemDataset
from model import create_model


def train_one_epoch(model, loader, optimizer, scheduler, device, verbose=True):
    """
    Одна эпоха обучения.

    Возвращает:
        средний train loss за эпоху.

    Как это работает:
        1. model.train() — режим обучения (включает dropout).
        2. Проход по батчам из loader.
        3. Для каждого батча: forward -> loss -> backward -> optimizer.step().
        4. Логирование каждого из N батчей.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # Перенос данных на GPU/CPU
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # 1. Обнуление градиентов с прошлой итерации
        optimizer.zero_grad()

        # 2. Forward pass. HuggingFace сам считает loss,
        #    если передать labels
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # 3. Backward pass (вычисление градиентов)
        loss.backward()

        # 4. Gradient clipping — защита от «взрыва» градиентов
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Шаг оптимизатора
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if verbose and n_batches % 200 == 0:
            print(f"    batch {n_batches}: loss = {loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """
    Считает средний loss на val-выборке.

    @torch.no_grad() — декоратор, отключающий градиенты.
    Без него val был бы медленнее, потому что PyTorch
    продолжал бы строить граф вычислений.

    Возвращает:
        средний val loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1

    return total_loss / n_batches


def train(
    train_sequences,
    val_sequences,
    item2idx,
    vocab_size,
    max_len=50,
    batch_size=256,
    n_epochs=3,
    lr=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=3,
    device=None,
    save_dir='../results/checkpoints',
    seed=1,
    limit_batches=None,
):
    """
    Полный цикл обучения с валидацией, scheduler и early stopping.

    Параметры:
        train_sequences, val_sequences — dict {user_id: [item_id, ...]}
        item2idx     — словарь из build_item_vocab
        vocab_size   — размер словаря
        max_len      — длина последовательности
        batch_size   — размер батча
        n_epochs     — максимум эпох
        lr           — learning rate (пиковый)
        weight_decay — L2-регуляризация в AdamW
        warmup_ratio — доля шагов на warmup
        patience     — early stopping
        device       — torch.device или None (авто)
        save_dir     — куда сохранять чекпоинты
        seed         — сид для воспроизводимости
        limit_batches — если задано, обрезает эпохи до N батчей (для отладки)

    Возвращает:
        dict с историей обучения: {'train_loss': [...], 'val_loss': [...]}
    """
    # Фиксация сида для воспроизводимости
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Создаём датасеты и лоадеры
    train_ds = NextItemDataset(train_sequences, item2idx, max_len=max_len)
    val_ds = NextItemDataset(val_sequences, item2idx, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    print(f"Train examples: {len(train_ds):,}")
    print(f"Val examples:   {len(val_ds):,}")
    print(f"Train batches/epoch: {len(train_loader):,}")

    # Модель рождается
    model = create_model(vocab_size=vocab_size).to(device)

    # Оптимизатор AdamW
    # weight_decay не применяется к bias и LayerNorm
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

    # Папка для чекпоинтов
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # История
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Начинаем обучение: {n_epochs} эпох, batch={batch_size}, lr={lr}")
    print(f"Warmup: {warmup_steps} шагов из {total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(1, n_epochs + 1):
        print(f"--- Эпоха {epoch}/{n_epochs} ---")
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            verbose=(limit_batches is None),
        )
        val_loss = evaluate_loss(model, val_loader, device)
        dt = time.time() - t0

        # Текущий learning rate
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"  train_loss = {train_loss:.4f}")
        print(f"  val_loss   = {val_loss:.4f}")
        print(f"  lr         = {current_lr:.2e}")
        print(f"  время      = {dt:.1f} сек\n")

        # Сохранение модели эпохи
        torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pt')

        # Early stopping по val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            print(f"  Новый best val_loss: {best_val_loss:.4f}\n")
        else:
            patience_counter += 1
            print(f"  Патience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch}")
                break

    # Сохранение истории
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Обучение завершено. Лучший val_loss: {best_val_loss:.4f}")
    print(f"Чекпоинты сохранены в: {save_dir}")
    print(f"{'='*60}")

    return history
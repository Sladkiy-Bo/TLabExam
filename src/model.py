# src/model.py
"""
Создание GPT-2 модели для next-item prediction.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def create_model(vocab_size, n_layer=2, n_head=4, n_embd=128,
                 n_positions=50, dropout=0.2):
    """
    Создаёт GPT-2 без предобученных весов.

    Параметры:
        vocab_size  — размер словаря (из build_item_vocab)
        n_layer     — число Transformer-блоков
        n_head      — число голов attention
        n_embd      — размер эмбеддингов
        n_positions — максимальная длина последовательности
        dropout     — вероятность dropout

    Возвращает:
        GPT2LMHeadModel — готовая модель.
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=dropout,   # dropout в residual connections
        embd_pdrop=dropout,    # dropout в эмбеддингах
        attn_pdrop=dropout,    # dropout в attention

    )

    model = GPT2LMHeadModel(config)
    return model


def recommend_top_k(model, context, k=10, item_pad_idx=0):
    """
    Возвращает top-k item_id по последней позиции контекста.

    Используется при оценке модели: для каждого пользователя мы хотим
    получить ранжированный список рекомендаций.

    Параметры:
        model        — обученная GPT-2
        context      — LongTensor формы (1, seq_len) или (seq_len,) — история юзера
        k            — сколько рекомендаций вернуть
        item_pad_idx — индекс PAD (не должен попасть в рекомендации)

    Возвращает:
        list[int] — top-k индексов (в словаре item2idx), от лучшего к худшему.

    Как это работает:
        1. Прогонка контекста через модель -> получение logits.
        2. logits.shape = (1, seq_len, vocab_size).
        3. Получение logits на последней позиции -> вектор (vocab_size,).
        4. Исключение PAD.
        5. Возвращение top-k.
    """
    model.eval()
    with torch.no_grad():
        if context.dim() == 1:
            context = context.unsqueeze(0)   # -> (1, seq_len)

        # Прогон через модель. Возвращает объект с атрибутом .logits
        outputs = model(input_ids=context)
        logits = outputs.logits               # (1, seq_len, vocab_size)

        # Получение последней позиции -> (vocab_size,)
        last_logits = logits[0, -1, :]

        # -inf на PAD, чтобы он не попал в топ
        last_logits = last_logits.clone()
        last_logits[item_pad_idx] = float('-inf')

        # Top-k
        top_k = torch.topk(last_logits, k).indices.tolist()
    return top_k
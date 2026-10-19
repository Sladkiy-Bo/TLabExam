"""
Reward-функции для RL. Каждая реализует интерфейс:
    reward_fn(contexts, actions) -> torch.Tensor (B,)

- contexts: list[list[item_id]] — истории пользователей (старые → новые)
- actions:  list[item_id]       — выбранные действия
"""

from abc import ABC, abstractmethod
import torch


class RewardFunction(ABC):
    """Базовый класс reward-функций."""

    @abstractmethod
    def __call__(self, contexts, actions):
        pass


class SyntheticReward(RewardFunction):
    """
    Для юнит-теста: reward = 1.0, если action ∈ target_set, иначе 0.0.

    Используется для проверки, что RL-петля работает корректно:
    после N шагов вероятность target_set в π_θ должна вырасти.
    """

    def __init__(self, target_set):
        self.target_set = set(int(x) for x in target_set)

    def __call__(self, contexts, actions):
        return torch.tensor(
            [1.0 if int(a) in self.target_set else 0.0 for a in actions],
            dtype=torch.float32,
        )


class RMReward(RewardFunction):
    """
    R0: сырой reward от Reward Model.

    reward(h, a) = sigmoid(RM(h + [a]))
    """

    def __init__(self, rm, item2idx, max_len=50, device='cuda'):
        self.rm = rm
        self.item2idx = item2idx
        self.max_len = max_len
        self.device = device

    @torch.no_grad()
    def __call__(self, contexts, actions):
        B = len(contexts)
        input_ids = torch.zeros(B, self.max_len, dtype=torch.long, device=self.device)

        for i, (ctx, act) in enumerate(zip(contexts, actions)):
            recent = ctx[-(self.max_len - 1):] if ctx else []
            indexed = [self.item2idx.get(int(x), 1) for x in recent]
            act_idx = self.item2idx.get(int(act), 1)
            seq = indexed + [act_idx]
            input_ids[i, -len(seq):] = torch.tensor(seq, device=self.device)

        logits = self.rm(input_ids)
        return torch.sigmoid(logits).cpu()


class ItemCenteredReward(RMReward):
    """
    R2: RM(h, a) − item_mean[a].

    Убирает per-item bias (популярность, длительность): RL поощряет
    «отклонение вверх от среднего для этого айтема», а не «просто
    популярное». Нейтрализует reward hacking на длине/популярности.
    """

    def __init__(self, rm, item2idx, item_mean_dict, default_mean=0.5, **kwargs):
        super().__init__(rm, item2idx, **kwargs)
        self.item_mean_dict = item_mean_dict
        self.default_mean = default_mean

    def __call__(self, contexts, actions):
        raw = super().__call__(contexts, actions)
        means = torch.tensor(
            [self.item_mean_dict.get(int(a), self.default_mean) for a in actions],
            dtype=torch.float32,
        )
        return raw - means


def build_item_mean_map(rm_index, idx2item):
    """
    Строит item_mean из train: средний reward для каждого item_id.

    Возвращает dict {item_id: mean_reward}. Используется в ItemCenteredReward.
    """
    import pandas as pd
    train_items = rm_index.items[rm_index.train_mask]
    train_rewards = rm_index.rewards[rm_index.train_mask]
    df = pd.DataFrame({'item': train_items, 'reward': train_rewards})
    means = df.groupby('item')['reward'].mean()
    return {idx2item[int(item_idx)]: float(m) for item_idx, m in means.items()}
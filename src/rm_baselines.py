"""
Бейзлайны для Reward Model.

Все предсказывают reward по простым правилам, без обучения.
Нужны, чтобы понять: наша обученная RM реально лучше или нет?
"""

import numpy as np
import pandas as pd


class GlobalMeanBaseline:
    """Предсказывает константу: среднее reward по train."""
    def __init__(self, rm_index):
        self.mean_reward = float(rm_index.rewards[rm_index.train_mask].mean())
    
    def predict(self, items, users):
        return np.full(len(items), self.mean_reward)


class ItemMeanBaseline:
    """Предсказывает среднее reward по айтему (из train). Fallback — глобальное среднее."""
    def __init__(self, rm_index):
        self.global_mean = float(rm_index.rewards[rm_index.train_mask].mean())
        
        train_items = rm_index.items[rm_index.train_mask]
        train_rewards = rm_index.rewards[rm_index.train_mask]
        
        df = pd.DataFrame({'item': train_items, 'reward': train_rewards})
        self.item_means = df.groupby('item')['reward'].mean().to_dict()
    
    def predict(self, items, users):
        return np.array([self.item_means.get(int(i), self.global_mean) for i in items])


class UserMeanBaseline:
    """Предсказывает среднее reward по пользователю (из train)."""
    def __init__(self, rm_index):
        self.global_mean = float(rm_index.rewards[rm_index.train_mask].mean())
        
        train_users = rm_index.users[rm_index.train_mask]
        train_rewards = rm_index.rewards[rm_index.train_mask]
        
        df = pd.DataFrame({'user': train_users, 'reward': train_rewards})
        self.user_means = df.groupby('user')['reward'].mean().to_dict()
    
    def predict(self, items, users):
        return np.array([self.user_means.get(int(u), self.global_mean) for u in users])


class AdditiveBaseline:
    """Предсказывает: item_mean + user_mean - global_mean."""
    def __init__(self, rm_index):
        self.global_mean = float(rm_index.rewards[rm_index.train_mask].mean())
        
        train_items = rm_index.items[rm_index.train_mask]
        train_users = rm_index.users[rm_index.train_mask]
        train_rewards = rm_index.rewards[rm_index.train_mask]
        
        df = pd.DataFrame({'item': train_items, 'user': train_users, 'reward': train_rewards})
        self.item_means = df.groupby('item')['reward'].mean().to_dict()
        self.user_means = df.groupby('user')['reward'].mean().to_dict()
    
    def predict(self, items, users):
        results = []
        for i, u in zip(items, users):
            item_m = self.item_means.get(int(i), self.global_mean)
            user_m = self.user_means.get(int(u), self.global_mean)
            pred = item_m + user_m - self.global_mean
            pred = np.clip(pred, 0.0, 1.0)  # reward ∈ [0,1]
            results.append(pred)
        return np.array(results)


def evaluate_baselines(rm_index, split='val'):
    """Считает MAE/RMSE для всех бейзлайнов."""
    mask = rm_index.val_mask if split == 'val' else rm_index.test_mask
    
    items = rm_index.items[mask]
    users = rm_index.users[mask]
    targets = rm_index.rewards[mask]
    
    print(f"Evaluating baselines on {split}: {len(targets):,} examples")
    
    baselines = {
        'Global mean': GlobalMeanBaseline(rm_index),
        'Item mean': ItemMeanBaseline(rm_index),
        'User mean': UserMeanBaseline(rm_index),
        'Additive': AdditiveBaseline(rm_index),
    }
    
    results = {}
    for name, baseline in baselines.items():
        preds = baseline.predict(items, users)
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        results[name] = {'mae': mae, 'rmse': rmse}
    
    return results
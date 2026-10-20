"""Независимый судья (LightGBM) для честной оценки RL-reward.

Идея: RM обучена на последовательностях и её можно «взломать» RL-политикой -
она может найти айтемы, где RM ошибается и завышает оценку. Судья обучен
на ДРУГИХ признаках (табличные: item_mean, user_mean, метаданные контента)
и использует другую архитектуру (LightGBM). Его оценка независима от RM.

Если судья видит тот же рост reward, что и RM -> улучшение настоящее.
Если судья не видит роста -> RM эксплуатируется RL (reward hacking).
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class JudgeModel:
    """LightGBM-судья на табличных признаках айтема и пользователя."""

    def __init__(self, **lgb_params):
        self.model = None
        self.item_mean = {}
        self.user_mean = {}
        self.item_count = {}
        self.user_history_len = {}
        self.global_mean = 0.5
        self.items_en = None
        self._content_cats = None
        self._age_cats = None
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 1,
            **lgb_params,
        }

    #  Внутренние хелперы

    def _prepare_stats(self, train_df):
        self.global_mean = float(train_df['watched_pct'].mean() / 100.0)
        self.item_mean = (train_df.groupby('item_id')['watched_pct'].mean() / 100.0).to_dict()
        self.user_mean = (train_df.groupby('user_id')['watched_pct'].mean() / 100.0).to_dict()
        self.item_count = train_df.groupby('item_id').size().to_dict()
        self.user_history_len = train_df.groupby('user_id').size().to_dict()

    def _prepare_items_meta(self, items_df):
        # Ожидаемые колонки: item_id, content_type, release_year, is_series,
        # for_kids, age_rating. Если каких-то нет - 'unknown'.
        df = items_df.copy()
        for col in ['content_type', 'release_year', 'is_series', 'for_kids', 'age_rating']:
            if col not in df.columns:
                df[col] = 'unknown' if col in ['content_type', 'age_rating'] else 0
        self.items_en = df.set_index('item_id')

    def _extract_features(self, user_ids, item_ids):
        n = len(user_ids)
        feats = {
            'item_mean': np.zeros(n),
            'user_mean': np.zeros(n),
            'item_count_log': np.zeros(n),
            'history_len_log': np.zeros(n),
            'release_year': np.zeros(n),
            'is_series': np.zeros(n),
            'for_kids': np.zeros(n),
            'content_type': np.empty(n, dtype=object),
            'age_rating': np.empty(n, dtype=object),
        }
        item_index = self.items_en.index if self.items_en is not None else set()
        for i, (u, it) in enumerate(zip(user_ids, item_ids)):
            feats['item_mean'][i] = self.item_mean.get(it, self.global_mean)
            feats['user_mean'][i] = self.user_mean.get(u, self.global_mean)
            feats['item_count_log'][i] = np.log1p(self.item_count.get(it, 0))
            feats['history_len_log'][i] = np.log1p(self.user_history_len.get(u, 0))
            row = self.items_en.loc[it] if it in item_index else None
            feats['release_year'][i] = (float(row['release_year'])
                                        if row is not None and pd.notna(row['release_year'])
                                        else 2000.0)
            feats['is_series'][i] = (int(row['is_series'])
                                     if row is not None and pd.notna(row['is_series'])
                                     else 0)
            feats['for_kids'][i] = (int(row['for_kids'])
                                    if row is not None and pd.notna(row['for_kids'])
                                    else 0)
            feats['content_type'][i] = (str(row['content_type'])
                                        if row is not None and pd.notna(row['content_type'])
                                        else 'unknown')
            feats['age_rating'][i] = (str(row['age_rating'])
                                      if row is not None and pd.notna(row['age_rating'])
                                      else 'unknown')
        df = pd.DataFrame(feats)
        # Categoricals - с фиксированными категориями, чтобы fit и predict совпадали
        if self._content_cats is None:  # режим fit
            df['content_type'] = df['content_type'].astype('category')
            df['age_rating'] = df['age_rating'].astype('category')
            self._content_cats = df['content_type'].cat.categories
            self._age_cats = df['age_rating'].cat.categories
        else:
            df['content_type'] = pd.Categorical(df['content_type'], categories=self._content_cats)
            df['age_rating'] = pd.Categorical(df['age_rating'], categories=self._age_cats)
        return df

    #  Обучение / предсказание

    def fit(self, train_df, items_df, val_df=None, early_stopping=30):
        self._content_cats = None  # сброс на случай повторного fit
        self._prepare_stats(train_df)
        self._prepare_items_meta(items_df)

        X = self._extract_features(train_df['user_id'].values, train_df['item_id'].values)
        y = (train_df['watched_pct'] / 100.0).values

        self.model = lgb.LGBMRegressor(**self.lgb_params)
        if val_df is not None:
            X_val = self._extract_features(val_df['user_id'].values, val_df['item_id'].values)
            y_val = (val_df['watched_pct'] / 100.0).values
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
            )
        else:
            self.model.fit(X, y)
        return self

    def predict(self, user_ids, item_ids):
        X = self._extract_features(np.asarray(user_ids), np.asarray(item_ids))
        return self.model.predict(X)

    #  Метрики и оценка предсказаний

    def evaluate_on_df(self, df, sample_n=None):
        """MAE, RMSE, R² на произвольном DataFrame (user_id, item_id, watched_pct)."""
        if sample_n is not None and sample_n < len(df):
            df = df.sample(n=sample_n, random_state=42)
        X = self._extract_features(df['user_id'].values, df['item_id'].values)
        y = (df['watched_pct'] / 100.0).values
        preds = self.model.predict(X)
        mae = float(np.mean(np.abs(y - preds)))
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def score_predictions_topk(self, preds_dict, K=10):
        """
        Средний judge reward top-K по всем пользователям из preds_dict.
        preds_dict: {user_id: [item_id, ...]}
        """
        all_u, all_i = [], []
        for u, items in preds_dict.items():
            for it in items[:K]:
                all_u.append(u)
                all_i.append(it)
        if not all_u:
            return 0.0
        preds = self.predict(np.array(all_u), np.array(all_i))
        df = pd.DataFrame({'u': all_u, 'p': preds})
        return float(df.groupby('u')['p'].mean().mean())

    def mean_item_mean_topk(self, preds_dict, K=10):
        """
        Средний item_mean рекомендованных айтемов.
        Диагностика: если вырос по сравнению с base - RL сдвинулся в популярное.
        """
        vals = []
        for u, items in preds_dict.items():
            topk = items[:K]
            if not topk:
                continue
            vals.append(np.mean([self.item_mean.get(it, self.global_mean) for it in topk]))
        return float(np.mean(vals)) if vals else 0.0

    #  Сохранение / загрузка

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path.with_suffix('.txt')))
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({
                'item_mean': self.item_mean,
                'user_mean': self.user_mean,
                'item_count': self.item_count,
                'user_history_len': self.user_history_len,
                'global_mean': self.global_mean,
                'items_en': self.items_en,
                '_content_cats': self._content_cats,
                '_age_cats': self._age_cats,
                'lgb_params': self.lgb_params,
            }, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        obj = cls()
        obj.model = lgb.Booster(model_file=str(path.with_suffix('.txt')))
        with open(path.with_suffix('.pkl'), 'rb') as f:
            state = pickle.load(f)
        for k, v in state.items():
            setattr(obj, k, v)
        return obj
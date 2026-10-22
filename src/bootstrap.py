import numpy as np

def bootstrap_per_target(user_metrics, n_samples=1000, seed=1, chunk=256):
    """Bootstrap для метрики с per-target значениями.
    Возвращает (mean, [lo, hi]) 95% CI."""
    rng = np.random.default_rng(seed)
    users = list(user_metrics.keys())
    n = len(users)

    sums = np.fromiter((user_metrics[u].sum() for u in users),
                       dtype=np.float64, count=n)
    cnts = np.fromiter((len(user_metrics[u]) for u in users),
                       dtype=np.float64, count=n)

    point = sums.sum() / cnts.sum()

    boot_means = np.empty(n_samples, dtype=np.float64)
    for s in range(0, n_samples, chunk):
        e = min(s + chunk, n_samples)
        idx = rng.integers(0, n, size=(e - s, n))
        numer = sums[idx].sum(axis=1)
        denom = cnts[idx].sum(axis=1)
        boot_means[s:e] = numer / denom

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, (lo, hi)


def bootstrap_per_user(user_means, n_samples=1000, seed=1, chunk=1024):
    """Bootstrap для метрики с одним значением на пользователя."""
    rng = np.random.default_rng(seed)
    users = list(user_means.keys())
    vals = np.array([user_means[u] for u in users], dtype=np.float64)
    n = len(users)
    point = vals.mean()

    boot_means = np.empty(n_samples, dtype=np.float64)
    for s in range(0, n_samples, chunk):
        e = min(s + chunk, n_samples)
        idx = rng.integers(0, n, size=(e - s, n))
        boot_means[s:e] = vals[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, (lo, hi)


def paired_bootstrap_per_target(metrics_a, metrics_b, n_samples=1000, seed=1, chunk=256):
    """Paired bootstrap для разницы a - b (per-target)."""
    rng = np.random.default_rng(seed)
    users = [u for u in metrics_a if u in metrics_b]
    n = len(users)

    sums_a = np.fromiter((metrics_a[u].sum() for u in users),
                         dtype=np.float64, count=n)
    cnts_a = np.fromiter((len(metrics_a[u]) for u in users),
                         dtype=np.float64, count=n)
    sums_b = np.fromiter((metrics_b[u].sum() for u in users),
                         dtype=np.float64, count=n)
    cnts_b = np.fromiter((len(metrics_b[u]) for u in users),
                         dtype=np.float64, count=n)

    point = (sums_a.sum() / cnts_a.sum()) - (sums_b.sum() / cnts_b.sum())

    diffs = np.empty(n_samples, dtype=np.float64)
    for s in range(0, n_samples, chunk):
        e = min(s + chunk, n_samples)
        idx = rng.integers(0, n, size=(e - s, n))            # общий idx для a и b
        mean_a = sums_a[idx].sum(axis=1) / cnts_a[idx].sum(axis=1)
        mean_b = sums_b[idx].sum(axis=1) / cnts_b[idx].sum(axis=1)
        diffs[s:e] = mean_a - mean_b

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, (lo, hi)


def paired_bootstrap_per_user(means_a, means_b, n_samples=1000, seed=42, chunk=1024):
    """Paired bootstrap для разницы a - b (per-user)."""
    rng = np.random.default_rng(seed)
    users = [u for u in means_a if u in means_b]
    a = np.array([means_a[u] for u in users], dtype=np.float64)
    b = np.array([means_b[u] for u in users], dtype=np.float64)
    n = len(users)
    point = a.mean() - b.mean()

    diffs = np.empty(n_samples, dtype=np.float64)
    for s in range(0, n_samples, chunk):
        e = min(s + chunk, n_samples)
        idx = rng.integers(0, n, size=(e - s, n))            # общий idx для a и b
        diffs[s:e] = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, (lo, hi)


print("Bootstrap functions ready")
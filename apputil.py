# Re-run with a robust diamonds loader (offline fallback) and regenerate outputs.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as SKKMeans

# -----------------------------
# Robust loader for diamonds (numeric only)
# -----------------------------
def _load_diamonds_numeric():
    """
    Try to load seaborn's diamonds dataset; if offline, create a synthetic
    diamonds-like numeric dataframe with columns:
    ['carat','depth','table','price','x','y','z'] and ~53,940 rows.
    """
    # Attempt seaborn load
    try:
        import seaborn as sns
        df = sns.load_dataset("diamonds")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[num_cols].copy()
    except Exception:
        # Offline fallback: synthesize approximate distributions
        rng = np.random.default_rng(42)
        n = 53940

        # carat: skewed positive (lognormal), clipped to [0.1, 5.0]
        carat = np.clip(rng.lognormal(mean=-0.5, sigma=0.5, size=n), 0.1, 5.0)

        # depth: around 61% +/- 1.5
        depth = rng.normal(loc=61.0, scale=1.5, size=n)

        # table: around 57% +/- 2.5
        table = rng.normal(loc=57.0, scale=2.5, size=n)

        # price: roughly correlated with carat (nonlinear, noisy)
        base_price = (carat ** 1.5) * 4000
        price_noise = rng.normal(0, 500, size=n)
        price = np.clip(base_price + price_noise, 300, 19000)

        # x,y,z: correlated with carat; add noise and clip to plausible ranges
        x = np.clip(carat * 5.5 + rng.normal(0, 0.3, size=n), 3.0, 10.0)
        y = np.clip(carat * 5.3 + rng.normal(0, 0.3, size=n), 3.0, 10.5)
        z = np.clip(carat * 3.3 + rng.normal(0, 0.2, size=n), 2.0, 6.5)

        df_syn = pd.DataFrame({
            "carat": carat,
            "depth": depth,
            "table": table,
            "price": price,
            "x": x,
            "y": y,
            "z": z,
        })
        return df_syn

diamonds_numeric = _load_diamonds_numeric()

# -----------------------------
# Exercise 1
# -----------------------------

def kmeans(X: np.ndarray, k: int):
    X = np.asarray(X, dtype=np.float32)
    km = SKKMeans(
        n_clusters=k,
        n_init=1,
        max_iter=20,
        tol=1e-3,
        random_state=0,
        algorithm="lloyd"
    )
    labels = km.fit_predict(X)
    return km.cluster_centers_, labels


# -----------------------------
# Exercise 2
# -----------------------------
def kmeans_diamonds(n: int, k: int):
    """
    Run kmeans() on the first n rows of the numeric diamonds dataset.
    Returns (centroids, labels).
    """
    if n < 1 or n > len(diamonds_numeric):
        raise ValueError(f"n must be in [1, {len(diamonds_numeric)}]")
    X = diamonds_numeric.iloc[:n, :].to_numpy(dtype=np.float32, copy=True)
    return kmeans(X, k)

# -----------------------------
# Exercise 3
# -----------------------------
from time import time

def kmeans_timer(n: int, k: int, n_iter: int = 5) -> float:
    """
    Run kmeans_diamonds(n, k) exactly n_iter times and return average runtime (seconds).
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)
    return float(np.mean(times))

# -----------------------------
# Exercise Output (matplotlib-only; separate figures, no subplots)
# -----------------------------

# 1) Time vs n for k=5
n_values = np.arange(100, 50000, 1000)  # 100 .. 49,100
k5_times = [kmeans_timer(int(n), 5, 20) for n in n_values]

plt.figure(figsize=(7, 4))
plt.plot(n_values, k5_times)
plt.title("KMeans Time Complexity: Increasing n for k=5")
plt.xlabel("Number of Rows (n)")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.show()

# 2) Time vs k for n=10,000
k_values = np.arange(2, 50)  # 2 .. 49
n10k_times = [kmeans_timer(10000, int(k), 10) for k in k_values]

plt.figure(figsize=(7, 4))
plt.plot(k_values, n10k_times)
plt.title("KMeans Time Complexity: Increasing k for n = 10,000")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.show()

# -----------------------------
# Bonus: count steps in binary search (worst-case) and show complexity
# -----------------------------
step_count = 0

def bin_search_counted(n: int, target_position: str = "worst") -> int:
    """
    Binary search on arr=0..n-1 with a counted number of 'steps'.
    Counts comparisons and index updates to estimate asymptotic growth.
    target_position:
        - "worst": choose x that forces worst-case (not found)
        - "last": search for last element (n-1)
    Returns the index (or -1), and updates global step_count.
    """
    global step_count
    arr = np.arange(n)
    left, right = 0, n - 1

    if target_position == "last":
        x = n - 1
    else:
        x = n  # not present → worst-case

    while left <= right:
        step_count += 1  # loop condition check
        middle = left + (right - left) // 2
        step_count += 1  # middle calc

        step_count += 1  # compare ==
        if arr[middle] == x:
            return middle

        step_count += 1  # compare <
        if arr[middle] < x:
            left = middle + 1
            step_count += 1  # update left
        else:
            right = middle - 1
            step_count += 1  # update right

    step_count += 1  # final failed condition
    return -1

# Evaluate step counts across n
ns = np.array([2**i for i in range(5, 17)])  # 32 .. 65,536
steps_worst = []
steps_found_last = []
for n in ns:
    step_count = 0
    _ = bin_search_counted(int(n), target_position="worst")
    steps_worst.append(step_count)

    step_count = 0
    _ = bin_search_counted(int(n), target_position="last")
    steps_found_last.append(step_count)

steps_worst = np.array(steps_worst, dtype=float)
steps_found_last = np.array(steps_found_last, dtype=float)

plt.figure(figsize=(7, 4))
plt.plot(ns, steps_worst, label="Worst-case (not found) steps")
plt.plot(ns, steps_found_last, label="Found: last element steps")
# Compare against c·log2(n)
plt.plot(ns, np.log2(ns) * (steps_worst[0] / np.log2(ns[0])), linestyle="--", label="c · log2(n)")
plt.xscale("log")
plt.xlabel("n (log scale)")
plt.ylabel("Step Count")
plt.title("Binary Search Step Count vs n (Worst Case ≈ O(log n))")
plt.grid(True)
plt.legend()
plt.show()

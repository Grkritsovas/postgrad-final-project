import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


# core helpers
def _combined_bins(df: pd.DataFrame, q: int = 3) -> pd.Series:
    """
    Build combined stratification bins (valence x arousal) using quantiles.
    Falls back gracefully if qcut drops duplicate edges.
    Returns a string code per row: "{vbin}_{abin}"
    """
    v = df['valence_mean']
    a = df['arousal_mean']

    vbin = pd.qcut(v, q=q, labels=False, duplicates='drop')
    abin = pd.qcut(a, q=q, labels=False, duplicates='drop')

    # report actual number of bins via categories present
    return vbin.astype(str) + "_" + abin.astype(str)


def _l1_dist(p: pd.Series, q: pd.Series) -> float:
    """L1 distance between two discrete distributions on the union of indices."""
    keys = set(p.index).union(set(q.index))
    return float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def _group_stratified_split(
    ids: pd.Index,
    groups: pd.Series,
    bins: pd.Series,
    val_size: float,
    random_state: int,
    trials: int = 64
) -> Tuple[pd.Index, pd.Index]:
    """
    Group-aware split (no artist leakage) with iterative search to minimize
    distribution gap of 'bins' between train and val.
    """
    ids = ids.to_numpy()
    g = groups.loc[ids].fillna('UNKNOWN').to_numpy()
    b = bins.loc[ids]

    best = None
    best_score = np.inf

    # target is balanced train/val distributions
    for i in range(trials):
        rs = random_state + i
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=rs)
        train_idx, val_idx = next(gss.split(ids, groups=g))

        train_ids = ids[train_idx]
        val_ids = ids[val_idx]

        train_dist = b.loc[train_ids].value_counts(normalize=True)
        val_dist = b.loc[val_ids].value_counts(normalize=True)
        score = _l1_dist(train_dist, val_dist)

        if score < best_score:
            best_score = score
            best = (pd.Index(train_ids), pd.Index(val_ids))

    return best


# original split (fixed test by id threshold)

def create_original_split(
    core_df: pd.DataFrame,
    test_song_id_start: int = 2001, # ids >= 2001 go to test
    val_size: float = 0.15,
    random_state: int = 42,
    trials: int = 64
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Returns train_ids, val_ids, test_ids.

    - Test = ids >= test_song_id_start
    - Train/Val = remaining, group-aware, stratified on (valence, arousal)
    """
    df = core_df.copy()
    df.index = df.index.astype(int)

    all_ids = df.index
    test_ids = all_ids[all_ids >= test_song_id_start]
    pool_ids = all_ids[all_ids < test_song_id_start]

    if len(pool_ids) == 0:
        raise ValueError("No songs available for train/val split (all are test by threshold).")

    # stratification bins computed on the pool to match its distribution
    bins_pool = _combined_bins(df.loc[pool_ids])
    artists_pool = df.loc[pool_ids, 'artist_name']

    train_ids, val_ids = _group_stratified_split(
        ids=pool_ids,
        groups=artists_pool,
        bins=bins_pool,
        val_size=val_size,
        random_state=random_state,
        trials=trials
    )

    # make deterministic order
    return train_ids.sort_values(), val_ids.sort_values(), test_ids.sort_values()


# augmented split

def create_augmented_split(
    core_df: pd.DataFrame,
    target_train: float = 0.70,
    target_val: float = 0.15,
    target_test: float = 0.15,
    test_song_id_start: int = 2001,
    random_state: int = 42,
    trials_borrow: int = 64,
    trials_tv: int = 64
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Enlarges the test set while preserving:
    - no artist leakage
    - reasonable valence/arousal balance (iterative search for borrowed set)
    """
    assert abs((target_train + target_val + target_test) - 1.0) < 1e-6, "Targets must sum to 1."

    df = core_df.copy()
    df.index = df.index.astype(int)

    all_ids = df.index
    original_test = all_ids[all_ids >= test_song_id_start]
    pool_ids = all_ids[all_ids < test_song_id_start]

    # how many test items do we want?
    n_total = len(all_ids)
    n_target_test = int(round(n_total * target_test))
    n_to_borrow = n_target_test - len(original_test)

    if n_to_borrow <= 0:
        # Target smaller or equal to existing test: fall back to original + standard train/val
        return create_original_split(
            core_df=df,
            test_song_id_start=test_song_id_start,
            val_size=target_val,
            random_state=random_state,
            trials=trials_tv
        )

    # build bins on the pool and global for balance scoring
    bins_global = _combined_bins(df)
    bins_pool = bins_global.loc[pool_ids]
    artists_pool = df.loc[pool_ids, 'artist_name'].fillna('UNKNOWN')

    # iterative search: select borrowed subset via GroupShuffleSplit that best matches
    # the global test distribution (by bins) when combined with original test
    best_borrowed = None
    best_score = np.inf

    # proportion for borrowing; rounding is ok (group-level sampling)
    borrow_prop = min(1.0, max(0.0, n_to_borrow / max(len(pool_ids), 1)))

    for i in range(trials_borrow):
        rs = random_state + 1000 + i
        gss = GroupShuffleSplit(n_splits=1, test_size=borrow_prop, random_state=rs)
        keep_idx, borrow_idx = next(gss.split(pool_ids, groups=artists_pool.values))

        borrowed_ids = pd.Index(pool_ids[borrow_idx])
        test_candidate = original_test.union(borrowed_ids)

        # score: how close is test_candidate to global distribution?
        test_dist = bins_global.loc[test_candidate].value_counts(normalize=True)
        global_dist = bins_global.value_counts(normalize=True)
        score = _l1_dist(test_dist, global_dist)

        # prefer sizes closer to n_target_test if distributions tie
        size_penalty = abs(len(test_candidate) - n_target_test) / max(1, n_total)
        total_score = score + 0.1 * size_penalty

        if total_score < best_score:
            best_score = total_score
            best_borrowed = borrowed_ids

    final_test = original_test.union(best_borrowed)

    # remaining pool for train/val
    remaining = all_ids.difference(final_test)

    # val proportion within remaining to hit global target_val
    # (remaining size fraction is 1 - target_test)
    val_within_remaining = target_val / max(1e-9, (1.0 - target_test))

    bins_remaining = _combined_bins(df.loc[remaining])
    artists_remaining = df.loc[remaining, 'artist_name']

    train_ids, val_ids = _group_stratified_split(
        ids=remaining,
        groups=artists_remaining,
        bins=bins_remaining,
        val_size=val_within_remaining,
        random_state=random_state,
        trials=trials_tv
    )

    return train_ids.sort_values(), val_ids.sort_values(), final_test.sort_values()

# reusable K-fold splits (5 folds seem better, but with 3 quantized bins, cause of sparse regions)

def create_kfold_splits(
    core_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    StratifiedGroupKFold on (valence, arousal) bins with artists as groups.
    Returns list of (train_ids, val_ids) per fold.
    """
    df = core_df.copy()
    df.index = df.index.astype(int)

    bins = _combined_bins(df)
    artists = df['artist_name'].fillna('UNKNOWN')

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    splits: List[Tuple[pd.Index, pd.Index]] = []
    X = np.zeros((len(df), 1))  # placeholder; groups/y drive the split
    for tr_idx, va_idx in sgkf.split(X, y=bins, groups=artists):
        splits.append((df.index[tr_idx].sort_values(), df.index[va_idx].sort_values()))
    return splits


# save/load helpers

def save_splits_triplet(
    train_ids: pd.Index,
    val_ids: pd.Index,
    test_ids: pd.Index,
    out_dir: str,
    name: str,
    meta: Optional[Dict] = None
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    pd.Series(train_ids, name='song_id').to_csv(os.path.join(out_dir, f'{name}_train.csv'), index=False)
    pd.Series(val_ids,   name='song_id').to_csv(os.path.join(out_dir, f'{name}_val.csv'),   index=False)
    pd.Series(test_ids,  name='song_id').to_csv(os.path.join(out_dir, f'{name}_test.csv'),  index=False)

    manifest = {
        "type": "triplet",
        "name": name,
        "counts": {"train": int(len(train_ids)), "val": int(len(val_ids)), "test": int(len(test_ids))},
        "meta": meta or {}
    }
    with open(os.path.join(out_dir, f'{name}_meta.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


def load_splits_triplet(out_dir: str = "../data/splits", name: str = 'custom') -> Tuple[pd.Index, pd.Index, pd.Index]:
    train = pd.read_csv(os.path.join(out_dir, f'{name}_train.csv'))['song_id']
    val   = pd.read_csv(os.path.join(out_dir, f'{name}_val.csv'))['song_id']
    test  = pd.read_csv(os.path.join(out_dir, f'{name}_test.csv'))['song_id']
    return pd.Index(train), pd.Index(val), pd.Index(test)


def save_kfold_splits(
    folds: List[Tuple[pd.Index, pd.Index]],
    out_dir: str,
    name: str,
    meta: Optional[Dict] = None
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, (tr, va) in enumerate(folds):
        pd.Series(tr, name='song_id').to_csv(os.path.join(out_dir, f'{name}_fold{i}_train.csv'), index=False)
        pd.Series(va, name='song_id').to_csv(os.path.join(out_dir, f'{name}_fold{i}_val.csv'),   index=False)

    manifest = {
        "type": "kfold",
        "name": name,
        "k": len(folds),
        "fold_counts": [{"train": int(len(tr)), "val": int(len(va))} for tr, va in folds],
        "meta": meta or {}
    }
    with open(os.path.join(out_dir, f'{name}_meta.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


def load_kfold_splits(out_dir: str = "../data/splits", name: str = 'cv5', k: int = 5) -> List[Tuple[pd.Index, pd.Index]]:
    out = []
    for i in range(k):
        tr = pd.read_csv(os.path.join(out_dir, f'{name}_fold{i}_train.csv'))['song_id']
        va = pd.read_csv(os.path.join(out_dir, f'{name}_fold{i}_val.csv'))['song_id']
        out.append((pd.Index(tr), pd.Index(va)))
    return out


# quick analysis

def analyze_split(core_df: pd.DataFrame, train_ids, val_ids, test_ids, title=""):
    all_ids = core_df.index
    def pct(n): return 100 * n / max(1, len(all_ids))
    print(f"\n--- {title} ---")
    print(f"Total: {len(all_ids)}")
    print(f"Train: {len(train_ids)} ({pct(len(train_ids)):.1f}%) | "
          f"Val: {len(val_ids)} ({pct(len(val_ids)):.1f}%) | "
          f"Test: {len(test_ids)} ({pct(len(test_ids)):.1f}%)")

    for col in ["valence_mean", "arousal_mean"]:
        print(f"{col}: "
              f"train μ={core_df.loc[train_ids, col].mean():.3f}, "
              f"val μ={core_df.loc[val_ids, col].mean():.3f}, "
              f"test μ={core_df.loc[test_ids, col].mean():.3f}")

    train_art = set(core_df.loc[train_ids, 'artist_name'].dropna())
    val_art   = set(core_df.loc[val_ids,   'artist_name'].dropna())
    test_art  = set(core_df.loc[test_ids,  'artist_name'].dropna())
    print(f"Artist overlap | T∩V: {len(train_art & val_art)}, "
          f"T∩Te: {len(train_art & test_art)}, "
          f"V∩Te: {len(val_art & test_art)}")

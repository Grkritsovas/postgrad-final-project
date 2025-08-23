import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple

def split_train_val_test_both_targets(
    X: pd.DataFrame,
    y_valence: pd.Series,
    y_arousal: pd.Series,
    labels_df: pd.DataFrame,  # contains artist_name
    test_song_id_start: int = 2001,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify_both: bool = True
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Split considering both valence and arousal distributions.
    """
    
    # Ensure consistent types
    X.index = X.index.astype(int)
    y_valence.index = y_valence.index.astype(int)
    y_arousal.index = y_arousal.index.astype(int)
    labels_df.index = labels_df.index.astype(int)
    
    # Get common songs
    common_ids = (X.index
                  .intersection(y_valence.index)
                  .intersection(y_arousal.index)
                  .intersection(labels_df.index))
    
    print(f"Total songs available: {len(common_ids)}")
    
    # Predefined test split
    test_ids = common_ids[common_ids >= test_song_id_start]
    train_val_pool = common_ids[common_ids < test_song_id_start]
    
    print(f"Test set (song_id >= {test_song_id_start}): {len(test_ids)} songs")
    print(f"Train+Val pool: {len(train_val_pool)} songs")
    
    if len(train_val_pool) == 0:
        raise ValueError("No songs for train/val split!")
    
    # Get artist groups
    pool_artists = labels_df.loc[train_val_pool, 'artist_name'].fillna('UNKNOWN')
    
    if stratify_both:
        # Create combined stratification based on both valence and arousal
        val_pool = y_valence.loc[train_val_pool]
        aro_pool = y_arousal.loc[train_val_pool]
        
        # Bin both dimensions and combine
        val_bins = pd.qcut(val_pool, q=5, labels=False, duplicates='drop')
        aro_bins = pd.qcut(aro_pool, q=5, labels=False, duplicates='drop')
        
        # Create combined bins (e.g., "val_bin2_aro_bin3")
        combined_bins = val_bins.astype(str) + '_' + aro_bins.astype(str)
        
        print(f"Created stratification with {combined_bins.nunique()} combined bins")
        
        # Try to balance the combined bins across train/val
        # Note: GroupShuffleSplit doesn't directly support stratification,
        # but we can try multiple splits and pick the most balanced one
        
        best_split = None
        best_balance_score = float('inf')
        
        for seed in range(random_state, random_state + 10):  # Try 10 different splits
            gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            train_idx, val_idx = next(gss.split(train_val_pool, groups=pool_artists))
            
            train_ids_candidate = train_val_pool[train_idx]
            val_ids_candidate = train_val_pool[val_idx]
            
            # Calculate balance score for this split
            train_bins = combined_bins.loc[train_ids_candidate]
            val_bins = combined_bins.loc[val_ids_candidate]
            
            # Calculate distribution differences
            train_dist = train_bins.value_counts(normalize=True)
            val_dist = val_bins.value_counts(normalize=True)
            
            # Align distributions and calculate difference
            all_bins = set(train_dist.index).union(set(val_dist.index))
            balance_score = 0
            for bin_name in all_bins:
                train_prop = train_dist.get(bin_name, 0)
                val_prop = val_dist.get(bin_name, 0)
                balance_score += abs(train_prop - val_prop)
            
            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_split = (train_ids_candidate, val_ids_candidate)
        
        train_ids, val_ids = best_split
        print(f"Selected split with balance score: {best_balance_score:.3f}")
        
    else:
        # Simple split without stratification
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(gss.split(train_val_pool, groups=pool_artists))
        train_ids = train_val_pool[train_idx]
        val_ids = train_val_pool[val_idx]
    
    # Final stats
    print(f"Final split:")
    print(f"  Train: {len(train_ids)} songs ({len(train_ids)/len(common_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} songs ({len(val_ids)/len(common_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} songs ({len(test_ids)/len(common_ids)*100:.1f}%)")
    
    # Check distributions
    if stratify_both:
        print(f"\nValence distribution check:")
        train_val_mean = y_valence.loc[train_ids].mean()
        val_val_mean = y_valence.loc[val_ids].mean()
        test_val_mean = y_valence.loc[test_ids].mean()
        print(f"  Train valence mean: {train_val_mean:.3f}")
        print(f"  Val valence mean:   {val_val_mean:.3f}")
        print(f"  Test valence mean:  {test_val_mean:.3f}")
        
        print(f"\nArousal distribution check:")
        train_aro_mean = y_arousal.loc[train_ids].mean()
        val_aro_mean = y_arousal.loc[val_ids].mean()
        test_aro_mean = y_arousal.loc[test_ids].mean()
        print(f"  Train arousal mean: {train_aro_mean:.3f}")
        print(f"  Val arousal mean:   {val_aro_mean:.3f}")
        print(f"  Test arousal mean:  {test_aro_mean:.3f}")
    
    # Check artist leakage
    train_artists = set(labels_df.loc[train_ids, 'artist_name'].dropna())
    val_artists = set(labels_df.loc[val_ids, 'artist_name'].dropna())
    test_artists = set(labels_df.loc[test_ids, 'artist_name'].dropna())
    
    overlap = train_artists.intersection(val_artists)
    if overlap:
        print(f"{len(overlap)} artists in both train and val")
    else:
        print("No artist leakage between train and val")
    
    return train_ids, val_ids, test_ids


# USAGE:
# train_ids, val_ids, test_ids = split_train_val_test_both_targets(
#     X, y_valence, y_arousal, labels,
#     test_song_id_start=2001,
#     stratify_both=True  # Balance both valence and arousal
# )
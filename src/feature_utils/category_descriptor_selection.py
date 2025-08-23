"""
Category-specific descriptor selection for audio features.

This module selects the optimal k descriptors for EACH category independently,
ensuring all categories have the same number of descriptors while maximizing
their individual predictive power.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor


#-------- Core Selection Functions

def evaluate_descriptor_for_category(
    agg_df: pd.DataFrame,
    y: pd.Series,
    category: str,
    category_bases: List[str],
    descriptor: str,
    method: str = 'correlation'
) -> float:
    """
    Evaluate how well a descriptor performs for a specific category.
    
    Args:
        agg_df: Aggregated features DataFrame
        y: Target variable
        category: Category name
        category_bases: Base feature names in this category
        descriptor: Descriptor to evaluate
        method: 'correlation' or 'rf' (random forest)
    
    Returns:
        Performance score (higher is better)
    """
    # Collect features for this descriptor
    features = []
    for base in category_bases:
        feature_name = f"{base}_{descriptor}"
        if feature_name in agg_df.columns:
            features.append(feature_name)
    
    if not features:
        return 0.0
    
    # Create category feature by averaging
    category_feature = agg_df[features].mean(axis=1)
    
    # Align with target
    common_idx = category_feature.index.intersection(y.index)
    if len(common_idx) < 10:
        return 0.0
    
    aligned_feature = category_feature.loc[common_idx]
    aligned_y = y.loc[common_idx]
    
    if method == 'correlation':
        # Simple correlation score
        return abs(aligned_feature.corr(aligned_y))
    
    elif method == 'rf':
        # Use RF with cross-validation
        X = aligned_feature.values.reshape(-1, 1)
        X_train, X_val, y_train, y_val = train_test_split(
            X, aligned_y, test_size=0.3, random_state=42
        )
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, pred)
        
        # Convert RMSE to score (lower RMSE = higher score)
        # Normalize by target std to make it comparable
        score = 1 / (1 + rmse / aligned_y.std())
        return score
    
    return 0.0


def select_top_k_descriptors_per_category(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    k: int = 3,
    method: str = 'rf',
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Select top-k descriptors for EACH category independently.
    
    Args:
        agg_df: Aggregated features
        y: Target variable
        hierarchy_map: Feature hierarchy mapping
        k: Number of descriptors to select per category
        method: Evaluation method ('correlation' or 'rf')
        verbose: Print progress
        
    Returns:
        Dictionary mapping category -> list of selected descriptors
    """
    # Get available descriptors
    available_descriptors = set()
    for col in agg_df.columns:
        if '_' in col:
            descriptor = col.rsplit('_', 1)[-1]
            available_descriptors.add(descriptor)
    available_descriptors = sorted(list(available_descriptors))
    
    if verbose:
        print(f"Selecting top-{k} descriptors for each category")
        print(f"Available descriptors: {available_descriptors}")
        print(f"Evaluation method: {method}")
        print("-" * 60)
    
    category_selections = {}
    
    for category in sorted(hierarchy_map['level1'].unique()):
        # Get base features for this category
        category_bases = hierarchy_map[
            hierarchy_map['level1'] == category
        ]['feature_name'].tolist()
        
        if verbose:
            print(f"\n{category} ({len(category_bases)} base features):")
        
        # Evaluate each descriptor for this category
        descriptor_scores = {}
        for descriptor in available_descriptors:
            score = evaluate_descriptor_for_category(
                agg_df, y, category, category_bases, descriptor, method
            )
            descriptor_scores[descriptor] = score
            
            if verbose and score > 0:
                print(f"  {descriptor}: {score:.4f}")
        
        # Select top-k descriptors
        sorted_descriptors = sorted(
            descriptor_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected = [desc for desc, score in sorted_descriptors[:k] if score > 0]
        
        # If we don't have k good descriptors, pad with best available
        if len(selected) < k:
            remaining = [desc for desc, _ in sorted_descriptors 
                        if desc not in selected][:k-len(selected)]
            selected.extend(remaining)
        
        category_selections[category] = selected[:k]
        
        if verbose:
            print(f"  → Selected: {category_selections[category]}")
    
    return category_selections

#-------- Feature Building Functions

# smoothened version, averaging all low-level features inside the mid-level (suboptimal approach-baseline)
def create_features_with_category_selections(
    agg_df: pd.DataFrame,
    hierarchy_map: pd.DataFrame,
    category_selections: Dict[str, List[str]],
    aggregation_levels: List[str] = ['level1']
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build:
      - low_level_df: selected base×descriptor columns
      - mid_level_dfs: {level_name -> mid-level df averaged within that level}
    Notes:
      - category_selections are keyed by level1 categories
      - Handles any subset of ['level1','level2'] present in hierarchy_map.
    """
    if 'feature_name' not in hierarchy_map.columns:
        raise ValueError("hierarchy_map must contain 'feature_name'.")

    # map base -> row of hierarchy (for fast lookup)
    hmap = hierarchy_map.set_index('feature_name')

    # collect selected low-level cols (dedupe, keep order)
    selected = []
    for cat, descs in category_selections.items():
        bases = hierarchy_map[hierarchy_map['level1'] == cat]['feature_name'].tolist()
        for base in bases:
            for d in descs:
                col = f"{base}_{d}"
                if col in agg_df.columns:
                    selected.append(col)
    # stable unique
    seen = set()
    selected_low = [c for c in selected if not (c in seen or seen.add(c))]

    low_level_df = agg_df[selected_low].copy()

    # build requested mid-level frames
    mid_level_dfs: Dict[str, pd.DataFrame] = {}
    for level in aggregation_levels:
        if level not in hmap.columns:
            # skip silently if that level is not in the map
            continue

        groups: Dict[str, List[str]] = {}
        for col in low_level_df.columns:
            base = col.rsplit('_', 1)[0]
            if base in hmap.index:
                grp = hmap.loc[base, level]
                groups.setdefault(grp, []).append(col)

        feat = {grp: low_level_df[cols].mean(axis=1) for grp, cols in groups.items() if cols}
        mid_level_dfs[level] = pd.DataFrame(feat)

    return low_level_df, mid_level_dfs

# optimal approach to build the mid and high levels
def build_per_base_X(
    agg_df: pd.DataFrame,
    hierarchy_map: pd.DataFrame,
    category_selections: Dict[str, List[str]],
) -> pd.DataFrame:
    """One column per selected base×descriptor (no category averaging)."""
    cols, seen = [], set()
    for cat, descs in category_selections.items():
        bases = hierarchy_map[hierarchy_map['level1'] == cat]['feature_name'].tolist()
        for b in bases:
            for d in descs:
                c = f"{b}_{d}"
                if c in agg_df.columns and c not in seen:
                    cols.append(c); seen.add(c)
    return agg_df[cols].copy()

def create_level2_features(
    selected_low_level_df: pd.DataFrame,
    hierarchy_map: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates selected low-level features into more granular mid-level 
    features based on their 'level2' category.
    
    Args:
        selected_low_level_df: DataFrame with selected low-level features
        hierarchy_map: Hierarchy mapping with 'level2' column
        
    Returns:
        DataFrame with level2 mid-level features
    """
    # Check if level2 exists in hierarchy_map
    if 'level2' not in hierarchy_map.columns:
        print("Warning: 'level2' not found in hierarchy_map, returning empty DataFrame")
        return pd.DataFrame()
    
    # Create mapping from base feature name to level2 category
    feature_to_level2 = {}
    for _, row in hierarchy_map.iterrows():
        feature_to_level2[row['feature_name']] = row['level2']
    
    # Group columns by their level2 category
    level2_groups = {}
    for col in selected_low_level_df.columns:
        # Extract base name (remove descriptor suffix)
        base_name = col.rsplit('_', 1)[0] if '_' in col else col
        
        if base_name in feature_to_level2:
            level2_cat = feature_to_level2[base_name]
            if level2_cat not in level2_groups:
                level2_groups[level2_cat] = []
            level2_groups[level2_cat].append(col)
    
    # Create level2 features by averaging within each group
    level2_features = {}
    for level2_cat, cols in level2_groups.items():
        if cols:
            level2_features[level2_cat] = selected_low_level_df[cols].mean(axis=1)
    
    return pd.DataFrame(level2_features)

def pca_within_level2(selected_low_df: pd.DataFrame, hierarchy_map: pd.DataFrame, var_keep=0.8, random_state=42) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    fmap = hierarchy_map.set_index('feature_name')
    groups = {}
    for col in selected_low_df.columns:
        base = col.rsplit('_', 1)[0]
        if base in fmap.index:
            g = fmap.loc[base, 'level2']
            groups.setdefault(g, []).append(col)
    out = {}
    for g, cols in groups.items():
        Xg = selected_low_df[cols].fillna(0).values
        if Xg.shape[1] == 1:
            out[f"{g}_pc1"] = selected_low_df[cols[0]]
            continue
        pca = PCA(random_state=random_state)
        Z = pca.fit_transform(Xg)
        k = int(np.searchsorted(pca.explained_variance_ratio_.cumsum(), var_keep) + 1)
        for i in range(k):
            out[f"{g}_pc{i+1}"] = Z[:, i]
    return pd.DataFrame(out, index=selected_low_df.index)

#-------- Utility Functions

def make_sample_weights(labels_df: pd.DataFrame, id_index: pd.Index, std_col: str, scheme="inv_var", eps=1e-6) -> pd.Series:
    stds = labels_df.loc[id_index, std_col].astype(float)
    if scheme == "inv_var":
        w = 1.0 / (stds**2 + eps)
    elif scheme == "inv":
        w = 1.0 / (stds + eps)
    else:  # 'soft'
        w = 1.0 / (1.0 + stds)
    return (w / w.mean()).astype(float)

def rmse_cv_with_weights(model, X, y, sample_weight=None, cv=5, random_state=42):
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rmses = []
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        wtr = None if sample_weight is None else sample_weight.iloc[tr]
        model.fit(Xtr, ytr, sample_weight=wtr)
        pred = model.predict(Xte)
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
    return np.array(rmses)

def pick_best_k(results_df: pd.DataFrame) -> int:
    return int(results_df.loc[results_df['rmse_mean'].idxmin(), 'k'])

#--------- Analysis functions

def cv_rmse_across_k(
    agg_df, y, hierarchy_map, ks=range(1,9),
    cv=5, random_state=42, method='rf',
    aggregation_mode='mean_level2',  # 'mean_level2' | 'per_base' | 'pca_level2'
):
    rows = []
    for k in ks:
        sel = select_top_k_descriptors_per_category(agg_df, y, hierarchy_map, k=k, method=method, verbose=False)

        if aggregation_mode == 'per_base':
            X = build_per_base_X(agg_df, hierarchy_map, sel).fillna(0)
        elif aggregation_mode == 'pca_level2':
            low_df = build_per_base_X(agg_df, hierarchy_map, sel).fillna(0)
            X = pca_within_level2(low_df, hierarchy_map, var_keep=0.8)
        else:  # mean_level2
            _, mids = create_features_with_category_selections(agg_df, hierarchy_map, sel, aggregation_levels=['level2'])
            X = mids['level2'].fillna(0)

        model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        scores = -cross_val_score(model, X, y.loc[X.index], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        rows.append({"k": k, "rmse_mean": scores.mean(), "rmse_std": scores.std()})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

# weighted version based on the stds of the labels
def cv_rmse_across_k_weighted(
    agg_df, y, hierarchy_map, labels_df, std_col,
    ks=range(1,9),method='rf', cv=5, random_state=42, aggregation_mode='per_base'
):
    rows = []
    for k in ks:
        sel = select_top_k_descriptors_per_category(agg_df, y, hierarchy_map, k=k, method=method, verbose=False)
        if aggregation_mode == 'per_base':
            X = build_per_base_X(agg_df, hierarchy_map, sel).fillna(0)
        elif aggregation_mode == 'pca_level2':
            low_df = build_per_base_X(agg_df, hierarchy_map, sel).fillna(0)
            X = pca_within_level2(low_df, hierarchy_map, var_keep=0.8)
        else:
            _, mids = create_features_with_category_selections(agg_df, hierarchy_map, sel, aggregation_levels=['level2'])
            X = mids['level2'].fillna(0)

        w = make_sample_weights(labels_df, X.index, std_col=std_col, scheme="inv_var")
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        scores = rmse_cv_with_weights(model, X, y.loc[X.index], sample_weight=w, cv=cv, random_state=random_state)
        rows.append({"k": k, "rmse_mean": scores.mean(), "rmse_std": scores.std()})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

def permutation_rmse(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    k_values=(1,3),
    method = 'rf',
    n_perm: int = 50,
    cv: int = 5,
    random_state: int = 42,
    aggregation_level: str = 'level2'
) -> Dict[int, np.ndarray]:
    """
    Shuffle labels n_perm times; record mean CV RMSE for each k.
    Returns: {k: np.array of length n_perm}
    """
    rng = np.random.RandomState(random_state)
    model = RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1)
    out = {int(k): [] for k in k_values}

    for _ in range(n_perm):
        y_perm = pd.Series(shuffle(y.values, random_state=rng), index=y.index)
        for k in k_values:
            sel = select_top_k_descriptors_per_category(
                agg_df, y_perm, hierarchy_map, k=int(k), method=method, verbose=False
            )
            _, mids = create_features_with_category_selections(
                agg_df, hierarchy_map, sel, aggregation_levels=[aggregation_level]
            )
            if aggregation_level not in mids:
                continue
            X = mids[aggregation_level].fillna(0)
            scores = -cross_val_score(model, X, y_perm, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
            out[int(k)].append(scores.mean())

    return {k: np.array(v) for k, v in out.items()}

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def model_family_rmse_across_k(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    ks: range = range(1,9),
    method='rf',
    cv: int = 5,
    aggregation_level: str = 'level2'
) -> Dict[str, pd.DataFrame]:
    """
    Compare Ridge and GBR across k.
    Returns: {"Ridge": df, "GBR": df} each with [k, rmse_mean, rmse_std]
    """
    models = {
        "Ridge": RidgeCV(alphas=np.logspace(-3,3,13)),
        "GBR": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        rows = []
        for k in ks:
            sel = select_top_k_descriptors_per_category(
                agg_df, y, hierarchy_map, k=k, method=method, verbose=False
            )
            _, mids = create_features_with_category_selections(
                agg_df, hierarchy_map, sel, aggregation_levels=[aggregation_level]
            )
            if aggregation_level not in mids:
                raise ValueError(f"aggregation_level '{aggregation_level}' not found.")
            X = mids[aggregation_level].fillna(0)

            scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
            rows.append({"k": k, "rmse_mean": scores.mean(), "rmse_std": scores.std()})
        results[name] = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    return results

def shap_summary_for_k(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    k: int = 3,
    method: str = 'rf', # or 'correlation', for selecting top k descriptors per category
    mode: str = 'level2',  # 'level2' or 'per_base'
    aggregation_level: str = 'level2',
    n_estimators: int = 500,
    random_state: int = 42,
    sample_n: int = 800,
    sample_weight: Optional[pd.Series] = None
):
    """Fit RF on requested level, return (rf, X_used, mean_abs_shap, order_index, explainer)"""
    import shap
    from sklearn.ensemble import RandomForestRegressor

    sel = select_top_k_descriptors_per_category(
        agg_df, y, hierarchy_map, k=k, method= method, verbose=False
    )
    
    if mode == 'level2':
        # averaged features
        _, mids = create_features_with_category_selections(
            agg_df, hierarchy_map, sel, aggregation_levels=[aggregation_level]
        )
        X = mids[aggregation_level].fillna(0)
    elif mode == 'per_base':
        # keep individual features
        X = build_per_base_X(agg_df, hierarchy_map, sel).fillna(0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Train model
    y_aligned = y.loc[X.index]
    w = sample_weight.loc[X.index] if sample_weight is not None else None
    
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X, y_aligned, sample_weight=w)

    # SHAP
    background = X.sample(min(sample_n, len(X)), random_state=random_state)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(background)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    
    # If per_base mode, aggregate SHAP values by level2
    if mode == 'per_base':
        # Create mapping from feature to level2
        feature_to_level2 = {}
        for col in X.columns:
            base_name = col.rsplit('_', 1)[0]
            # Find this base in hierarchy_map
            level2 = hierarchy_map[hierarchy_map['feature_name'] == base_name]['level2'].values
            if len(level2) > 0:
                feature_to_level2[col] = level2[0]
            else:
                feature_to_level2[col] = 'Unknown'
        
        # Aggregate SHAP values by level2 group
        grouped_shap = {}
        for i, col in enumerate(X.columns):
            group = feature_to_level2[col]
            if group not in grouped_shap:
                grouped_shap[group] = 0
            grouped_shap[group] += mean_abs[i]
        
        # Convert to sorted series
        grouped_shap = pd.Series(grouped_shap).sort_values(ascending=False)
        
        return rf, X, mean_abs, order, explainer, grouped_shap
    
    return rf, X, mean_abs, order, explainer

#--------- Visualization Functions

def visualize_category_selections(
    category_selections: Dict[str, List[str]],
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    method = 'rf'
):
    """Visualize the selected descriptors for each category."""
    all_descriptors = set()
    for descs in category_selections.values():
        all_descriptors.update(descs)
    all_descriptors = sorted(list(all_descriptors))
    
    categories = sorted(category_selections.keys())
    
    selection_matrix = pd.DataFrame(0, index=categories, columns=all_descriptors)
    for cat, descs in category_selections.items():
        for desc in descs:
            if desc in selection_matrix.columns:
                selection_matrix.loc[cat, desc] = 1
    
    performance_matrix = pd.DataFrame(index=categories, columns=all_descriptors)
    for category in categories:
        category_bases = hierarchy_map[
            hierarchy_map['level1'] == category
        ]['feature_name'].tolist()
        
        for descriptor in all_descriptors:
            score = evaluate_descriptor_for_category(
                agg_df, y, category, category_bases, 
                descriptor, method=method
            )
            performance_matrix.loc[category, descriptor] = score
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Selection matrix
    sns.heatmap(
        selection_matrix, cmap='RdBu_r', center=0.5, annot=True, 
        fmt='g', cbar_kws={'label': 'Selected'}, ax=axes[0]
    )
    axes[0].set_title('Selected Descriptors by Category')
    axes[0].set_xlabel('Descriptors')
    axes[0].set_ylabel('Categories')
    
    # 2. Performance heatmap
    sns.heatmap(
        performance_matrix.astype(float), cmap='YlOrRd', annot=True, 
        fmt='.3f', cbar_kws={'label': 'Score'}, ax=axes[1]
    )
    axes[1].set_title('Descriptor Performance by Category')
    axes[1].set_xlabel('Descriptors')
    axes[1].set_ylabel('Categories')
    
    plt.tight_layout()
    plt.show()
    
    return selection_matrix, performance_matrix

# Quick test function
def test_improvement(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    category_selections: Dict[str, List[str]],
    baseline_descriptors: List[str] = ['mean']
) -> None:
    """
    Test improvement over baseline (e.g., using only mean).
    """
    from sklearn.model_selection import cross_val_score
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Baseline: same descriptors for all
    baseline_features = {}
    for category in hierarchy_map['level1'].unique():
        category_bases = hierarchy_map[
            hierarchy_map['level1'] == category
        ]['feature_name'].tolist()
        
        category_features = []
        for base in category_bases:
            for desc in baseline_descriptors:
                feature_name = f"{base}_{desc}"
                if feature_name in agg_df.columns:
                    category_features.append(feature_name)
        
        if category_features:
            baseline_features[category] = agg_df[category_features].mean(axis=1)
    
    baseline_df = pd.DataFrame(baseline_features)
    
    # Optimized: category-specific selections
    # Should be:
    _, optimized_dfs = create_features_with_category_selections(
        agg_df, hierarchy_map, category_selections, aggregation_levels=['level1']
    )
    optimized_df = optimized_dfs['level1']
    
    # Align and evaluate
    common_idx = baseline_df.index.intersection(y.index)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Baseline performance
    X_baseline = baseline_df.loc[common_idx].fillna(0)
    y_aligned = y.loc[common_idx]
    baseline_scores = -cross_val_score(
        rf, X_baseline, y_aligned, cv=5, 
        scoring='neg_root_mean_squared_error'
    )
    
    # Optimized performance  
    X_optimized = optimized_df.loc[common_idx].fillna(0)
    optimized_scores = -cross_val_score(
        rf, X_optimized, y_aligned, cv=5,
        scoring='neg_root_mean_squared_error'
    )
    
    print(f"\nBaseline (using {baseline_descriptors}):")
    print(f"  Mean RMSE: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")
    
    print(f"\nOptimized (category-specific):")
    print(f"  Mean RMSE: {optimized_scores.mean():.4f} ± {optimized_scores.std():.4f}")
    
    improvement = (baseline_scores.mean() - optimized_scores.mean()) / baseline_scores.mean() * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    if improvement > 0:
        print("Category-specific selection improves performance!")
    else:
        print("Baseline performs better - may need different evaluation method")
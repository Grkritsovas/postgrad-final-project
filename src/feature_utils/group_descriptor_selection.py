"""
Category-specific descriptor selection for audio features.

This module selects the optimal k descriptors for EACH category independently,
ensuring all categories have the same number of descriptors while maximizing
their individual predictive power.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


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
    method: str = 'correlation',
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


def analyze_optimal_k(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    max_k: int = 5,
    method: str = 'correlation',
    n_estimators: int = 100,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, int]:
    """
    Find optimal number of descriptors (k) by testing different values.
    
    Returns:
        Tuple of (results_df, optimal_k)
    """
    results = []
    
    print("Testing different k values for descriptor selection...")
    print("=" * 60)
    
    for k in range(1, max_k + 1):
        print(f"\nTesting k={k}...")
        
        # Get category-specific selections
        selections = select_top_k_descriptors_per_category(
            agg_df, y, hierarchy_map, k=k, method=method, verbose=False
        )
        
        # Create mid-level features with these selections
        mid_level_features = {}
        total_features_used = 0
        
        for category, descriptors in selections.items():
            category_bases = hierarchy_map[
                hierarchy_map['level1'] == category
            ]['feature_name'].tolist()
            
            category_features = []
            for base in category_bases:
                for desc in descriptors:
                    feature_name = f"{base}_{desc}"
                    if feature_name in agg_df.columns:
                        category_features.append(feature_name)
            
            if category_features:
                mid_level_features[category] = agg_df[category_features].mean(axis=1)
                total_features_used += len(category_features)
        
        if not mid_level_features:
            continue
            
        # Evaluate performance
        mid_df = pd.DataFrame(mid_level_features)
        common_idx = mid_df.index.intersection(y.index)
        
        if len(common_idx) < 10:
            continue
        
        X = mid_df.loc[common_idx].fillna(0)
        y_aligned = y.loc[common_idx]
        
        # Split and evaluate
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_aligned, test_size=test_size, random_state=random_state
        )
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = rf.predict(X_train)
        val_pred = rf.predict(X_val)
        
        train_rmse = root_mean_squared_error(y_train, train_pred)
        val_rmse = root_mean_squared_error(y_val, val_pred)
        
        # Count unique descriptors used
        unique_descriptors = set()
        for desc_list in selections.values():
            unique_descriptors.update(desc_list)
        
        result = {
            'k': k,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'overfitting_gap': val_rmse - train_rmse,
            'n_categories': len(mid_level_features),
            'total_low_features': total_features_used,
            'unique_descriptors': len(unique_descriptors),
            'avg_features_per_category': total_features_used / len(mid_level_features)
        }
        results.append(result)
        
        print(f"  Validation RMSE: {val_rmse:.4f}")
        print(f"  Overfitting gap: {result['overfitting_gap']:.4f}")
        print(f"  Total features: {total_features_used}")
    
    results_df = pd.DataFrame(results)
    
    # Find optimal k (best validation performance with reasonable overfitting)
    # Penalize overfitting in selection
    results_df['score'] = results_df['val_rmse'] + 0.1 * results_df['overfitting_gap']
    optimal_k = results_df.loc[results_df['score'].idxmin(), 'k']
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL K = {optimal_k}")
    print(f"{'='*60}")
    
    return results_df, int(optimal_k)


def create_features_with_category_selections(
    agg_df: pd.DataFrame,
    hierarchy_map: pd.DataFrame,
    category_selections: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create both low-level and mid-level features using category-specific selections.
    
    Returns:
        Tuple of (selected_low_level_df, mid_level_df)
    """
    # Collect selected low-level features
    selected_low_features = []
    
    for category, descriptors in category_selections.items():
        category_bases = hierarchy_map[
            hierarchy_map['level1'] == category
        ]['feature_name'].tolist()
        
        for base in category_bases:
            for desc in descriptors:
                feature_name = f"{base}_{desc}"
                if feature_name in agg_df.columns:
                    selected_low_features.append(feature_name)
    
    # Create low-level feature DataFrame
    low_level_df = agg_df[selected_low_features].copy()
    
    # Create mid-level features
    mid_level_features = {}
    
    for category, descriptors in category_selections.items():
        category_bases = hierarchy_map[
            hierarchy_map['level1'] == category
        ]['feature_name'].tolist()
        
        category_features = []
        for base in category_bases:
            for desc in descriptors:
                feature_name = f"{base}_{desc}"
                if feature_name in agg_df.columns:
                    category_features.append(feature_name)
        
        if category_features:
            mid_level_features[category] = agg_df[category_features].mean(axis=1)
    
    mid_level_df = pd.DataFrame(mid_level_features)
    
    return low_level_df, mid_level_df


def visualize_category_selections(
    category_selections: Dict[str, List[str]],
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame
):
    """Visualize the selected descriptors for each category"""
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
                descriptor, method='correlation'
            )
            performance_matrix.loc[category, descriptor] = score

    
    # CHANGED: from 3 to 2 subplots and adjusted figsize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Selection matrix (no changes here)
    sns.heatmap(
        selection_matrix, cmap='RdBu_r', center=0.5, annot=True, 
        fmt='g', cbar_kws={'label': 'Selected'}, ax=axes[0]
    )
    axes[0].set_title('Selected Descriptors by Category')
    axes[0].set_xlabel('Descriptors')
    axes[0].set_ylabel('Categories')
    
    # 2. Performance heatmap (no changes here)
    sns.heatmap(
        performance_matrix.astype(float), cmap='YlOrRd', annot=True, 
        fmt='.3f', cbar_kws={'label': 'Correlation'}, ax=axes[1]
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
    _, optimized_df = create_features_with_category_selections(
        agg_df, hierarchy_map, category_selections
    )
    
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
        print("✓ Category-specific selection improves performance!")
    else:
        print("✗ Baseline performs better - may need different evaluation method")
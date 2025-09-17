import pandas as pd
import numpy as np
import json
from typing import Dict, List
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
import shap

def base_of(col: str) -> str: return col.rsplit("_", 1)[0]
def desc_of(col: str) -> str: return col.rsplit("_", 1)[-1]

def _l1_merge(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    """L1-normalize a,b by their sums and return 0.5*(a+b). Treats targets equally."""
    sa, sb = float(a.sum()), float(b.sum())
    a = a / (sa + eps) if sa > 0 else a
    b = b / (sb + eps) if sb > 0 else b
    return (a.add(b, fill_value=0.0) * 0.5).sort_values(ascending=False)

def make_sample_weights(labels_df: pd.DataFrame, id_index: pd.Index, std_col: str, scheme="inv_var", eps=1e-6) -> pd.Series:
    stds = labels_df.loc[id_index, std_col].astype(float)
    if scheme == "inv_var":
        w = 1.0 / (stds**2 + eps)
    elif scheme == "inv":
        w = 1.0 / (stds + eps)
    else:  # 'soft'
        w = 1.0 / (1.0 + stds)

    # clip extremes
    lo, hi = w.quantile([0.01, 0.99])
    w = w.clip(lo, hi)

    return (w / w.mean()).astype(float)

def cv_rmse(
    model, X, y, splits,
    train_weight: pd.Series | None = None,
    val_weight:   pd.Series | None = None,
    eval_weighted: bool = False
):
    """Predefined splits. If train_weight is given, fit with it.
       Score with plain RMSE unless eval_weighted=True (then WRMSE)."""
    rmses = []
    for train_ids, val_ids in splits:
        train_idx = X.index.intersection(train_ids)
        val_idx   = X.index.intersection(val_ids)

        Xtr, Xval = X.loc[train_idx], X.loc[val_idx]
        ytr, yval = y.loc[train_idx], y.loc[val_idx]

        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        Xval = Xval.fillna(med)

        if train_weight is not None:
            wtr = train_weight.loc[train_idx]
            model.fit(Xtr, ytr, sample_weight=wtr)
        else:
            model.fit(Xtr, ytr)

        pred = model.predict(Xval)

        if eval_weighted and (val_weight is not None):
            wval = val_weight.loc[val_idx]
            rmse = float(np.sqrt(np.average((yval - pred)**2, weights=wval)))
        else:
            rmse = float(np.sqrt(((yval - pred)**2).mean()))
        rmses.append(rmse)

    return np.array(rmses)

def pick_best_k(results_df: pd.DataFrame) -> int:
    return int(results_df.loc[results_df['rmse_mean'].idxmin(), 'k'])

# evaluate descriptor by correlation - only used by visualize category selections- to show correlation of picked features alongside the RF picked features decision
def evaluate_descriptor_for_category(
    agg_df: pd.DataFrame,
    y: pd.Series,
    category: str,
    category_bases: List[str],
    descriptor: str
) -> float:
    
    features = []
    for base in category_bases:
        feature = f"{base}_{descriptor}"
        if feature in agg_df.columns:
            features.append(feature)
    
    if not features:
        return 0.0
    
    # Create category feature with NaN handling
    category_feature = agg_df[features].mean(axis=1)
    
    # Remove NaN values before correlation
    valid_mask = category_feature.notna() & y.notna()
    category_feature = category_feature[valid_mask]
    y_filtered = y[valid_mask]
    
    common_idx = category_feature.index.intersection(y_filtered.index)
    if len(common_idx) < 10:
        return 0.0
    
    aligned_feature = category_feature.loc[common_idx]
    aligned_y = y.loc[common_idx]

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

def category_mass_from_model(
    model,
    X_train: pd.DataFrame,
    hierarchy_map: pd.DataFrame,
    level: str = "perceptual",
    sample_n: int = 800,
    seed: int = 42,
) -> pd.Series:
    """Total category effect = SUM of per-column mean|SHAP| inside each category."""
    bg = X_train.sample(min(sample_n, len(X_train)), random_state=seed)
    expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = expl.shap_values(bg, check_additivity=False) # (n_samples, n_features)
    mean_abs = np.abs(sv).mean(axis=0) # per-column |SHAP|
    return _sum_by_group(mean_abs, bg.columns, hierarchy_map, level=level)

def _sum_by_group(mean_abs, cols, hierarchy_map, level="perceptual", unknown="Unknown") -> pd.Series:
    base_to_group = hierarchy_map.set_index("feature")[level].to_dict()
    sums = defaultdict(float)
    for c, v in zip(cols, mean_abs):
        base = c.rsplit("_", 1)[0]
        g = base_to_group.get(base, unknown)
        sums[g] += float(v)
    return pd.Series(sums, dtype=float).sort_values(ascending=False)

def save_results(name: str, history: Dict, test_metrics: Dict, save_dir: Path):
    """Save experiment results to JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'name': name,
        'history': history,
        'test_metrics': test_metrics
    }
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(save_dir / f'{name}.json', 'w') as f:
        json.dump(convert(results), f, indent=2)
import re
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import shap
import sys
sys.path.append('..')

from src.perceptual_descriptor_selection import (select_top_k_descriptors_per_category_shap,
                                                 create_features_with_category_selections)

def model_family_rmse_across_k(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    ks: range = range(1,9),
    cv: int = 5,
    aggregation_level: str = 'perceptual'
) -> Dict[str, pd.DataFrame]:
    
    models = {
        "Ridge": RidgeCV(alphas=np.logspace(-3,3,13)),
        "GBR": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for name, model in models.items():
        rows = []
        for k in ks:
            fold_scores = []
            for train_ids, val_ids in kf.split(agg_df):
                train_idx = agg_df.index[train_ids]
                val_idx = agg_df.index[val_ids]
                
                sel = select_top_k_descriptors_per_category_shap(
                    agg_df, y, hierarchy_map, k=k,
                    n_estimators=100,
                    train_index=train_idx, # Fixed: was undefined
                    verbose=False
                )
                _, mids = create_features_with_category_selections(
                    agg_df, hierarchy_map, sel, aggregation_levels=[aggregation_level]
                )
                if aggregation_level not in mids:
                    raise ValueError(f"aggregation_level '{aggregation_level}' not found.")
                
                X = mids[aggregation_level]
                X_train = X.loc[train_idx]
                X_val = X.loc[val_idx]
                
                med = X_train.median(numeric_only=True)
                X_train = X_train.fillna(med)
                X_val = X_val.fillna(med)
                
                model.fit(X_train, y.loc[train_idx])
                pred = model.predict(X_val)
                rmse = np.sqrt(((y.loc[val_idx] - pred)**2).mean())
                fold_scores.append(rmse)
            
            rows.append({"k": k, "rmse_mean": np.mean(fold_scores), "rmse_std": np.std(fold_scores)})
        
        results[name] = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    return results

#--------PCA selections
# organize PCA-transformed features into logical groups based on their column names
def parse_pca_groups(X_pca: pd.DataFrame) -> dict[str, list[str]]:
    """Map '{group}_PCi' columns to groups; put anything else under '__extras__'."""
    pat = re.compile(r"^(.*)_PC(\d+)$")
    groups, extras = {}, []
    for c in X_pca.columns:
        m = pat.match(c)
        if m:
            g = m.group(1)
            groups.setdefault(g, []).append(c)
        else:
            extras.append(c)
    for g, cols in groups.items():
        groups[g] = sorted(cols, key=lambda s: int(s.rsplit('PC', 1)[-1]))
    if extras:
        groups['__extras__'] = extras
    return groups

# Feature constructor for the PCA workflow. Selects the top m components per group to build the final X matrix
def build_X_pca_m(X_pca: pd.DataFrame, m: int, groups: dict[str, list[str]], keep_extras=True) -> pd.DataFrame:
    cols = []
    for g, cols_g in groups.items():
        if g == '__extras__':
            continue
        cols.extend(cols_g[:min(m, len(cols_g))])
    if keep_extras and '__extras__' in groups:
        cols.extend(groups['__extras__'])
    return X_pca[cols].copy()
# data preparation helper that ensures validation/test data is imputed using medians calculated only from the training data
def prepare_train_eval_with_train_medians(X_pca, train_ids, eval_ids, m, groups):
    """Impute eval with TRAIN medians to avoid leakage."""
    Xtr = build_X_pca_m(X_pca.loc[train_ids], m, groups)
    med = Xtr.median(numeric_only=True)
    Xev = build_X_pca_m(X_pca.loc[eval_ids],  m, groups).fillna(med)
    return Xtr.fillna(med), Xev

# get a single Series of SHAP scores summing SHAP values of principal components within a group.
def shap_grouped_from_pcs(model, X, group_map: dict[str, list[str]]):
    """Return group-level |SHAP| importances by summing |SV| across the group's PCs."""
    expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    bg = X.sample(min(800, len(X)), random_state=42)
    sv = expl.shap_values(bg) # [n_samples, n_features]
    mean_abs = np.abs(sv).mean(axis=0) # per-column |SHAP|

    # column -> group
    col_to_group = {}
    for g, cols in group_map.items():
        if g == '__extras__':
            for c in cols:
                col_to_group[c] = 'minorness' if 'minor' in c else 'extras'
        else:
            for c in cols:
                col_to_group[c] = g

    ser = (pd.Series(mean_abs, index=bg.columns)
             .groupby(col_to_group).sum()
             .sort_values(ascending=False))
    return ser, expl, bg, sv

# evaluation pipeline for the PCA approach. It tests different values of m to find the sweet spot for model performance.
def cv_over_m_joint(X_pca, y_val, y_aro, splits, m_values, n_estimators=200):
    """Use precomputed dev-only splits (e.g., cv3/cv5) to pick m jointly for valence+arousal."""
    groups = parse_pca_groups(X_pca)
    rows = []
    for m in m_values:
        fold_v, fold_a = [], []
        for tr_ids, va_ids in splits:
            Xtr = build_X_pca_m(X_pca.loc[tr_ids], m, groups)
            Xva = build_X_pca_m(X_pca.loc[va_ids], m, groups)
            med = Xtr.median(numeric_only=True)
            Xtr, Xva = Xtr.fillna(med), Xva.fillna(med)

            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

            rf.fit(Xtr, y_val.loc[Xtr.index]); pv = rf.predict(Xva)
            rmse_v = float(np.sqrt(((y_val.loc[Xva.index] - pv)**2).mean()))

            rf.fit(Xtr, y_aro.loc[Xtr.index]); pa = rf.predict(Xva)
            rmse_a = float(np.sqrt(((y_aro.loc[Xva.index] - pa)**2).mean()))

            fold_v.append(rmse_v); fold_a.append(rmse_a)
        rows.append({
            "m": m,
            "rmse_val_mean": float(np.mean(fold_v)),
            "rmse_val_std":  float(np.std(fold_v)),
            "rmse_aro_mean": float(np.mean(fold_a)),
            "rmse_aro_std":  float(np.std(fold_a)),
            "rmse_mean":      float(0.5*(np.mean(fold_v)+np.mean(fold_a))),
        })
    return pd.DataFrame(rows).sort_values("m").reset_index(drop=True)

#  CV model selection on, evaluating models on a fixed PCA-based feature design.
def cv_models_on_pca(
    X_pca: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[pd.Index,pd.Index]],
    m: int,
    groups: dict[str, list[str]],
    models: Dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    For each model, build PCA design inside each fold (impute val with train medians),
    evaluate RMSE on the precomputed splits. Returns leaderboard.
    """
    if models is None:
        models = {
            "RF":   RandomForestRegressor(n_estimators=400, max_features='sqrt',
                                          min_samples_leaf=2, random_state=42, n_jobs=-1),
            "GBR":  GradientBoostingRegressor(random_state=42),
            "Ridge":Pipeline([("sc", StandardScaler(with_mean=True, with_std=True)),
                              ("ridge", RidgeCV(alphas=np.logspace(-3,3,13)))]),
            "ElasticNet":Pipeline([("sc", StandardScaler()), 
                                   ("enet", ElasticNetCV(l1_ratio=[.1,.5,.9,1.0], 
                                                         alphas=np.logspace(-3,1,12),
                                                         random_state=42))]),
            "SVR":  Pipeline([("sc", StandardScaler()), ("svr", SVR(C=3.0, epsilon=0.1))]),
        }

    rows = []
    for name, model in models.items():
        fold_scores = []
        for tr_ids, va_ids in splits:
            # Build fold design with train medians (no leakage)
            Xtr = build_X_pca_m(X_pca.loc[tr_ids], m, groups)
            med = Xtr.median(numeric_only=True)
            Xtr = Xtr.fillna(med)

            Xva = build_X_pca_m(X_pca.loc[va_ids], m, groups).fillna(med)

            ytr = y.loc[Xtr.index]
            yva = y.loc[Xva.index]

            model.fit(Xtr, ytr)
            p = model.predict(Xva)
            rmse = float(np.sqrt(((yva - p) ** 2).mean()))
            fold_scores.append(rmse)

        rows.append({"model": name,
                     "rmse_mean": float(np.mean(fold_scores)),
                     "rmse_std":  float(np.std(fold_scores))})
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)

#-------Statistical & Validation Tests
def permutation_rmse(
    build_fold_X, # callable: (train_ids, val_ids) -> (Xtr, Xva) already imputed with TRAIN medians
    y: pd.Series,
    splits: list[tuple[pd.Index, pd.Index]],
    model_fn=lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    n_perms: int = 200,
    random_state: int = 42
) -> tuple[float, float, float, float]:
    """Permutation test for any design. Returns (real_mean, null_mean, null_std, pval)"""
    rng = np.random.RandomState(random_state)

    # real
    real_folds = []
    for tr, va in splits:
        Xtr, Xva = build_fold_X(tr, va)
        mdl = model_fn()
        mdl.fit(Xtr, y.loc[Xtr.index])
        p = mdl.predict(Xva)
        real_folds.append(float(np.sqrt(((y.loc[Xva.index]-p)**2).mean())))
    real = float(np.mean(real_folds))

    # null
    null = []
    for _ in range(n_perms):
        y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        fold = []
        for tr, va in splits:
            Xtr, Xva = build_fold_X(tr, va)
            mdl = model_fn()
            mdl.fit(Xtr, y_perm.loc[Xtr.index])
            p = mdl.predict(Xva)
            fold.append(float(np.sqrt(((y_perm.loc[Xva.index]-p)**2).mean())))
        null.append(float(np.mean(fold)))
    null = np.array(null)
    pval = float(((null <= real).sum() + 1) / (len(null) + 1))
    return real, float(null.mean()), float(null.std()), pval

# Quick test function
def test_improvement(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    category_selections: Dict[str, List[str]],
    baseline_descriptors: List[str] = ['mean']
) -> None:
    """Test improvement over baseline (using only mean)"""
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Baseline: same descriptors for all
    baseline_features = {}
    for category in hierarchy_map['perceptual'].unique():
        category_bases = hierarchy_map[
            hierarchy_map['perceptual'] == category
        ]['feature'].tolist()
        
        category_features = []
        for base in category_bases:
            for desc in baseline_descriptors:
                feature = f"{base}_{desc}"
                if feature in agg_df.columns:
                    category_features.append(feature)
        
        if category_features:
            baseline_features[category] = agg_df[category_features].mean(axis=1)
    
    baseline_df = pd.DataFrame(baseline_features)
    
    # category-specific selections
    _, optimized_dfs = create_features_with_category_selections(
        agg_df, hierarchy_map, category_selections, aggregation_levels=['perceptual']
    )
    optimized_df = optimized_dfs['perceptual']
    
    # Align and evaluate
    common_idx = baseline_df.index.intersection(y.index)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Baseline performance
    X_baseline = baseline_df.loc[common_idx]
    y_aligned = y.loc[common_idx]
    baseline_scores = -cross_val_score(
        pipe, X_baseline, y_aligned, cv=5, 
        scoring='neg_root_mean_squared_error'
    )
    
    # Optimized performance
    X_optimized = optimized_df.loc[common_idx]
    optimized_scores = -cross_val_score(
        pipe, X_optimized, y_aligned, cv=5,
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

def shap_grouped_beeswarm(model, X_design, groups, max_display=12, title=None):
    expl = shap.TreeExplainer(model)
    bg   = X_design.sample(min(800, len(X_design)), random_state=42)
    sv   = expl.shap_values(bg)  # (n_samples, n_features)

    names, shap_cols, x_cols = [], [], []
    for g, cols in groups.items():
        cols = [c for c in cols if c in bg.columns]
        if not cols: 
            continue
        idxs = [bg.columns.get_loc(c) for c in cols]
        names.append(g if g != "__extras__" else "minorness/extras")
        shap_cols.append(sv[:, idxs].sum(axis=1))
        x_cols.append(bg.iloc[:, idxs].mean(axis=1).values)

    SHAP_G = np.column_stack(shap_cols)
    X_G    = np.column_stack(x_cols)
    shap.summary_plot(SHAP_G, X_G, feature_names=names, plot_type="dot",
                      max_display=max_display, show=False)
    plt.title(title or "SHAP (grouped by perceptual)"); plt.tight_layout(); plt.show()
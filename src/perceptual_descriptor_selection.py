import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
import shap
import sys
sys.path.append('..')

from src.make_dataset.split_data import *
from src.utils import make_sample_weights, _l1_merge, cv_rmse, base_of, desc_of

#-------- Core Selection Functions
# Find the general natural order of importance on a feature set with a full SHAP
def compute_descriptor_ranking_joint_shap(
    agg_df, y_v, y_a, hierarchy_map, train_index,
    n_estimators=100, sample_n=400, random_state=42
) -> dict[str, pd.Series]:
    """
    Returns, for each perceptual category, a Series that ranks LLDD columns
    (columns in agg_df) by joint (Valence/Arousal) mean|SHAP| via L1-merge.
    Use this for selection. No descriptor-type averaging involved.
    """
    out = {}
    cats = sorted(hierarchy_map['perceptual'].unique())
    for cat in cats:
        bases = hierarchy_map[hierarchy_map['perceptual']==cat]['feature'].tolist()
        cat_cols = [c for c in agg_df.columns if base_of(c) in bases]
        if not cat_cols:
            out[cat] = pd.Series(dtype=float)
            continue

        X_cat = agg_df[cat_cols]
        Xtr = X_cat.loc[train_index]
        med = Xtr.median(numeric_only=True); Xtr = Xtr.fillna(med)

        rf_v = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1).fit(Xtr, y_v.loc[train_index])
        rf_a = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state+1, n_jobs=-1).fit(Xtr, y_a.loc[train_index])

        # SHAP -> per-column mean |SHAP|
        bg = Xtr.sample(min(sample_n, len(Xtr)), random_state=random_state)
        sv_v = shap.TreeExplainer(rf_v, feature_perturbation="tree_path_dependent").shap_values(bg, check_additivity=False)
        sv_a = shap.TreeExplainer(rf_a, feature_perturbation="tree_path_dependent").shap_values(bg, check_additivity=False)
        mean_v = np.abs(sv_v).mean(axis=0); mean_a = np.abs(sv_a).mean(axis=0)

        # MEAN per descriptor (alpha=1.0), then L1-merge V+A
        # Keep per-column SHAP values (no aggregation)
        desc_v = pd.Series(mean_v, index=bg.columns)
        desc_a = pd.Series(mean_a, index=bg.columns)
        desc_v_z = (desc_v - desc_v.mean()) / desc_v.std()
        desc_a_z = (desc_a - desc_a.mean()) / desc_a.std()
        #joint_ranking = (desc_v_z.add(desc_a_z, fill_value=0.0) * 0.5).sort_values(ascending=False) # z_score
        #print("Valence sum", desc_v.sum(), "Arousal sum", desc_a.sum())
        joint_ranking  = _l1_merge(desc_v, desc_a)  # fair joint per-descriptor ranking # L1
        #joint_ranking = (desc_v.add(desc_a, fill_value=0.0) * 0.5).sort_values(ascending=False)
        out[cat] = joint_ranking # store the full ranking (pd.Series)
    return out

# average SHAP importances from two targets (VA) to get one descriptor set
def select_top_k_descriptors_joint_shap(
    agg_df, y_v, y_a, hierarchy_map, k, train_index, n_estimators=100, sample_n=800
):
    out = {}
    for cat in sorted(hierarchy_map['perceptual'].unique()):
        bases = hierarchy_map[hierarchy_map['perceptual']==cat]['feature'].tolist()
        cat_cols = [c for c in agg_df.columns if base_of(c) in bases]
        if not cat_cols:
            out[cat] = []
            continue

        X_cat = agg_df[cat_cols]
        Xtr = X_cat.loc[train_index]
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)

        # train two RFs
        rf_v = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1).fit(Xtr, y_v.loc[train_index])
        rf_a = RandomForestRegressor(n_estimators=n_estimators, random_state=43, n_jobs=-1).fit(Xtr, y_a.loc[train_index])

        # SHAP -> per-column mean |SHAP|
        bg = Xtr.sample(min(sample_n, len(Xtr)), random_state=42)
        sv_v = shap.TreeExplainer(rf_v, feature_perturbation="tree_path_dependent").shap_values(bg, check_additivity=False)
        sv_a = shap.TreeExplainer(rf_a, feature_perturbation="tree_path_dependent").shap_values(bg, check_additivity=False)
        mean_v = np.abs(sv_v).mean(axis=0)
        mean_a = np.abs(sv_a).mean(axis=0)

        # MEAN per descriptor (alpha=1.0), then L1-merge V+A
        desc_v = pd.Series(mean_v, bg.columns)
        desc_a = pd.Series(mean_a, bg.columns)
        combo  = _l1_merge(desc_v, desc_a)   # fair joint per-descriptor ranking

        out[cat] = combo.sort_values(ascending=False).head(k).index.tolist()

    return out

# selects top_k_descriptors individually for V OR A
# (used for: SHAP summary, cv_rmse_across_k_weighted_nested,k_sweep_weighting_compare_nested)
def select_top_k_descriptors_per_category_shap(
    agg_df, y, hierarchy_map, k=3, n_estimators=100, train_index=None, verbose=False, sample_n=800
):
    """Per category:Fit RF on TRAIN -> compute SHAP on train -> pick top-k LLDD columns"""
    if train_index is None:
        raise ValueError("Pass train_index for leak-free selection.")
    y_tr = y.loc[train_index]

    selections = {}
    for cat in sorted(hierarchy_map['perceptual'].unique()):
        bases = hierarchy_map[hierarchy_map['perceptual']==cat]['feature'].tolist()
        cat_cols = [c for c in agg_df.columns if base_of(c) in bases]
        if not cat_cols:
            selections[cat] = []
            continue

        X_cat = agg_df[cat_cols]
        # train-only slice + impute with train medians
        X_tr = X_cat.loc[train_index]
        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med)

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr.loc[X_tr.index])

        # SHAP on train subset (background = train sample)
        bg = X_tr.sample(min(sample_n, len(X_tr)), random_state=42)
        expl = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
        sv = expl.shap_values(bg, check_additivity=False) # shape: [n_samples, n_features]

        mean_abs = np.abs(sv).mean(axis=0)  # per-column |SHAP|
        col_scores = pd.Series(mean_abs, index=bg.columns).sort_values(ascending=False)
        keep = list(col_scores.index[:k])

        selections[cat] = keep

        if verbose:
            print(f"{cat:20s} -> {keep}")
    return selections

# returns up to k highest importance ranked LLDD features
def selections_from_rank(rankings, k: int):
    out = {}
    for cat, r in rankings.items():
        if isinstance(r, pd.Series):
            out[cat] = list(r.sort_values(ascending=False).index[:k])
        else: # list/tuple
            out[cat] = list(r)[:k]
    return out


# Constructs a feature matrix from selected LLDDs
def build_per_base_X(
    agg_df: pd.DataFrame,
    hierarchy_map: pd.DataFrame,
    category_selections: Dict[str, List[str]],) -> pd.DataFrame:
    """One column per selected base x LLDD (no category averaging)"""
    cols, seen = [], set()
    for cat, descs in category_selections.items():
        bases = hierarchy_map[hierarchy_map['perceptual'] == cat]['feature'].tolist()
        for b in bases:
            for d in descs:
                if d in agg_df.columns: # It's already a full column name
                    c = d
                else: # Legacy: it's a descriptor type
                    c = f"{b}_{d}"
                if c in agg_df.columns and c not in seen:
                    cols.append(c); seen.add(c)
    return agg_df.loc[:, cols].astype(float)

# FULL cross validation implementation with re-ranking the features every time
def cv_rmse_across_k_nested_joint(
    agg_df, y_val, y_aro, hierarchy_map, splits,
    ks=range(1,6), aggregation_mode='per_base',
    n_estimators=100, sample_n=800):
    """
    For each k:
      Joint SHAP selection on TRAIN ONLY (valence+arousal)
      Build features (per-base or averaged)
      Fit separate RFs for valence and arousal
      Report per-fold RMSEs and their averages
    """
    rows = []
    for k in ks:
        fold_v, fold_a = [], []
        for train_ids, val_ids in splits:
            sel = select_top_k_descriptors_joint_shap(
                agg_df, y_val, y_aro, hierarchy_map, k=k,
                train_index=agg_df.index.intersection(train_ids),
                n_estimators=n_estimators, sample_n=sample_n
            )

            if aggregation_mode == 'per_base':
                X_all = build_per_base_X(agg_df, hierarchy_map, sel)
            else:
                _, mids = create_features_with_category_selections(
                    agg_df, hierarchy_map, sel, aggregation_levels=['perceptual']
                )
                X_all = mids['perceptual']

            tr = X_all.index.intersection(train_ids)
            va = X_all.index.intersection(val_ids)

            med = X_all.loc[tr].median(numeric_only=True)
            Xtr = X_all.loc[tr].fillna(med)
            Xva = X_all.loc[va].fillna(med)

            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

            rf.fit(Xtr, y_val.loc[tr]); pv = rf.predict(Xva)
            rmse_v = float(np.sqrt(((y_val.loc[va]-pv)**2).mean()))

            rf.fit(Xtr, y_aro.loc[tr]); pa = rf.predict(Xva)
            rmse_a = float(np.sqrt(((y_aro.loc[va]-pa)**2).mean()))

            fold_v.append(rmse_v); fold_a.append(rmse_a)

        rows.append({
            "k": k,
            "rmse_val_mean": float(np.mean(fold_v)),
            "rmse_val_std":  float(np.std(fold_v)),
            "rmse_aro_mean": float(np.mean(fold_a)),
            "rmse_aro_std":  float(np.std(fold_a)),
            "rmse_mean":      float(0.5*(np.mean(fold_v)+np.mean(fold_a)))
        })
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

# FAST CV: Evaluates joint V/A models across k using cached LLDD rankings from compute_descriptor_ranking_joint_shap
def cv_rmse_across_k_nested_joint_ranked(
    agg_df, y_val, y_aro, hierarchy_map, splits,
    rankings_per_fold: dict,  # {fold_id: {cat: pd.Series(desc->score)}}
    ks=range(1,6), aggregation_mode='per_base', n_estimators=100):
    rows = []
    for k in ks:
        fold_v, fold_a = [], []
        for fold_id, (train_ids, val_ids) in enumerate(splits):
            rankings = rankings_per_fold[fold_id]

            # build selections from top-k of each category
            sel = {
                cat: rankings.get(cat, pd.Series(dtype=float)).sort_values(ascending=False).index[:k].tolist()
                for cat in hierarchy_map['perceptual'].unique()
            }

            if aggregation_mode == 'per_base':
                X_all = build_per_base_X(agg_df, hierarchy_map, sel)
            else:
                _, mids = create_features_with_category_selections(
                    agg_df, hierarchy_map, sel, aggregation_levels=['perceptual']
                )
                X_all = mids['perceptual']

            tr = X_all.index.intersection(train_ids)
            va = X_all.index.intersection(val_ids)
            med = X_all.loc[tr].median(numeric_only=True)
            Xtr, Xva = X_all.loc[tr].fillna(med), X_all.loc[va].fillna(med)

            rf_v = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            rf_a = RandomForestRegressor(n_estimators=n_estimators, random_state=43, n_jobs=-1)
            rf_v.fit(Xtr, y_val.loc[tr])
            pv = rf_v.predict(Xva)
            rf_a.fit(Xtr, y_aro.loc[tr])
            pa = rf_a.predict(Xva)

            fold_v.append(float(np.sqrt(((y_val.loc[va]-pv)**2).mean())))
            fold_a.append(float(np.sqrt(((y_aro.loc[va]-pa)**2).mean())))

        rows.append({
            "k": k,
            "rmse_val_mean": float(np.mean(fold_v)),
            "rmse_val_std":  float(np.std(fold_v)),
            "rmse_aro_mean": float(np.mean(fold_a)),
            "rmse_aro_std":  float(np.std(fold_a)),
            "rmse_mean":      float(0.5*(np.mean(fold_v)+np.mean(fold_a))),
        })
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

# Feature Building Functions
# wrapper that 1) calls per_base for the normal granular LLDD sets X for development
# 2) to create smoothened versions for visualizations calls create_features_with_category_selections
def build_X_for_mode(agg_df, hierarchy_map, selections, aggregation_mode='per_base'):
    if aggregation_mode == 'per_base':
        return build_per_base_X(agg_df, hierarchy_map, selections)
    # averaged-perceptual
    _, mids = create_features_with_category_selections(
        agg_df, hierarchy_map, selections, aggregation_levels=['perceptual']
    )
    return mids['perceptual']

# smoothened version, averaging all LLDD features inside the mid-level (suboptimal approach-baseline)
def create_features_with_category_selections(
    agg_df: pd.DataFrame,
    hierarchy_map: pd.DataFrame,
    category_selections: Dict[str, List[str]],
    aggregation_levels: List[str] = ['perceptual']
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build:
      - low_level_df: selected base×descriptor columns
      - mid_level_dfs: {level_name -> mid-level df averaged within that level}
    Notes:
      - category_selections are keyed by perceptual categories
    """
    if 'feature' not in hierarchy_map.columns:
        raise ValueError("hierarchy_map must contain 'feature'.")

    # map base -> row of hierarchy (for fast lookup)
    hmap = hierarchy_map.set_index('feature')

    # collect selected low-level cols (dedupe, keep order)
    selected = []
    for cat, descs in category_selections.items():
        bases = hierarchy_map[hierarchy_map['perceptual'] == cat]['feature'].tolist()
        for base in bases:
            for d in descs:
                if d in agg_df.columns: # already a full column name
                    selected.append(d)
                else:
                    col = f"{base}_{d}" # descriptor type -> expand
                    if col in agg_df.columns:
                        selected.append(col)
    # dedupe keep-order
    seen = set()
    selected_low = [c for c in selected if not (c in seen or seen.add(c))]

    low_level_df = agg_df[selected_low].copy()

    # build requested mid-level frames
    mid_level_dfs: Dict[str, pd.DataFrame] = {}
    for level in aggregation_levels:
        if level not in hmap.columns:     
            continue # level not in the map

        groups: Dict[str, List[str]] = {}
        for col in low_level_df.columns:
            base = base_of(col)
            if base in hmap.index:
                grp = hmap.loc[base, level]
                groups.setdefault(grp, []).append(col)

        feat = {grp: low_level_df[cols].mean(axis=1) for grp, cols in groups.items() if cols}
        mid_level_dfs[level] = pd.DataFrame(feat, index=low_level_df.index)

    return low_level_df, mid_level_dfs


def shap_summary_for_k(
    agg_df: pd.DataFrame,
    y: pd.Series,
    hierarchy_map: pd.DataFrame,
    k: int = 3,
    mode: str = "per_base", # "per_base" or "perceptual" (averaged)
    aggregation_level: str = "perceptual", # which mid-level to build when averaged
    n_estimators: int = 100,
    random_state: int = 42,
    sample_n: int = 800,
    sample_weight: Optional[pd.Series] = None,
    return_feature_level: bool = False
):
    """
    Select top-k LLDD *types* per perceptual category on the TRAIN universe (all rows by default),
    then either:
      - mode="per_base": build the per-base LLDD design and return per-LLDD SHAP (optionally),
        plus a grouped-by-category SHAP series
      - mode="perceptual": build the averaged mid-level design (one column per category),
        and return SHAP directly on those categories
    """
    # 0) selection (same for both modes)
    sel = select_top_k_descriptors_per_category_shap(
        agg_df, y, hierarchy_map, k=k,
        n_estimators=n_estimators,
        train_index=agg_df.index,
        verbose=False
    )

    # 1) build design
    mode_norm = mode.lower()
    if mode_norm in {"perceptual", "perceptual_mean", "averaged"}:
        # averaged design: one feature per perceptual category
        _, mids = create_features_with_category_selections(
            agg_df, hierarchy_map, sel, aggregation_levels=[aggregation_level]
        )
        X = mids[aggregation_level]
    elif mode_norm == "per_base":
        X = build_per_base_X(agg_df, hierarchy_map, sel)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'per_base' or 'perceptual'.")

    # 2) impute, fit
    X = X.fillna(X.median(numeric_only=True))
    y_aligned = y.loc[X.index]
    w = sample_weight.loc[X.index] if sample_weight is not None else None

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X, y_aligned, sample_weight=w)

    # 3) SHAP
    bg = X.sample(min(sample_n, len(X)), random_state=random_state)
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(bg, check_additivity=False)
    mean_abs = np.abs(sv).mean(axis=0)
    mean_abs_ser = pd.Series(mean_abs, index=bg.columns)
    order = np.argsort(-mean_abs)

    if mode_norm == "per_base":
        # map LLDD -> category and sum (grouped view)
        base_to_group = hierarchy_map.set_index("feature")[aggregation_level].to_dict()
        ll_to_group = {c: base_to_group.get(base_of(c), "Unknown") for c in bg.columns}
        grouped_shap = mean_abs_ser.groupby(ll_to_group).sum().sort_values(ascending=False)

        if return_feature_level:
            # per-LLDD SHAP (mean_abs_ser, order) + grouped_shap
            return rf, X, mean_abs_ser, order, explainer, grouped_shap
        else:
            # only grouped (category) SHAP is returned
            return rf, X, None, None, explainer, grouped_shap

    else: # averaged -> X columns are categories already
        # here mean_abs_ser is already category-level
        grouped_shap = None
        return rf, X, mean_abs_ser.sort_values(ascending=False), order, explainer, grouped_shap

def aggregate_shap_for_summary(shap_values, X, hierarchy_map, level='perceptual'):
    """Aggregate SHAP values and feature values to group level for summary plot"""
    
    # Create mapping from column to group
    col_to_group = {}
    for col in X.columns:
        base_name = base_of(col)
        group = hierarchy_map[hierarchy_map['feature'] == base_name][level].values
        col_to_group[col] = group[0] if len(group) > 0 else 'Unknown'
    
    # Get unique groups
    unique_groups = sorted(set(col_to_group.values()))
    
    # Initialize aggregated arrays
    n_samples = shap_values.shape[0]
    shap_grouped = np.zeros((n_samples, len(unique_groups)))
    X_grouped = np.zeros((n_samples, len(unique_groups)))
    
    # Aggregate
    for group_idx, group in enumerate(unique_groups):
        # Find all columns belonging to this group
        cols_in_group = [i for i, col in enumerate(X.columns) if col_to_group[col] == group]
        
        if cols_in_group:
            # Sum SHAP values (preserves additivity)
            shap_grouped[:, group_idx] = shap_values[:, cols_in_group].sum(axis=1)
            # Average feature values for coloring
            X_grouped[:, group_idx] = X.iloc[:, cols_in_group].mean(axis=1).values
    
    return shap_grouped, X_grouped, unique_groups

# --- nested, weighted ---
def cv_rmse_across_k_weighted_nested(
    agg_df, y, hierarchy_map, labels_df, std_col, splits,
    ks=range(1,6), aggregation_mode='per_base'
):
    rows = []
    for k in ks:
        fold_rmses = []
        for train_ids, val_ids in splits:
            sel = select_top_k_descriptors_per_category_shap(
                agg_df, y, hierarchy_map, k=k,
                n_estimators=100,
                train_index=agg_df.index.intersection(train_ids),
                verbose=False
            )
            if aggregation_mode == 'per_base':
                X_all = build_per_base_X(agg_df, hierarchy_map, sel)
            else:
                _, mids = create_features_with_category_selections(
                    agg_df, hierarchy_map, sel, aggregation_levels=['perceptual']
                )
                X_all = mids['perceptual']
            tr = X_all.index.intersection(train_ids)
            va = X_all.index.intersection(val_ids)
            med = X_all.loc[tr].median(numeric_only=True)
            Xtr = X_all.loc[tr].fillna(med)
            Xva = X_all.loc[va].fillna(med)

            wtr = make_sample_weights(labels_df, tr, std_col=std_col)

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            fold_scores = cv_rmse(
                model, pd.concat([Xtr, Xva]), y, splits=[(tr, va)],
                train_weight=wtr, eval_weighted=False
            )
            fold_rmses.append(fold_scores[0])

        # consistent column names with utils.pick_best_k
        rows.append({"k": k, "rmse_mean": float(np.mean(fold_rmses)),
                          "rmse_std":  float(np.std(fold_rmses))})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

def k_sweep_weighting_compare_nested(
    X, y, hierarchy_map, splits, labels, std_col='valence_std',
    ks=range(1,7), use_marginal=True
):
    rows_unw, rows_w = [], []
    for k in ks:
        fold_unw, fold_w = [], []
        for tr_ids, va_ids in splits:
            # Nested selection on TRAIN ONLY
            if use_marginal and k > 1:
                sel = select_top_k_descriptors_per_category_shap(
                    X, y, hierarchy_map, k=k,
                    n_estimators=100,
                    train_index=X.index.intersection(tr_ids),
                    verbose=False
                )
            else:
                sel = select_top_k_descriptors_per_category_shap(
                    X, y, hierarchy_map, k=k,
                    n_estimators=100,
                    train_index=X.index.intersection(tr_ids),
                    verbose=False
                )

            # Build per-base design + leak-free median fill
            X_all = build_per_base_X(X, hierarchy_map, sel)
            tr = X_all.index.intersection(tr_ids)
            va = X_all.index.intersection(va_ids)
            med = X_all.loc[tr].median(numeric_only=True)
            Xtr, Xva = X_all.loc[tr].fillna(med), X_all.loc[va].fillna(med)
            ytr, yva = y.loc[tr], y.loc[va]

            # Train weights (train-only)
            wtr = make_sample_weights(labels, tr, std_col=std_col)

            # Fit models
            m_unw = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
            m_w   = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr, sample_weight=wtr)

            # evaluate with plain RMSE
            p_unw = m_unw.predict(Xva)
            p_w   = m_w.predict(Xva)
            fold_unw.append(float(np.sqrt(((yva - p_unw)**2).mean())))
            fold_w.append(  float(np.sqrt(((yva - p_w  )**2).mean())))

        rows_unw.append({"k": k, "rmse_mean": np.mean(fold_unw), "rmse_std": np.std(fold_unw)})
        rows_w.append(  {"k": k, "rmse_mean": np.mean(fold_w),   "rmse_std": np.std(fold_w)})

    return pd.DataFrame(rows_unw), pd.DataFrame(rows_w)


def select_top_k_descriptors_global_shap(X, y_v, y_a, hierarchy_map, train_index,
                                         n_estimators=200, sample_n=800, k=3):
    Xtr = X.loc[train_index].fillna(X.loc[train_index].median(numeric_only=True))
    rf_v = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1).fit(Xtr, y_v.loc[train_index])
    rf_a = RandomForestRegressor(n_estimators=n_estimators, random_state=43, n_jobs=-1).fit(Xtr, y_a.loc[train_index])

    bg   = Xtr.sample(min(sample_n, len(Xtr)), random_state=42)
    sv_v = shap.TreeExplainer(rf_v, feature_perturbation="tree_path_dependent").shap_values(bg, check_additivity=False)
    sv_a = shap.TreeExplainer(rf_a, feature_perturbation="tree_path_dependent").shap_values(bg, check_additivity=False)

    # global per-column scores
    glob = (np.abs(sv_v).mean(axis=0) + np.abs(sv_a).mean(axis=0)) / 2.0
    glob = pd.Series(glob, index=bg.columns).sort_values(ascending=False)

    # slice top-k within each perceptual category
    sel = {}
    for cat in sorted(hierarchy_map['perceptual'].unique()):
        bases = set(hierarchy_map[hierarchy_map['perceptual']==cat]['feature'])
        cat_cols = [c for c in glob.index if base_of(c) in bases]
        sel[cat] = cat_cols[:k]
    return sel
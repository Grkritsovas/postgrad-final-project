import re
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

def load_hierarchy_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['is_regex'] = df.get('is_regex', 0).astype(int)
    return df

def _expand_map_to_columns(agg_df: pd.DataFrame, hmap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = list(agg_df.columns)
    for _, r in hmap.iterrows():
        patt = str(r['feature_name'])
        if int(r.get('is_regex', 0)) == 1:
            matched = [c for c in cols if re.fullmatch(patt, c)]
        else:
            matched = [patt] if patt in agg_df.columns else []
        for c in matched:
            rows.append({'feature_name': c, 'level1': r['level1'], 'level2': r['level2']})
    return pd.DataFrame(rows)

def create_mid_level_from_map(
    agg_df: pd.DataFrame,
    hmap: pd.DataFrame,
    level: str = 'level1',
    reducer: str = 'mean'
) -> Tuple[pd.DataFrame, Dict]:
    expanded = _expand_map_to_columns(agg_df, hmap)
    mid = {}
    for cat, sub in expanded.groupby(level):
        feats = [f for f in sub['feature_name'] if f in agg_df.columns]
        if not feats: continue
        if reducer == 'mean':   mid[cat] = agg_df[feats].mean(axis=1)
        elif reducer == 'median': mid[cat] = agg_df[feats].median(axis=1)
        elif reducer == 'max':  mid[cat] = agg_df[feats].max(axis=1)
        elif reducer == 'std':  mid[cat] = agg_df[feats].std(axis=1)
        else:                   mid[cat] = agg_df[feats].mean(axis=1)
    mid_df = pd.DataFrame(mid, index=agg_df.index)

    coverage = {
        'features_in_map': int(hmap.shape[0]),
        'features_expanded': int(expanded.shape[0]),
        'features_matched': int(expanded['feature_name'].nunique()),
        'columns_in_agg': int(agg_df.shape[1]),
        'coverage_ratio': float(expanded['feature_name'].nunique() / max(1, agg_df.shape[1]))
    }
    return mid_df, coverage
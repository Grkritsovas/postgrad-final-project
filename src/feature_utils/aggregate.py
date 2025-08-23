import numpy as np
import pandas as pd
from typing import Iterable, Dict

DEFAULT_8 = ['mean','std','min','max','q25','q75','skew','kurtosis']

def _stats(s: pd.Series, which: Iterable[str]) -> Dict[str, float]:
    out = {}
    for w in which:
        if w == 'mean': out[w] = s.mean()
        elif w == 'std': out[w] = s.std()
        elif w == 'min': out[w] = s.min()
        elif w == 'max': out[w] = s.max()
        elif w == 'q25': out[w] = s.quantile(0.25)
        elif w == 'q75': out[w] = s.quantile(0.75)
        elif w == 'skew': out[w] = s.skew()
        elif w == 'kurtosis': out[w] = s.kurtosis()
        elif w == 'range': out[w] = s.max() - s.min()
        elif w == 'iqr': out[w] = s.quantile(0.75) - s.quantile(0.25)
        elif w == 'cv': out[w] = s.std()/s.mean() if s.mean()!=0 else 0.0
        elif w == 'median': out[w] = s.median()
        elif w == 'mad': out[w] = (s - s.median()).abs().median()
        elif w == 'variation': out[w] = s.diff().abs().mean()
        elif w == 'trend':
            out[w] = (s.iloc[-5:].mean() - s.iloc[:5].mean()) if len(s)>=10 else (s.iloc[-1]-s.iloc[0])
    return out

def aggregate_low(df_frames: pd.DataFrame, which=DEFAULT_8, sep='_') -> pd.Series:
    """Aggregate one song's frames→static descriptors for every low-level feature."""
    if df_frames.empty:
        raise ValueError("Empty DataFrame")
    if df_frames.shape[0] < 3:
        print(f"Warning: Only {df_frames.shape[0]} frames, results may be unreliable")

    out = {}
    for col in df_frames.columns:
        st = _stats(df_frames[col], which)
        for k, v in st.items():
            out[f"{col}{sep}{k}"] = v
    return pd.Series(out)

def descriptor_distribution(agg_series: pd.Series) -> Dict[str,int]:
    counts = {}
    for name in agg_series.index:
        suffix = name.rsplit('_', 1)[-1]
        counts[suffix] = counts.get(suffix, 0) + 1
    return counts

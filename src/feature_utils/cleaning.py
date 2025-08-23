import numpy as np
import pandas as pd

_TIME_KEYS = {'frametime','frameTime','time','frame'}

def remove_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in _TIME_KEYS:
            return df.drop(columns=[c])
    # heuristic: first column looks like time
    first = df.columns[0]
    if (first.lower() in _TIME_KEYS) or (
        pd.api.types.is_numeric_dtype(df[first]) and df[first].is_monotonic_increasing
    ):
        return df.drop(columns=[first])
    return df

def clean(df: pd.DataFrame, strategy: str = "mean_fill") -> pd.DataFrame:
    df = remove_time_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
    if strategy == "mean_fill":
        # Only iterate over numeric columns for filling
        for c in df.select_dtypes(include=np.number).columns:
            if df[c].isna().all():
                df[c] = 0
            else:
                df[c] = df[c].fillna(df[c].mean())
    elif strategy == "zero_fill":
        df = df.fillna(0)
    elif strategy == "drop_cols":
        thresh = int(len(df) * 0.5)
        df = df.dropna(axis=1, thresh=thresh)
    elif strategy == "drop_rows":
        df = df.dropna(axis=0)
    return df
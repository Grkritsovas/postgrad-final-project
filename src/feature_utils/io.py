from pathlib import Path
import pandas as pd
from typing import Iterator, Tuple

def load_opensmile_csv(path: Path, sep: str = ';') -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

def load_many_csvs(root: Path, pattern: str = "*.csv", sep: str = ';') -> Iterator[Tuple[str, pd.DataFrame]]:
    for p in sorted(root.glob(pattern)):
        sid = p.stem
        yield sid, load_opensmile_csv(p, sep=sep)
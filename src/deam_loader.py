# src/deam_loader.py

import pandas as pd
from pathlib import Path
import logging

# Module-level setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_annotations(annotations_path: Path) -> pd.DataFrame:
    """Loads and validates the DEAM annotations CSV file."""
    logger.info(f"Loading annotations from: {annotations_path}")
    df = pd.read_csv(annotations_path)
    expected = {"song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Annotations file missing expected columns. Found: {list(df.columns)}")
    return df

def load_metadata(metadata_dir: Path) -> pd.DataFrame:
    """Loads and merges all metadata CSVs from a directory."""
    logger.info(f"Loading metadata from directory: {metadata_dir}")
    files = sorted(metadata_dir.glob("metadata_*.csv"))
    if not files:
        raise FileNotFoundError(f"No metadata files found in {metadata_dir}")
    
    frames = [pd.read_csv(f)[["song_id", "Artist", "Track"]] for f in files]
    meta_df = pd.concat(frames, ignore_index=True)
    meta_df = meta_df.drop_duplicates(subset="song_id", keep="first")
    meta_df = meta_df.rename(columns={"Artist": "artist_name", "Track": "track_name"})
    return meta_df

def create_deam_base_dataset(annotations_path: str, metadata_dir: str) -> pd.DataFrame:
    # Load main annotations (1-2000)
    annotations = pd.read_csv(annotations_path)
    
    # Try to load test annotations (2000-2058) from same directory
    test_path = Path(annotations_path).parent / "static_annotations_averaged_songs_2000_2058.csv"
    
    if test_path.exists():
        test_annotations = pd.read_csv(test_path)
        # Keep only the columns we need (same as main annotations)
        required_cols = ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]
        test_annotations = test_annotations[required_cols]
        
        annotations = pd.concat([annotations, test_annotations], ignore_index=True)
        print(f"Loaded {len(test_annotations)} test songs")
    
    # Load all metadata files
    metadata_files = sorted(Path(metadata_dir).glob("metadata_*.csv"))
    metadata_list = []
    for f in metadata_files:
        df = pd.read_csv(f)[["song_id", "Artist", "Track"]]
        metadata_list.append(df)
    
    metadata = pd.concat(metadata_list).drop_duplicates(subset="song_id")
    metadata.columns = ["song_id", "artist_name", "track_name"]
    
    # Merge
    core_df = pd.merge(annotations, metadata, on="song_id", how="inner")
    
    # Clean text
    core_df["artist_name"] = core_df["artist_name"].str.strip()
    core_df["track_name"] = core_df["track_name"].str.strip()
    
    # Reorder columns
    final_cols = [
        "song_id", "track_name", "artist_name",
        "valence_mean", "arousal_mean", "valence_std", "arousal_std"
    ]
    core_df = core_df[final_cols]
    
    print(f"Created dataset with {len(core_df)} songs")
    print(f"  Train/val (≤2000): {len(core_df[core_df['song_id'] <= 2000])}")
    print(f"  Test (>2000): {len(core_df[core_df['song_id'] > 2000])}")
    
    return core_df
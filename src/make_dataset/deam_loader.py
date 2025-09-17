import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from typing import Set, Tuple, List, Dict
from tqdm.auto import tqdm
import sys
from sklearn.decomposition import PCA
sys.path.append('..')

from src.io import load_opensmile_csv
from src.aggregate import aggregate_low
from src.make_dataset.cleaning import clean

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

def create_features_2080(features_path: Path, descriptor_set=None) -> Tuple[pd.DataFrame, List]:
    rows = []
    ids = []
    failed_ids = []
    
    csv_files = sorted(features_path.glob("*.csv"))
    
    for csv_path in tqdm(csv_files, desc="Aggregating features"):
        try:
            sid = int(csv_path.stem)
            # Load and clean
            df = load_opensmile_csv(csv_path, sep=';')
            df_clean = clean(df)
            # Quality check
            if df_clean.shape[0] < 5 or df_clean.shape[1] < 10:
                failed_ids.append(sid)
                continue
            # Aggregate to statistical descriptors
            if descriptor_set:
                agg = aggregate_low(df_clean, which=descriptor_set)
            else:
                agg = aggregate_low(df_clean)  # DEFAULT_8
            
            rows.append(agg)
            ids.append(sid)
            
        except Exception as e:
            failed_ids.append(int(csv_path.stem) if csv_path.stem.isdigit() else csv_path.stem)

    agg_df = pd.DataFrame(rows, index=ids)
    agg_df.index.name = 'song_id'
    agg_df.index = agg_df.index.astype(int)
    agg_df = agg_df.sort_index()
    
    return agg_df, failed_ids

DESCRIPTORS = [
        "mean", "std", "min", "max", "q25", "q75", "skew", "kurtosis",
        "range", "iqr", "cv", "median", "mad", "variation", "trend"
        
    ]

import re

# Keep openSMILE functionals; strip only dataset-level stats
# "audSpec_Rfilt_sma_de[10]_amean_std" -> core "audSpec_Rfilt_sma_de[10]_amean"
def core_of(col: str) -> str:
    return re.sub(r'_(?:mean|std|min|max|q25|q75|skew|kurtosis)$', '', col)

def core_of_full(col: str) -> str:
    return re.sub(r'_(?:mean|std|min|max|q25|q75|skew|kurtosis|range|iqr|cv|median|mad|variation|trend)$', '', col)

def base_of(col: str) -> str:
    return re.sub(r'_(?:de_)?(?:amean|stddev)$', '', core_of(col))

def _is_mean(core: str) -> bool:    return core.endswith('_amean')
def _is_stddev(core: str) -> bool:  return core.endswith('_stddev')
def _has_de(core: str) -> bool:     return '_de_' in core

def _audspec_band(core: str):
    m = re.search(r'audSpec_Rfilt_sma(?:_de)?\[(\d+)\]', core)
    return int(m.group(1)) if m else None

def match_core_rule(core: str):
    """
    Map a *core* feature name (keeps openSMILE functional suffix) to:
    (perceptual, musical, acoustic).
    Design:
    - High/Med confidence rules: (F0, voicing, HNR, RMS, flux, harmonicity, entropy).
    - Family-level inclusions: (MFCC, audSpec bands, rolloff).
    """

    # Pitch / Voicing (High)
    if core.startswith('F0final_sma_'):
        if _is_mean(core):                 return ("melodiousness","pitch","fundamental_freq")
        if core.endswith('_de_amean'):     return ("melodiousness","pitch","pitch_delta")
        if _is_stddev(core):               return ("tonal_stability","pitch","pitch_delta")
        return None

    if core.startswith('voicingFinalUnclipped_sma_'):
        if _is_mean(core) or core.endswith('_de_amean'):
            return ("melodiousness","voice","voicing")
        if _is_stddev(core):
            return ("tonal_stability","voice","voicing")
        return None

    # AudSpec_Rfilt *deltas* by band (Med)
    if core.startswith('audSpec_Rfilt_sma_de['):
        b = _audspec_band(core)
        if b is None: return None
        if  0 <= b <=  5: return ("rhythmic_complexity","rhythm","audspec_delta_low")
        if  6 <= b <=  9: return ("articulation","rhythm","audspec_delta_lowmid")
        if 10 <= b <= 15: return ("articulation","rhythm","audspec_delta_mid")
        if 16 <= b <= 19: return ("articulation","rhythm","audspec_delta_uppermid")
        if 20 <= b <= 25: return ("rhythmic_complexity","rhythm","audspec_delta_high")
        return None

    # Voice quality (Med, with caveat)
    if core.startswith('jitterLocal_sma_') or core.startswith('jitterDDP_sma_'):
        # Keep in one bucket to avoid PCA split; perceptual salience is debated.
        return ("articulation","voice_quality","jitter")
    
    if core.startswith('shimmerLocal_sma_'):
        return ("articulation","voice_quality","shimmer")

    if core.startswith('logHNR_sma_'):
        # Means -> cleaner/less noisy attacks; variability -> instability tied to roughness.
        if _is_mean(core) or core.endswith('_de_amean'):
            return ("articulation","voice_quality","noise_ratio")
        if _is_stddev(core):
            return ("dissonance","voice_quality","noise_ratio")
        return None

    # Energy / Loudness envelope (High)
    if (core.startswith('pcm_RMSenergy_sma_') or
        core.startswith('loudness_sma_') or
        core.startswith('audspec_lengthL1norm_sma_') or
        core.startswith('audspecRasta_lengthL1norm_sma_')):
        # Keep families coherent: means → stability; stddev → complexity
        if _is_mean(core) or core.endswith('_de_amean'):
            return ("rhythmic_stability","dynamics","energy")
        if _is_stddev(core):
            return ("rhythmic_complexity","dynamics","energy")
        return None

    # Mid-high band energy (Med)
    if core.startswith('pcm_fftMag_fband1000-4000_sma_'):
        if _is_mean(core) or core.endswith('_de_amean'):
            return ("articulation","timbre","band_midhigh")
        if _is_stddev(core):
            return ("dissonance","timbre","band_midhigh")
        return None

    # Sharpness / Brightness (High)
    if core.startswith('pcm_fftMag_psySharpness_sma_'):
        return ("dissonance","timbre","sharpness")

    if core.startswith('pcm_fftMag_spectralCentroid_sma_'):
        # Treat as "brightness" in musical tag to reflect timbre axis
        return ("dissonance","brightness","spectral_centroid")

    # Tonality vs. noisiness (High)
    if core.startswith('pcm_fftMag_spectralEntropy_sma_'):
        # Low entropy -> more tonal; variability in entropy -> instability
        if _is_stddev(core):
            return ("tonal_stability","timbre","spectral_complexity")
        else:
            return ("dissonance","timbre","spectral_complexity")

    # Spectral change / onsets (High)
    if core.startswith('pcm_fftMag_spectralFlux_sma_'):
        if _is_stddev(core):
            return ("rhythmic_complexity","rhythm","spectral_change")
        else:
            return ("articulation","rhythm","spectral_change")

    # Harmonicity (High) 
    if core.startswith('pcm_fftMag_spectralHarmonicity_sma_'):
        if _is_stddev(core):
            return ("dissonance","harmony","harmonicity")
        else:
            return ("tonal_stability","harmony","harmonicity")

    #  Spectral slope (Med) 
    if core.startswith('pcm_fftMag_spectralSlope_sma_'):
        return ("articulation","brightness","spectral_slope")

    # ZCR (Med) 
    if core.startswith('pcm_zcr_sma_'):
        if _is_stddev(core):
            return ("rhythmic_complexity","rhythm","zero_crossing")
        else:
            return ("articulation","rhythm","zero_crossing")

    # Spectral rolloff (Med)
    if core.startswith('pcm_fftMag_spectralRollOff'):
        # Keep entire family coherent with brightness axis
        if _is_stddev(core):
            return ("tonal_stability","brightness","spectral_rolloff")
        else:
            return ("dissonance","brightness","spectral_rolloff")

    # Spectral variance / skewness / kurtosis (Low)
    if (core.startswith('pcm_fftMag_spectralVariance_sma_') or
        core.startswith('pcm_fftMag_spectralSkewness_sma_') or
        core.startswith('pcm_fftMag_spectralKurtosis_sma_')):
        return ("dissonance","timbre","spectral_shape")

    # MFCCs - timbre, possibly connected to tonal stability
    if (core.startswith('pcm_fftMag_mfcc_sma_') or
        core.startswith('pcm_fftMag_mfcc_sma_de[')):
        # Map all MFCC stats to one group
        return ("tonal_stability","timbre","mfcc")

    # AudSpec_Rfilt *static* bands (Low-Med)
    # Keep bands in stable buckets so PCA per group is clean.
    if core.startswith('audSpec_Rfilt_sma['):
        b = _audspec_band(core)
        if b is None: return None
        if  0 <= b <=  5: return ("rhythmic_stability","spectral","audspec_low")
        if  6 <= b <=  9: return ("tonal_stability","spectral","audspec_lowmid")
        if 10 <= b <= 15: return ("melodiousness","spectral","audspec_mid")
        if 16 <= b <= 19: return ("articulation","spectral","audspec_uppermid")
        if 20 <= b <= 25: return ("dissonance","spectral","audspec_high")
        return None

    # Not covered
    return None

def build_hierarchy_map(agg_df: pd.DataFrame,
                        report_csv: Path = Path("../results/unmapped_features.csv")) -> pd.DataFrame:
    cores: Set[str] = set(core_of(c) for c in agg_df.columns)
    rows, unmapped = [], []

    for core in sorted(cores):
        res = match_core_rule(core)
        if res is None:
            rows.append({"feature": core, "perceptual": "unmapped", "musical": "-"})
            unmapped.append(core)
        else:
            per, musical, _ = res
            rows.append({"feature": core, "perceptual": per, "musical": musical})

    df_map = pd.DataFrame(rows, columns=["feature", "perceptual", "musical"])

    if unmapped:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unmapped_core": sorted(unmapped)}).to_csv(report_csv, index=False)
        print(f"{len(unmapped)} unmapped features saved to {report_csv}")

    total = len(cores); n_unmapped = len(unmapped)
    print("\n=== Feature Hierarchy Mapping Summary ===")
    print(f"Total features: {total}")
    print(f"Mapped features: {total - n_unmapped} ({(total - n_unmapped)/total:.1%})")
    print(f"Unmapped features: {n_unmapped} ({n_unmapped/total:.1%})")

    print("\n=== Perceptual Dimension Distribution ===")
    counts = df_map['perceptual'].value_counts()
    for dim in ['melodiousness','articulation','rhythmic_stability',
                'rhythmic_complexity','dissonance','tonal_stability','unmapped']:
        print(f"  {dim:20s}: {counts.get(dim, 0):3d} cores")

    return df_map

def filter_unmapped_features(features_df: pd.DataFrame, hierarchy_df: pd.DataFrame, enriched=False) -> pd.DataFrame:
    """Keep only columns whose **core** mapped (perceptual != 'unmapped')."""
    mapped_cores = set(hierarchy_df[hierarchy_df['perceptual'] != 'unmapped']['feature'])
    if not enriched:
        keep_cols = [col for col in features_df.columns if core_of(col) in mapped_cores]
    else:
        keep_cols = [col for col in features_df.columns if core_of_full(col) in mapped_cores]
    print(f"Filtered features: {len(features_df.columns)} -> {len(keep_cols)}")
    return features_df[keep_cols]

def remove_correlated_within_groups(
    features_df: pd.DataFrame, 
    hierarchy_df: pd.DataFrame, 
    threshold: float = 0.95
) -> pd.DataFrame:
    """Remove highly correlated features within each perceptual group."""
    
    # Create mapping from hierarchy DataFrame
    mapping = {row['feature']: row['perceptual'] 
              for _, row in hierarchy_df.iterrows()}
    
    features_to_drop = set()
    
    for group in ['melodiousness', 'articulation', 'rhythmic_stability', 
                  'rhythmic_complexity', 'dissonance', 'tonal_stability']:
        
        # Find columns belonging to this group
        group_cols = [col for col in features_df.columns 
              if mapping.get(core_of_full(col)) == group]
        
        if len(group_cols) > 1:
            corr_matrix = features_df[group_cols].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [col for col in upper.columns 
                      if any(upper[col] > threshold)]
            features_to_drop.update(to_drop)
    
    print(f"Removing {len(features_to_drop)} correlated features")
    return features_df.drop(columns=list(features_to_drop))

from sklearn.preprocessing import StandardScaler

def pca_per_group(
    features_df: pd.DataFrame, 
    hierarchy_df: pd.DataFrame, 
    n_components: int = None, 
    variance_explained: float = 0.95,
    train_ids: pd.Index = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply PCA within each perceptual group.
    Returns transformed features and fitted PCA objects for later use.
    """
    
    # Create mapping from hierarchy DataFrame
    mapping = {row['feature']: row['perceptual'] 
              for _, row in hierarchy_df.iterrows()}
    
    result_dfs = []
    pca_models = {}  # Store for applying to test set
    
    for group in ['melodiousness', 'articulation', 'rhythmic_stability',
                  'rhythmic_complexity', 'dissonance', 'tonal_stability']:
        
        # Find columns belonging to this group
        group_cols = [col for col in features_df.columns 
                     if mapping.get(core_of_full(col)) == group]
        
        if not group_cols:
            continue
            
        X_group = features_df[group_cols]

        scaler = StandardScaler().fit(X_group.loc[train_ids])
        X_train = scaler.transform(X_group.loc[train_ids])
        X_all   = scaler.transform(X_group)
        
        # Determine n_components
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_train)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum >= variance_explained) + 1
            n_comp = max(1, min(n_comp, len(group_cols)))  # Ensure valid range
        else:
            n_comp = min(n_components, len(group_cols))
        
        # fit PCA on train, transform all
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_train)
        X_pca = pca.transform(X_all)
        
        # Store the model and column info
        pca_models[group] = {
            'pca': pca,
            'scaler': scaler,
            'columns': group_cols,
            'n_components': n_comp
        }
        
        # Create meaningful column names
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f"{group}_PC{i+1}" for i in range(n_comp)],
            index=features_df.index
        )
        result_dfs.append(pca_df)
    
    return pd.concat(result_dfs, axis=1), pca_models

def apply_pca_models(features_df: pd.DataFrame, pca_models: Dict) -> pd.DataFrame:
    """Apply pre-fitted PCA models to new data (e.g., test set)."""
    result_dfs = []
    
    for group, model_info in pca_models.items():
        X_group = features_df[model_info['columns']]

        if model_info.get('scaler') is not None:
            X_group = model_info['scaler'].transform(X_group)
        else:
            X_group = X_group.values
        X_pca = model_info['pca'].transform(X_group)
        
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f"{group}_PC{i+1}" for i in range(model_info['n_components'])],
            index=features_df.index
        )
        result_dfs.append(pca_df)
    
    return pd.concat(result_dfs, axis=1)

def create_rnn_dataset(features_dir="../data/raw/features", min_rows=90):
    data = {}
    
    for csv_path in Path(features_dir).glob("*.csv"):
        song_id = int(csv_path.stem)
        df = pd.read_csv(csv_path, sep=';')
        
        # Sub-sample to exactly 90 rows
        if len(df) >= min_rows:
            # Take evenly spaced samples
            indices = np.linspace(0, len(df)-1, min_rows, dtype=int)
            df_sampled = df.iloc[indices].reset_index(drop=True)
        else:
            # Pad with repetition if too short
            df_sampled = df
            while len(df_sampled) < min_rows:
                df_sampled = pd.concat([df_sampled, df])
            df_sampled = df_sampled.iloc[:min_rows].reset_index(drop=True)
        
        data[song_id] = df_sampled.values  # Shape: (90, 260)
    
    # Save as a 3D array: (n_songs, 90, 260)
    songs_array = np.array([data[sid] for sid in sorted(data.keys())])
    songs_ids = np.array(sorted(data.keys()))

    output_path = '../data/processed/features_rnn_90x260.npz'
    # Ensure the directory exists before saving
    os.makedirs(Path(output_path).parent, exist_ok=True) 
    
    np.savez('../data/processed/features_rnn_90x260.npz', 
             features=songs_array, 
             song_ids=songs_ids)
    
    return songs_array, songs_ids
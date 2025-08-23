import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict

def load_and_prepare_data(high_path: str, mid_path: str = None, low_path: str = None) -> pd.DataFrame:
    """Load and merge DEAM feature datasets."""
    df = pd.read_parquet(high_path)
    
    if mid_path:
        mid = pd.read_parquet(mid_path)
        df = df.merge(mid, on='song_id', suffixes=('', '_mid'))
    
    if low_path:
        low = pd.read_parquet(low_path)
        df = df.merge(low, on='song_id', suffixes=('', '_low'))
    
    # Create derived features
    df['key_mode'] = df['key'].astype(str) + '_' + df['mode'].astype(str)
    
    return df

def plot_emotion_distributions(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
    """Plot distributions of valence and arousal metrics."""
    columns = ['valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']
    titles = [col.replace('_', ' ').title() for col in columns]
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    for i, (col, title) in enumerate(zip(columns, titles)):
        row, col_pos = i // 2, i % 2
        axs[row, col_pos].hist(df[col].dropna(), bins=30, alpha=0.7)
        axs[row, col_pos].set_title(f'Distribution of {title}')
        axs[row, col_pos].set_xlabel(title)
        axs[row, col_pos].set_ylabel('Frequency')
        axs[row, col_pos].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_emotion_scatter(df: pd.DataFrame, metric_type: str = 'mean') -> None:
    """Plot valence vs arousal scatter with reference lines."""
    val_col = f'valence_{metric_type}'
    aro_col = f'arousal_{metric_type}'
    
    val_mid = df[val_col].mean()
    aro_mid = df[aro_col].mean()
    
    plt.figure(figsize=(6, 4.5))
    plt.scatter(df[val_col], df[aro_col], alpha=0.7, s=25)
    plt.axhline(y=aro_mid, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=val_mid, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.title(f'Valence {metric_type.title()} vs Arousal {metric_type.title()}')
    plt.xlabel(f'Valence {metric_type.title()}')
    plt.ylabel(f'Arousal {metric_type.title()}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_feature_emotion_relationship(df: pd.DataFrame, 
                                       feature_col: str, 
                                       emotion_col: str,
                                       threshold_offset: float = 0) -> pd.Series:
    """Analyze relationship between a feature and emotion using normalized ratios."""
    threshold = df[emotion_col].median() + threshold_offset
    
    # High emotion subset
    high_emotion = df[df[emotion_col] > threshold]
    freq_high = high_emotion[feature_col].value_counts()
    freq_total = df[feature_col].value_counts()
    
    # Normalized ratio
    ratio = (freq_high / freq_total).sort_values(ascending=False)
    return ratio

def plot_feature_emotion_analysis(df: pd.DataFrame, 
                                feature_col: str, 
                                emotion_col: str,
                                top_n: int = 7,
                                threshold_offset: float = 0) -> None:
    """Plot bar chart of top features associated with high emotion."""
    ratio = analyze_feature_emotion_relationship(df, feature_col, emotion_col, threshold_offset)
    top_features = ratio.head(top_n)
    
    threshold = df[emotion_col].median() + threshold_offset
    
    plt.figure(figsize=(8, 4))
    plt.bar(top_features.index, top_features.values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(f'High-{emotion_col.split("_")[0]} / Overall')
    plt.title(f'Top {top_n} {feature_col} for {emotion_col} > {threshold:.2f}')
    plt.tight_layout()
    plt.show()

def plot_bpm_comparison(df: pd.DataFrame, 
                       emotion_col: str, 
                       threshold_offset: float = 0) -> None:
    """Plot BPM distribution comparison between all tracks and high emotion tracks."""
    threshold = df[emotion_col].median() + threshold_offset
    bins = range(int(df['bpm'].min()), int(df['bpm'].max()) + 10, 10)
    
    plt.figure(figsize=(8, 4))
    plt.hist(df['bpm'], bins=bins, alpha=0.4, label='All tracks')
    plt.hist(df[df[emotion_col] > threshold]['bpm'], bins=bins, 
             alpha=0.6, label=f'High {emotion_col.split("_")[0]}')
    plt.xlabel('BPM')
    plt.ylabel('Count')
    plt.legend()
    plt.title(f'BPM Distribution: All vs High {emotion_col.split("_")[0].title()}')
    plt.tight_layout()
    plt.show()

def get_feature_correlations(df: pd.DataFrame, 
                           target_cols: List[str] = ['valence_mean', 'arousal_mean'],
                           exclude_patterns: List[str] = ['lyrics', 'track_name', 'artist_name']) -> Dict[str, pd.Series]:
    """Get feature correlations with target emotions."""
    # Select numeric columns excluding specified patterns
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if not any(pattern in col for pattern in exclude_patterns)
                   and col not in target_cols]
    
    correlations = {}
    for target in target_cols:
        corr_data = df[feature_cols + [target]].corr()[target].drop(target)
        correlations[target] = corr_data.abs().sort_values(ascending=False)
    
    return correlations

def plot_top_correlations(correlations: Dict[str, pd.Series], top_n: int = 10) -> None:
    """Plot top feature correlations for each emotion dimension."""
    fig, axes = plt.subplots(1, len(correlations), figsize=(6*len(correlations), 4))
    if len(correlations) == 1:
        axes = [axes]
    
    for i, (emotion, corr_series) in enumerate(correlations.items()):
        top_features = corr_series.head(top_n)
        axes[i].bar(range(len(top_features)), top_features.values)
        axes[i].set_xticks(range(len(top_features)))
        axes[i].set_xticklabels(top_features.index, rotation=45, ha='right')
        axes[i].set_ylabel('|Pearson r|')
        axes[i].set_title(f'Top {top_n} Features vs {emotion}')
    
    plt.tight_layout()
    plt.show()
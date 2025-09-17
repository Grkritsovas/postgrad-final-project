import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import umap

import sys
sys.path.append('..')

from src.io import load_opensmile_csv

sns.set_style("whitegrid")

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

def create_correlation_df(core_df, df_perceptual, df_musical, metadata_df=None):

    combined_df = core_df[['valence_mean', 'arousal_mean']].copy()

    # Merge the feature DataFrames (same Index)
    combined_df = combined_df.merge(df_perceptual, left_index=True, right_index=True)
    combined_df = combined_df.merge(df_musical, left_index=True, right_index=True)

    # If the optional metadata DataFrame is provided, merge it too
    if metadata_df is not None:
        combined_df = combined_df.merge(metadata_df, left_index=True, right_index=True)

    return combined_df

def plot_emotion_distributions(df: pd.DataFrame, VIZ_PATH: Path, figsize: Tuple[int, int] = (12, 8)) -> None:
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
    plt.savefig(VIZ_PATH / 'emotion_distributions.png', dpi=150)
    plt.show()

def plot_emotion_scatter(df: pd.DataFrame, VIZ_PATH: Path, metric_type: str = 'mean') -> None:
    """Plot valence vs arousal scatter with reference lines."""
    val_col = f'valence_{metric_type}'
    aro_col = f'arousal_{metric_type}'
    
    val_mid = df[val_col].mean()
    aro_mid = df[aro_col].mean()
    
    plt.figure(figsize=(6, 4.5))
    plt.scatter(df[aro_col], df[val_col], alpha=0.7, s=25)
    plt.axhline(y=val_mid, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=aro_mid, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.title(f'Valence {metric_type.title()} vs Arousal {metric_type.title()}')
    plt.xlabel(f'Valence {metric_type.title()}')
    plt.ylabel(f'Arousal {metric_type.title()}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'emotion_scatter.png', dpi=150)
    plt.show()

def analyze_feature_emotion_relationship(df: pd.DataFrame, 
                                       feature_col: str, 
                                       emotion_col: str,
                                       threshold_offset: float = 0) -> pd.Series:
    """Analyze relationship between a feature and emotion using normalized ratios"""
    threshold = df[emotion_col].median() + threshold_offset
    
    # High emotion subset
    high_emotion = df[df[emotion_col] > threshold]
    freq_high = high_emotion[feature_col].value_counts()
    freq_total = df[feature_col].value_counts()
    
    # Normalized ratio
    ratio = (freq_high / freq_total).sort_values(ascending=False)
    return ratio

def plot_key_mode_distributions(df: pd.DataFrame, VIZ_PATH: Path, 
                               emotions: List[str] = ['valence_mean', 'arousal_mean']) -> None:
    """Create violin plots showing emotion distributions for each key/mode"""
    fig, axes = plt.subplots(len(emotions), 2, figsize=(14, 6*len(emotions)))
    if len(emotions) == 1:
        axes = axes.reshape(1, -1)
    
    for i, emotion in enumerate(emotions):
        # Plot by key
        ax = axes[i, 0]
        key_order = df.groupby('key')[emotion].median().sort_values(ascending=False).index
        sns.violinplot(data=df, x='key', y=emotion, order=key_order, ax=ax, inner='box')
        ax.set_title(f'{emotion.replace("_", " ").title()} Distribution by Key')
        ax.set_xlabel('Key')
        ax.set_ylabel(emotion.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        
        # Plot by mode
        ax = axes[i, 1]
        sns.violinplot(data=df, x='mode', y=emotion, ax=ax, inner='box')
        ax.set_title(f'{emotion.replace("_", " ").title()} Distribution by Mode')
        ax.set_xlabel('Mode')
        ax.set_ylabel(emotion.replace('_', ' ').title())
        
        # Add median line
        ax.axhline(df[emotion].median(), color='red', linestyle='--', 
                  alpha=0.5, label=f'Overall median')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'key_mode_distributions.png', dpi=150)
    plt.show()

def plot_bpm_correlations(df: pd.DataFrame, VIZ_PATH: Path,
                         emotions: List[str] = ['valence_mean', 'arousal_mean']) -> None:
    """
    Create scatter plots with regression lines to show BPM vs emotion relationships.
    Shows correlation strength and p-values directly on the plot.
    """
    fig, axes = plt.subplots(1, len(emotions), figsize=(6*len(emotions), 5))
    if len(emotions) == 1:
        axes = [axes]
    
    for i, emotion in enumerate(emotions):
        ax = axes[i]
        
        # Drop NaN values from bpm
        data = df[['bpm', emotion]].dropna()
        
        # Remove outliers for cleaner visualization
        q1, q3 = data['bpm'].quantile([0.01, 0.99])
        data = data[(data['bpm'] >= q1) & (data['bpm'] <= q3)]
        
        # Calculate correlations
        pearson_r, pearson_p = stats.pearsonr(data['bpm'], data[emotion])
        spearman_r, spearman_p = stats.spearmanr(data['bpm'], data[emotion])
        
        # Create scatter plot with regression line
        sns.regplot(data=data, x='bpm', y=emotion, ax=ax, 
                   scatter_kws={'alpha': 0.3, 's': 10},
                   line_kws={'color': 'red', 'linewidth': 2})
        
        # Add correlation info to plot
        textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.3f})\n'
        textstr += f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.3f})'
        
        # Interpret correlation strength
        strength = "No" if abs(pearson_r) < 0.1 else \
                  "Weak" if abs(pearson_r) < 0.3 else \
                  "Moderate" if abs(pearson_r) < 0.5 else "Strong"
        
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'BPM vs {emotion.replace("_", " ").title()}\n({strength} correlation)')
        ax.set_xlabel('BPM')
        ax.set_ylabel(emotion.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'bpm_correlations.png', dpi=150)
    plt.show()

def plot_feature_correlations_heatmap(df: pd.DataFrame, VIZ_PATH: Path,
                                     features: Optional[List[str]] = None,
                                     emotions: List[str] = ['valence_mean', 'arousal_mean'],
                                     level = 'high') -> pd.DataFrame:
    """
    Create a comprehensive correlation heatmap with significance indicators
    Returns correlation DataFrame for further analysis
    """

    if features is None:
        features = [c for c in df.columns if c not in {'song_id', 'valence_mean', 'arousal_mean'}]
    # keep only columns that exist
    features = [c for c in features if c in df.columns]
    emotions = [e for e in emotions if e in df.columns]
    all_cols = features + emotions
    if not all_cols or not emotions:
        print("No valid features/emotions found in DataFrame.")
        return pd.DataFrame()

    # No global dropna: we'll do pairwise NA handling for p-values
    df_clean = df[all_cols]

    # correlation matrix (pairwise, via pandas)
    corr_method = 'spearman'
    corr_matrix = df_clean.corr(method=corr_method)

    # p-value matrix (pairwise with masks)
    n = len(all_cols)
    pval_matrix = pd.DataFrame(np.ones((n, n)), index=all_cols, columns=all_cols)

    for i, col1 in enumerate(all_cols):
        x = df_clean[col1].values
        for j, col2 in enumerate(all_cols):
            if i == j:
                pval_matrix.iloc[i, j] = 1.0
                continue
            y = df_clean[col2].values

            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                p = 1.0
            else:
                # constant or almost-constant vectors -> undefined correlation
                if np.nanstd(x[m]) < 1e-12 or np.nanstd(y[m]) < 1e-12:
                    p = 1.0
                else:
                    _, p = stats.spearmanr(x[m], y[m])
            pval_matrix.iloc[i, j] = p
    show_stars = len(features) <= 12
    # build annotated labels with significance
    # * for p<.05, ** for p<.01 (off-diagonal only)
    annot = corr_matrix.copy().astype(str)
    for i, c1 in enumerate(all_cols):
        for j, c2 in enumerate(all_cols):
            r = corr_matrix.iloc[i, j]
            if i == j or not np.isfinite(r):
                annot.iloc[i, j] = "" if i != j else f"{r:.2f}"
                continue
            p = pval_matrix.iloc[i, j]
            if show_stars:
                stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
                annot.iloc[i, j] = f"{r:.2f}{stars}"
            else:
                annot.iloc[i, j] = f"{r:.2f}"

    # --- plot ---
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        corr_matrix,
        annot=annot.values, # strings w/ stars
        fmt='', # because annot already formatted
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": .75},
        ax=ax,
        annot_kws={"size": 10}
    )
    ax.tick_params(labelsize=12)
    meth = "Spearman"
    ax.set_title(f'Feature Correlations with Emotions ({meth})\n* p<0.05, ** p<0.01 (uncorrected)',
                 fontsize=14)
    plt.tight_layout()

    # save
    suffix = {'high':'High','mid':'Mid','key':'Keys'}.get(level, 'Mode')
    out_name = f'correlation_heatmap_{suffix}.png'
    plt.savefig(VIZ_PATH / out_name, dpi=300)
    plt.show()

    # brief textual summary (top-3 per emotion by |r|) ---
    print("\n=== Key Findings (uncorrected p-values) ===")
    for emotion in emotions:
        print(f"\n{emotion.replace('_', ' ').title()}:")
        # sort by absolute correlation (drop emotions themselves)
        emotion_corr = corr_matrix[emotion].drop(emotions, errors='ignore').abs().sort_values(ascending=False)
        for feat, corr_val in emotion_corr.head(3).items():
            p_val = pval_matrix.loc[feat, emotion]
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            strength = "weak" if abs(corr_val) < 0.3 else "moderate" if abs(corr_val) < 0.5 else "strong"
            print(f"  - {feat}: r={corr_val:.3f}{sig} ({strength})")

    return corr_matrix

def plot_pca_and_umap(df: pd.DataFrame, VIZ_PATH: Path,
                      exclude_cols: Optional[List[str]] = None):
    """Automatically selects features and creates a dashboard comparing PCA and UMAP, colored by V & A"""
    # --- 1. Automatic Feature Selection ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_exclude = ['song_id', 'valence_mean', 'arousal_mean', 'valence_std', 'arousal_std']
    if exclude_cols:
        default_exclude.extend(exclude_cols)
    
    # Combined feature selection and removed the print statement
    features = [
        col for col in numeric_cols
        if col not in default_exclude
        and 'key' not in col
        and col != 'bpm'
    ]
    X = df[features]
    y_valence = df['valence_mean']
    y_arousal = df['arousal_mean']

    # Handle Missing Values
    if X.isnull().sum().sum() > 0:
        nan_count = X.isnull().sum().sum()
        print(f"\nWarning: Found {nan_count} missing values in feature data.")
        print("Imputing with column medians before scaling.")
        # Fill NaN in each column with that column's median
        X = X.fillna(X.median())

    # Scaling and Dimensionality Reduction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=['PC1', 'PC2'], index=features
    )

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # --- 3. Create and Display the 2x2 Dashboard Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Feature Space Analysis Dashboard", fontsize=20)

    # Row 1: PCA Plots
    axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_valence, cmap='coolwarm', alpha=0.7)
    axes[0, 0].set_title('PCA Colored by Valence')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_arousal, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('PCA Colored by Arousal')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')

    # Row 2: UMAP Plots
    axes[1, 0].scatter(X_umap[:, 0], X_umap[:, 1], c=y_valence, cmap='coolwarm', alpha=0.7)
    axes[1, 0].set_title('UMAP Colored by Valence')
    axes[1, 0].set_xlabel('UMAP Dimension 1')
    axes[1, 0].set_ylabel('UMAP Dimension 2')

    axes[1, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=y_arousal, cmap='viridis', alpha=0.7)
    axes[1, 1].set_title('UMAP Colored by Arousal')
    axes[1, 1].set_xlabel('UMAP Dimension 1')

    # Add colorbars and grids
    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
    fig.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Valence')
    fig.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Arousal')
    fig.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Valence')
    fig.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Arousal')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return loadings

def analyze_genre_emotions(df: pd.DataFrame, metadata_df: pd.DataFrame,
                                   VIZ_PATH: Path, min_songs: int = 10) -> (pd.DataFrame, dict):
    """Analyze and visualize emotion patterns by genre."""
    # Merge with metadata
    merged = df.merge(metadata_df[['song_id', 'Genre']], on='song_id', how='left')
    
    # Clean and extract the primary genre
    merged['primary_genre'] = merged['Genre'].fillna('Unknown').str.split(',').str[0].str.strip()
    
    # Group by genre and calculate statistics
    genre_stats = merged.groupby('primary_genre').agg({
        'valence_mean': ['mean', 'std', 'count'],
        'arousal_mean': ['mean', 'std', 'count'],
        'valence_std': 'mean',
        'arousal_std': 'mean'
    }).round(3)
    
    # Filter out genres with fewer than min_songs
    valid_genres = genre_stats[genre_stats[('valence_mean', 'count')] >= min_songs].index
    genre_stats = genre_stats.loc[valid_genres]
    
    fig = plt.figure(figsize=(18, 32))
    gs = GridSpec(4, 1, height_ratios=[1.5, 1, 1, 1])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    ax = axes[0]
    genres_to_plot = [g for g in genre_stats.index if g != 'Unknown']
    cmap = plt.colormaps.get_cmap('tab20')
    color_list = [cmap(i) for i in np.linspace(0, 1, len(genres_to_plot))]

    for i, genre in enumerate(genres_to_plot):
        ax.scatter(genre_stats.loc[genre, ('valence_mean', 'mean')],
                   genre_stats.loc[genre, ('arousal_mean', 'mean')],
                   s=genre_stats.loc[genre, ('valence_mean', 'count')] * 3,
                   alpha=0.8,
                   label=genre,
                   color=color_list[i])

    ax.axhline(df['arousal_mean'].median(), color='gray', linestyle='--', alpha=0.5, label='Median Arousal')
    ax.axvline(df['valence_mean'].median(), color='gray', linestyle='--', alpha=0.5, label='Median Valence')
    ax.set_xlabel('Average Valence', fontsize=16)
    ax.set_ylabel('Average Arousal', fontsize=16)
    ax.set_title('Genre Emotion Space (size = number of songs)', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Genres', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Valence by genre
    ax = axes[1]
    genre_valence = genre_stats[('valence_mean', 'mean')].sort_values()
    valence_colors = ['red' if v < df['valence_mean'].median() else 'green' for v in genre_valence.values]
    ax.barh(range(len(genre_valence)), genre_valence.values, color=valence_colors, alpha=0.6)
    ax.set_yticks(range(len(genre_valence)))
    ax.set_yticklabels(genre_valence.index, fontsize=14)
    ax.axvline(df['valence_mean'].median(), color='black', linestyle='--', alpha=0.7, label='Overall median')
    ax.set_xlabel('Average Valence', fontsize=16)
    ax.set_title('Genres Ranked by Valence', fontsize=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    # 3. Arousal by genre
    ax = axes[2]
    genre_arousal = genre_stats[('arousal_mean', 'mean')].sort_values()
    arousal_colors = ['blue' if a < df['arousal_mean'].median() else 'orange' for a in genre_arousal.values]
    ax.barh(range(len(genre_arousal)), genre_arousal.values, color=arousal_colors, alpha=0.6)
    ax.set_yticks(range(len(genre_arousal)))
    ax.set_yticklabels(genre_arousal.index, fontsize=14)
    ax.axvline(df['arousal_mean'].median(), color='black', linestyle='--', alpha=0.7, label='Overall median')
    ax.set_xlabel('Average Arousal', fontsize=16)
    ax.set_title('Genres Ranked by Arousal', fontsize=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    # 4. Emotion variability by genre
    ax = axes[3]
    total_var = genre_stats[('valence_std', 'mean')] + genre_stats[('arousal_std', 'mean')]
    sorted_genres = total_var.sort_values(ascending=False).index
    genre_stats_sorted = genre_stats.loc[sorted_genres]
    
    x = np.arange(len(genre_stats_sorted))
    width = 0.35
    ax.bar(x - width/2, genre_stats_sorted[('valence_std', 'mean')], width,
           label='Valence variability', alpha=0.7, color='coral')
    ax.bar(x + width/2, genre_stats_sorted[('arousal_std', 'mean')], width,
           label='Arousal variability', alpha=0.7, color='skyblue')
    
    ax.set_ylabel('Average Standard Deviation', fontsize=16)
    ax.set_title('Emotional Variability by Genre (Higher = more listener disagreement)', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(genre_stats_sorted.index, rotation=45, ha='right', fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent overlap and make space for the legend
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(VIZ_PATH / 'genre_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Capture summary stats in a dictionary
    summary = {
        "most_positive_genres": genre_stats[('valence_mean', 'mean')].nlargest(3),
        "most_energetic_genres": genre_stats[('arousal_mean', 'mean')].nlargest(3),
        "most_variable_genres": total_var.nlargest(3)
    }
    
    return genre_stats, summary

def find_emotion_outliers(df: pd.DataFrame, metadata_df: pd.DataFrame,
                         emotion: str = 'valence_mean', 
                         std_threshold: float = 2.0) -> pd.DataFrame:
    """
    Find and display interesting outliers with their metadata.
    Useful for case study analysis.
    """
    merged = df.merge(metadata_df[['song_id', 'Genre', 'Artist', 'Track']], 
                     on='song_id', how='left')
    
    # Find outliers
    mean = df[emotion].mean()
    std = df[emotion].std()
    
    high_outliers = merged[merged[emotion] > mean + std_threshold * std]
    low_outliers = merged[merged[emotion] < mean - std_threshold * std]
    
    print(f"\n=== {emotion.replace('_', ' ').title()} Outliers ===")
    print(f"Mean: {mean:.2f}, Std: {std:.2f}")
    
    print(f"\nHigh {emotion} (> {mean + std_threshold * std:.2f}):")
    for _, row in high_outliers.head(5).iterrows():
        print(f"  - {row['Track']} - {row['Artist']} ({row['Genre']}): {row[emotion]:.2f}")
    
    print(f"\nLow {emotion} (< {mean - std_threshold * std:.2f}):")
    for _, row in low_outliers.head(5).iterrows():
        print(f"  - {row['Track']} - {row['Artist']} ({row['Genre']}): {row[emotion]:.2f}")
    
    # High variability tracks
    if '_mean' in emotion:
        std_col = emotion.replace('_mean', '_std')
        if std_col in df.columns:
            high_var = merged.nlargest(5, std_col)
            print(f"\nHigh variability in {emotion.replace('_mean', '')}:")
            for _, row in high_var.iterrows():
                print(f"  - {row['Track']} - {row['Artist']}: std={row[std_col]:.2f}")
    
    return pd.concat([high_outliers, low_outliers])

# data rows validation
def analyze_raw_data_distribution(features_path: Path) -> Dict:
    """
    Analyze the distribution of rows across all CSV files.
    This tells us how many temporal frames each song has.
    """
    csv_files = sorted(features_path.glob("*.csv"))
    
    row_counts = {}
    column_counts = {}
    failed_files = []
    
    print(f"Analyzing {len(csv_files)} CSV files...")
    
    for csv_path in tqdm(csv_files, desc="Counting rows"):
        try:
            sid = int(csv_path.stem)
            df = load_opensmile_csv(csv_path, sep=';')
            
            # Store raw row count (before cleaning)
            row_counts[sid] = len(df)
            column_counts[sid] = len(df.columns)
            
        except Exception as e:
            failed_files.append(csv_path.stem)
    
    # Compute statistics
    row_values = list(row_counts.values())
    
    stats = {
        'total_files': len(csv_files),
        'successful': len(row_counts),
        'failed': len(failed_files),
        'row_stats': {
            'mean': np.mean(row_values),
            'std': np.std(row_values),
            'min': np.min(row_values),
            'max': np.max(row_values),
            'median': np.median(row_values),
            'q25': np.percentile(row_values, 25),
            'q75': np.percentile(row_values, 75)
        },
        'row_distribution': {},
        'test_vs_train': {}
    }
    
    # Distribution analysis
    thresholds = [90, 92, 100, 300, 400]
    for threshold in thresholds:
        count_above = sum(1 for r in row_values if r > threshold)
        stats['row_distribution'][f'above_{threshold}'] = {
            'count': count_above,
            'percentage': 100 * count_above / len(row_values)
        }
    
    # Test vs Train analysis
    test_rows = [r for sid, r in row_counts.items() if sid > 2000]
    train_rows = [r for sid, r in row_counts.items() if sid <= 2000]
    
    stats['test_vs_train'] = {
        'test_songs': {
            'count': len(test_rows),
            'mean_rows': np.mean(test_rows) if test_rows else 0,
            'std_rows': np.std(test_rows) if test_rows else 0,
            'above_92': sum(1 for r in test_rows if r > 92)
        },
        'train_songs': {
            'count': len(train_rows),
            'mean_rows': np.mean(train_rows) if train_rows else 0,
            'std_rows': np.std(train_rows) if train_rows else 0,
            'above_92': sum(1 for r in train_rows if r > 92)
        }
    }
    
    return stats, row_counts

def visualize_row_distribution(row_counts: Dict, VIZ_PATH: Path):
    """Create visualization of row distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Convert to arrays
    all_rows = list(row_counts.values())
    test_rows = [r for sid, r in row_counts.items() if sid > 2000]
    train_rows = [r for sid, r in row_counts.items() if sid <= 2000]
    
    # 1. Test vs Train comparison
    ax = axes[0]
    ax.hist([train_rows, test_rows], bins=20, label=['Train (<=2000)', 'Test (>2000)'], 
            alpha=0.7, edgecolor='black')
    ax.axvline(92, color='red', linestyle='--', label='92 rows threshold')
    ax.set_xlabel('Number of rows')
    ax.set_ylabel('Number of songs')
    ax.set_title('Train vs Test Row Distribution')
    ax.legend()
    
    # 2. Cumulative distribution
    ax = axes[1]
    sorted_all = np.sort(all_rows)
    cumulative = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
    ax.plot(sorted_all, cumulative * 100, label='All songs')
    ax.axvline(92, color='red', linestyle='--', alpha=0.5, label='92 rows')
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of rows')
    ax.set_ylabel('Cumulative % of songs')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'row_distribution_analysis.png', dpi=150)
    plt.show()
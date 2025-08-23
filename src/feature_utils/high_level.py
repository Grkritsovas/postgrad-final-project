import pandas as pd
import numpy as np
from typing import List

def to_high_level(mid_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines granular mid-level features (from level2) into a final, 
    compact set of high-level features.
    """
    high_level_rows = []

    for idx, row in mid_level_df.iterrows():
        features = {}

        # Pitch Consistency -> from Pitch & Voicing
        pitch_parts = [row[c] for c in ['Pitch', 'Voicing'] if c in row and pd.notna(row[c])]
        if pitch_parts:
            features['pitch_consistency'] = np.nanmean(pitch_parts)

        # Vocal Quality -> from Jitter, Shimmer & Harmonicity
        vocal_parts = [row[c] for c in ['Jitter', 'Shimmer', 'Harmonicity'] if c in row and pd.notna(row[c])]
        if vocal_parts:
            features['vocal_quality'] = np.nanmean(vocal_parts)

        # Timbre Complexity -> from all other Timbre sub-categories
        timbre_parts = [
            row[c] for c in [
                'SpectralShape', 'Sharpness', 'Complexity', 'Texture', 
                'MFCC_Formant', 'MFCC_Spectral', 'MFCC_Texture'
            ] if c in row and pd.notna(row[c])]
        if timbre_parts:
            features['timbre_complexity'] = np.nanmean(timbre_parts)

        # Dynamics / Energy -> from Energy & Loudness
        dynamic_parts = [row[c] for c in ['Energy', 'Loudness'] if c in row and pd.notna(row[c])]
        if dynamic_parts:
            features['energy_level'] = np.nanmean(dynamic_parts)

        # Rhythmic Intensity -> from Rhythm's 'Temporal' category
        if 'Temporal' in row and pd.notna(row['Temporal']):
            features['rhythmic_intensity'] = row['Temporal']

        # Tonal Balance -> from the specific band features
        if 'Band_MidHigh' in row and 'Band_LowMid' in row and pd.notna(row['Band_MidHigh']) and pd.notna(row['Band_LowMid']):
            features['tonal_balance'] = row['Band_MidHigh'] - row['Band_LowMid']
        
        high_level_rows.append(features)

    return pd.DataFrame(high_level_rows, index=mid_level_df.index)
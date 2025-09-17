import pandas as pd
import numpy as np
from typing import List

def create_perceptual_features(
    selected_low_level_df: pd.DataFrame, 
    hierarchy_map: pd.DataFrame
) -> pd.DataFrame:
    """Aggregates selected low-level features into perceptual-level features"""
    # Check if perceptual exists in hierarchy_map
    if 'perceptual' not in hierarchy_map.columns:
        print("Warning: 'perceptual' not found in hierarchy_map, returning empty DataFrame")
        return pd.DataFrame()
    
    # Create mapping from base feature name to perceptual category
    feature_to_perceptual = {}
    for _, row in hierarchy_map.iterrows():
        feature_to_perceptual[row['feature']] = row['perceptual']
    
    # Group columns by their perceptual category
    perceptual_groups = {}
    for col in selected_low_level_df.columns:
        # Extract base name (remove descriptor suffix)
        base_name = col.rsplit('_', 1)[0] if '_' in col else col
        
        if base_name in feature_to_perceptual:
            perceptual_cat = feature_to_perceptual[base_name]
            if perceptual_cat not in perceptual_groups:
                perceptual_groups[perceptual_cat] = []
            perceptual_groups[perceptual_cat].append(col)
    
    # Create perceptual features by averaging within each group
    perceptual_features = {}
    for perceptual_cat, cols in perceptual_groups.items():
        if cols:
            perceptual_features[perceptual_cat] = selected_low_level_df[cols].mean(axis=1)
    
    return pd.DataFrame(perceptual_features)

def create_musical_features( selected_low_level_df: pd.DataFrame, hierarchy_map: pd.DataFrame ) -> pd.DataFrame:
    """ Aggregates selected low-level features into more granular mid-level features based on their 'musical' category"""
    # Check if musical exists in hierarchy_map 
    if 'musical' not in hierarchy_map.columns: 
        print("Warning: 'musical' not found in hierarchy_map, returning empty DataFrame")
        return pd.DataFrame()
    # Create mapping from base feature name to musical category 
    feature_to_musical = {} 
    for _, row in hierarchy_map.iterrows(): 
        feature_to_musical[row['feature']] = row['musical']

    # Group columns by their musical category 
    musical_groups = {} 
    for col in selected_low_level_df.columns:
        # Extract base name (remove descriptor suffix) 
        base_name = col.rsplit('_', 1)[0] if '_' in col else col 
        if base_name in feature_to_musical:
            musical_cat = feature_to_musical[base_name]
            if musical_cat not in musical_groups: 
                musical_groups[musical_cat] = [] 
            musical_groups[musical_cat].append(col)
    # Create musical features by averaging within each group
    musical_features = {}
    for musical_cat, cols in musical_groups.items():
        if cols: 
            musical_features[musical_cat] = selected_low_level_df[cols].mean(axis=1)

    return pd.DataFrame(musical_features)
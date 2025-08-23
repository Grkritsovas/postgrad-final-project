from pathlib import Path
import numpy as np
import pandas as pd
import librosa

# madmom compbatability shims
if not hasattr(np, 'float'):
    np.float = float
    np.int = int

import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence

# Now import madmom (shims are applied)
try:
    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False
    print("Warning: madmom not available, key estimation will be disabled")

def canonical_bpm(b, lo=80, hi=160):
    if pd.isna(b) or b == 0: 
        return np.nan
    while b < lo:  
        b *= 2
    while b >= hi: 
        b /= 2
    return round(float(b), 2)

def estimate_bpm(path: Path) -> float:
    try:
        # Convert Path to string if needed
        if isinstance(path, Path):
            path = str(path)
        
        # Load audio
        y, sr = librosa.load(path, mono=True, duration=30)  # Limit to 30 seconds for speed
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        if tempo is not None and tempo > 0:
            return round(float(tempo), 2)
        else:
            return np.nan
            
    except Exception as e:
        print(f"BPM estimation failed: {e}")
        return np.nan

def estimate_key_with_confidence(path: Path):
    """Estimates key, mode, and confidence from an audio file."""
    if not MADMOM_AVAILABLE:
        return None, None, np.nan
    
    try:
        key_proc = CNNKeyRecognitionProcessor()
        probs = key_proc(str(path))
        confidence = float(np.max(probs))
        label = key_prediction_to_label(probs)
        key, mode = label.split()
        return key, mode, confidence
    except Exception as e:
        print(f"Key estimation failed for {path}: {e}")
        return None, None, np.nan

# Legacy function name for compatibility
def estimate_key_mode_confidence(path: Path):
    """Legacy wrapper - use estimate_key_with_confidence instead"""
    return estimate_key_with_confidence(path)
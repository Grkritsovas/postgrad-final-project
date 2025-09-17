import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import warnings
import json
from typing import Dict, Tuple, Optional

warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """audio preprocessor with multiple model configurations."""
    
    # Model-specific configurations
    CONFIGS = {
        'panns': {
            'target_sr': 32000,
            'n_mels': 64,
            'n_fft': 1024,
            'hop_length': 320,
            'duration_secs': 10,  # PANNs typically work with 10s clips
            'normalize': 'per_sample'
        },
        'ast': {
            'target_sr': 16000,
            'n_mels': 128,
            'n_fft': 400,
            'hop_length': 160,
            'duration_secs': 10,
            'normalize': 'global'
        },
        'vggish': {
            'target_sr': 16000,
            'n_mels': 64,
            'n_fft': 400,
            'hop_length': 160,
            'duration_secs': 0.96,  # VGGish uses ~1s patches
            'normalize': 'per_sample'
        },
        'musicnn': {
            'target_sr': 16000,
            'n_mels': 96,
            'n_fft': 512,
            'hop_length': 256,
            'duration_secs': 3,
            'normalize': 'per_sample'
        },
        'clap': {
            'target_sr': 48000,
            'n_mels': 64,
            'n_fft': 1024,
            'hop_length': 480,
            'duration_secs': 10,
            'normalize': 'global'
        }
    }
    
    def __init__(self, model_type: str = 'panns'):
        """Initialize with a specific model configuration."""
        if model_type not in self.CONFIGS:
            raise ValueError(f"Model type must be one of {list(self.CONFIGS.keys())}")
        
        self.config = self.CONFIGS[model_type]
        self.model_type = model_type
        
        # For global normalization
        self.global_mean = None
        self.global_std = None
    
    def load_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """Load audio with error handling for various formats."""
        try:
            # Try librosa first (handles mp3 well)
            waveform, sr = librosa.load(audio_path, sr=self.config['target_sr'], mono=True)
            return waveform
        except:
            try:
                # Fallback to soundfile for other formats
                waveform, sr = sf.read(audio_path)
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)  # Convert to mono
                # Resample if needed
                if sr != self.config['target_sr']:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.config['target_sr'])
                return waveform
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                return None
    
    def segment_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Segment audio into chunks based on model requirements."""
        target_sr = self.config['target_sr']
        duration_secs = self.config['duration_secs']
        target_length = int(target_sr * duration_secs)
        
        # For models that use short segments (like VGGish)
        if self.model_type == 'vggish':
            # Create overlapping segments
            hop_size = target_length // 2  # 50% overlap
            segments = []
            for i in range(0, len(waveform) - target_length, hop_size):
                segments.append(waveform[i:i + target_length])
            if len(segments) == 0:
                # If audio is too short, pad it
                padded = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
                segments.append(padded)
            return np.array(segments)
        
        # For models that use longer segments
        else:
            # For 45s audio, we can extract multiple 10s segments
            segments = []
            for i in range(0, len(waveform) - target_length, target_length):
                segments.append(waveform[i:i + target_length])
            
            # Handle last segment
            if len(waveform) % target_length != 0 and len(segments) < 4:
                last_segment = waveform[-target_length:]
                segments.append(last_segment)
            
            if len(segments) == 0:
                # If audio is shorter than target, pad it
                padded = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
                segments.append(padded)
            
            return np.array(segments)
    
    def compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram with model-specific parameters."""
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.config['target_sr'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels'],
            fmin=0,
            fmax=self.config['target_sr'] // 2
        )
        
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def normalize(self, log_mel: np.ndarray) -> np.ndarray:
        """Apply normalization based on model requirements."""
        if self.config['normalize'] == 'per_sample':
            # Normalize each spectrogram independently
            mean = np.mean(log_mel)
            std = np.std(log_mel) + 1e-6
            return (log_mel - mean) / std
        
        elif self.config['normalize'] == 'global':
            # Use global statistics (need to be computed first)
            if self.global_mean is None or self.global_std is None:
                # Fallback scaling if global stats are not computed
                return (log_mel + 80) / 80  # Simple scaling for dB values
            return (log_mel - self.global_mean) / (self.global_std + 1e-6)
        
        return log_mel
    
    def process_file(self, audio_path: Path) -> Optional[Dict]:
        """Process a single audio file."""
        # Load audio
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        # Segment audio
        segments = self.segment_audio(waveform)
        
        # Process each segment
        spectrograms = []
        for segment in segments:
            log_mel = self.compute_mel_spectrogram(segment)
            log_mel = self.normalize(log_mel)
            spectrograms.append(log_mel)
        
        return {
            'spectrograms': np.array(spectrograms),
            'song_id': audio_path.stem,
            'n_segments': len(spectrograms)
        }
    
    def compute_global_stats(self, audio_files: list, sample_size: int = 100):
        """Compute global mean and std from a sample of files."""
        print(f"Computing global statistics from {min(sample_size, len(audio_files))} files...")
        
        # Sample files
        sample_files = np.random.choice(audio_files, 
                                      size=min(sample_size, len(audio_files)), 
                                      replace=False)
        
        all_values = []
        for audio_path in tqdm(sample_files, desc="Computing stats"):
            waveform = self.load_audio(audio_path)
            if waveform is not None:
                # Take a segment from the middle
                target_length = int(self.config['target_sr'] * self.config['duration_secs'])
                if len(waveform) > target_length:
                    start = (len(waveform) - target_length) // 2
                    waveform = waveform[start:start + target_length]
                
                log_mel = self.compute_mel_spectrogram(waveform)
                all_values.append(log_mel.flatten())
        
        if all_values:
            all_values = np.concatenate(all_values)
            self.global_mean = np.mean(all_values)
            self.global_std = np.std(all_values)
            print(f"Global stats - Mean: {self.global_mean:.2f}, Std: {self.global_std:.2f}")
        else:
            print("Warning: Could not compute global stats, using defaults")
            self.global_mean = -40
            self.global_std = 20
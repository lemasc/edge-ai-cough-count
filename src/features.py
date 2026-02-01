import numpy as np
from scipy import stats, signal
import librosa

def extract_audio_features(audio_window, fs=16000):
    """
    Extract 65 audio features from single window

    1. **MFCC (52)**: 13 coefficients × 4 statistics (mean, std, min, max)
    2. **Spectral (10)**: Centroid, rolloff, bandwidth, flatness, contrast, PSD features, spectral spread/skewness/kurtosis
    3. **Time-domain (3)**: Zero-crossing rate, RMS energy, crest factor
    
    Args:
        audio_window: 1D array of audio samples
        fs: Sampling frequency (16000 Hz)
    
    Returns:
        np.array: 65 features
    """
    features = []
    
    # MFCC features (52)
    mfccs = librosa.feature.mfcc(y=audio_window, sr=fs, n_mfcc=13)
    for coef in mfccs:
        features.extend([np.mean(coef), np.std(coef), np.min(coef), np.max(coef)])
    
    # Spectral features (10)
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio_window, sr=fs)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=audio_window, sr=fs)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio_window, sr=fs)))
    features.append(np.mean(librosa.feature.spectral_flatness(y=audio_window)))
    features.append(np.mean(librosa.feature.spectral_contrast(y=audio_window, sr=fs)))
    
    # PSD-based features
    f, psd = signal.welch(audio_window, fs=fs)
    features.append(np.sum(psd))  # Total power
    dom_freq_idx = np.argmax(psd)
    features.append(f[dom_freq_idx])  # Dominant frequency
    
    # Spectral spread, skewness, kurtosis
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_mean = np.sum(f * psd_norm)
    features.append(np.sqrt(np.sum(((f - spectral_mean)**2) * psd_norm)))  # Spread
    features.append(np.sum(((f - spectral_mean)**3) * psd_norm))  # Skewness
    features.append(np.sum(((f - spectral_mean)**4) * psd_norm))  # Kurtosis
    
    # Time-domain features (3)
    features.append(librosa.feature.zero_crossing_rate(audio_window)[0].mean())
    rms = np.sqrt(np.mean(audio_window**2))
    features.append(rms)
    features.append(np.max(np.abs(audio_window)) / (rms + 1e-10))  # Crest factor
    
    return np.array(features)

def extract_imu_features(imu_window):
    """
    Extract 40 IMU features

    For 8 signals (3 accel + accel_L2 + 3 gyro + gyro_L2):
        - Line length, zero-crossing rate, kurtosis, crest factor, RMS = 5 features per signal
    
    Args:
        imu_window: (40, 6) array - [Accel_x, Accel_y, Accel_z, Gyro_Y, Gyro_P, Gyro_R]
    
    Returns:
        np.array: 40 features (8 signals × 5 features)
    """
    # Subtract mean per channel (paper requirement)
    imu_centered = imu_window - np.mean(imu_window, axis=0, keepdims=True)
    
    # Compute L2 norms
    accel_l2 = np.linalg.norm(imu_centered[:, 0:3], axis=1)
    gyro_l2 = np.linalg.norm(imu_centered[:, 3:6], axis=1)
    
    # Stack all 8 signals
    signals = np.column_stack([
        imu_centered[:, 0], imu_centered[:, 1], imu_centered[:, 2], accel_l2,
        imu_centered[:, 3], imu_centered[:, 4], imu_centered[:, 5], gyro_l2
    ])
    
    features = []
    for i in range(8):
        sig = signals[:, i]
        
        # Line length
        features.append(np.sum(np.abs(np.diff(sig))))
        
        # Zero crossing rate
        features.append(np.sum(np.diff(np.sign(sig)) != 0) / len(sig))
        
        # Kurtosis
        features.append(stats.kurtosis(sig))
        
        # Crest factor
        rms = np.sqrt(np.mean(sig**2))
        features.append(np.max(np.abs(sig)) / (rms + 1e-10))
        
        # RMS power
        features.append(rms)
    
    return np.array(features)
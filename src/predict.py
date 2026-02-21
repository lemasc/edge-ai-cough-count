"""predict.py: Shared sliding-window prediction pipeline for cough detection models."""

import numpy as np

from features import extract_audio_features, extract_imu_features
from helpers import FS_AUDIO, FS_IMU


def extract_features_for_window(audio_window, imu_window, modality='multimodal'):
    """
    Extract features from a single window of audio and IMU data.

    Args:
        audio_window: (N_audio,) audio samples
        imu_window: (N_imu, 6) IMU samples
        modality: 'imu', 'audio', or 'multimodal'

    Returns:
        np.array: Feature vector
    """
    features = []

    if modality in ['audio', 'multimodal']:
        audio_feat = extract_audio_features(audio_window, fs=FS_AUDIO)
        audio_feat = np.nan_to_num(audio_feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(audio_feat)

    if modality in ['imu', 'multimodal']:
        imu_feat = extract_imu_features(imu_window)
        imu_feat = np.nan_to_num(imu_feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(imu_feat)

    return np.concatenate(features)


def sliding_window_predict(audio, imu, model_data, modality='multimodal',
                           window_len=0.4, hop_size=0.05, threshold=None):
    """
    Apply model to continuous recording using sliding windows.

    Args:
        audio: (N_audio,) audio samples
        imu: (N_imu, 6) IMU samples
        model_data: Dict with 'model', 'scaler', 'threshold'
        modality: 'imu', 'audio', or 'multimodal'
        window_len: Window length in seconds
        hop_size: Hop size in seconds
        threshold: Classification threshold (None = use optimal from model)

    Returns:
        predictions: List of (start_time, end_time, probability) tuples (only above threshold)
        all_probs: Array of probabilities for each window
        window_times: Array of window center times
        all_windows: List of (start, end, center, prob) for ALL windows
    """
    model = model_data['model']
    scaler = model_data['scaler']
    if threshold is None:
        threshold = model_data['threshold']

    audio_win_samples = int(window_len * FS_AUDIO)
    audio_hop_samples = int(hop_size * FS_AUDIO)
    imu_win_samples = int(window_len * FS_IMU)
    imu_hop_samples = int(hop_size * FS_IMU)

    n_windows = (len(audio) - audio_win_samples) // audio_hop_samples + 1
    features_list = []
    window_times = []

    for i in range(n_windows):
        audio_start = i * audio_hop_samples
        audio_end = audio_start + audio_win_samples
        imu_start = i * imu_hop_samples
        imu_end = imu_start + imu_win_samples

        if audio_end > len(audio) or imu_end > len(imu):
            break

        audio_window = audio[audio_start:audio_end]
        imu_window = imu[imu_start:imu_end, :]

        features = extract_features_for_window(audio_window, imu_window, modality)
        features_list.append(features)

        center_time = (audio_start + audio_win_samples / 2) / FS_AUDIO
        window_times.append(center_time)

    X = np.array(features_list)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    all_windows = []
    predictions = []
    for i, (prob, center) in enumerate(zip(probs, window_times)):
        start = center - window_len / 2
        end = center + window_len / 2
        all_windows.append((start, end, center, prob))
        if prob >= threshold:
            predictions.append((start, end, prob))

    return predictions, probs, np.array(window_times), all_windows


def merge_detections(predictions, gap_threshold=0.3):
    """
    Merge consecutive detections that are close together.

    Args:
        predictions: List of (start, end, prob) tuples
        gap_threshold: Maximum gap between events to merge (seconds)

    Returns:
        merged: List of (start, end, max_prob) tuples
    """
    if not predictions:
        return []

    sorted_preds = sorted(predictions, key=lambda x: x[0])

    merged = []
    current_start, current_end, current_prob = sorted_preds[0]

    for start, end, prob in sorted_preds[1:]:
        if start - current_end <= gap_threshold:
            current_end = max(current_end, end)
            current_prob = max(current_prob, prob)
        else:
            merged.append((current_start, current_end, current_prob))
            current_start, current_end, current_prob = start, end, prob

    merged.append((current_start, current_end, current_prob))

    return merged


def create_dummy_audio(duration_seconds):
    """
    Create dummy audio data for when only IMU is provided.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        audio: (N,) zero-filled audio samples
    """
    n_samples = int(duration_seconds * FS_AUDIO)
    return np.zeros(n_samples, dtype=np.float32)


def create_dummy_imu(duration_seconds):
    """
    Create dummy IMU data for when only audio is provided.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        imu: (N, 6) zero-filled IMU samples
    """
    n_samples = int(duration_seconds * FS_IMU)
    return np.zeros((n_samples, 6), dtype=np.float32)

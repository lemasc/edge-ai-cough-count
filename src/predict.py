"""predict.py: Shared sliding-window prediction pipeline for cough detection models."""

import numpy as np
from joblib import Parallel, delayed

from features import extract_audio_features, extract_imu_features
from helpers import FS_AUDIO, FS_IMU, segment_cough


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


def _extract_one_window(i, audio, imu, audio_hop_samples, audio_win_samples,
                        imu_hop_samples, imu_win_samples, modality):
    """Extract features for one aligned audio/IMU window (for parallel execution)."""
    audio_start = i * audio_hop_samples
    imu_start = i * imu_hop_samples
    return extract_features_for_window(
        audio[audio_start:audio_start + audio_win_samples],
        imu[imu_start:imu_start + imu_win_samples, :],
        modality,
    )


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

    n_windows_raw = (len(audio) - audio_win_samples) // audio_hop_samples + 1
    n_windows = sum(
        1 for i in range(n_windows_raw)
        if (i * audio_hop_samples + audio_win_samples <= len(audio))
        and (i * imu_hop_samples + imu_win_samples <= len(imu))
    )

    features_list = Parallel(n_jobs=2, prefer='threads')(
        delayed(_extract_one_window)(
            i, audio, imu,
            audio_hop_samples, audio_win_samples,
            imu_hop_samples, imu_win_samples,
            modality,
        )
        for i in range(n_windows)
    )

    window_times = np.array([
        (i * audio_hop_samples + audio_win_samples / 2) / FS_AUDIO
        for i in range(n_windows)
    ])

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

    return predictions, probs, window_times, all_windows


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


def refine_cough_events(audio, candidate_segments, fs_audio=FS_AUDIO,
                        t_dedup=0.23, t_bout=0.55):
    """
    Post-process merged candidate segments using Cough-E-style peak refinement.

    Args:
        audio: (N,) audio samples at fs_audio Hz
        candidate_segments: List of (start, end, prob) from merge_detections()
        fs_audio: Audio sampling rate in Hz
        t_dedup: Min inter-peak gap to count as a new cough (seconds)
        t_bout: Max inter-peak gap within a cough bout (seconds)

    Returns:
        refined: List of (start, end, prob) tuples
    """
    if not candidate_segments:
        return []

    _, _, starts_idx, ends_idx, _, peak_locs_idx = segment_cough(
        audio, fs_audio,
        cough_padding=0.1,
        min_cough_len=0.1,
        th_l_multiplier=0.1,
        th_h_multiplier=2.0,
    )

    if len(starts_idx) == 0:
        return candidate_segments

    seg_starts = starts_idx / fs_audio
    seg_ends = ends_idx / fs_audio
    seg_peaks = np.array(peak_locs_idx) / fs_audio

    cand_tol = 0.3
    filtered = []
    for seg_st, seg_et, seg_pt in zip(seg_starts, seg_ends, seg_peaks):
        for cand_st, cand_et, prob in candidate_segments:
            if cand_st - cand_tol <= seg_pt <= cand_et + cand_tol:
                filtered.append([seg_st, seg_et, seg_pt, prob])
                break

    if not filtered:
        return candidate_segments

    filtered.sort(key=lambda x: x[2])

    deduped = [filtered[0][:]]
    for current in filtered[1:]:
        prev = deduped[-1]
        if current[2] - prev[2] < t_dedup:
            prev[1] = max(prev[1], current[1])
            prev[3] = max(prev[3], current[3])
        else:
            deduped.append(current[:])

    final = []
    for i, current in enumerate(deduped):
        start_t, end_t, peak_t, prob = current
        if i + 1 < len(deduped):
            next_start = deduped[i + 1][0]
            next_peak = deduped[i + 1][2]
            if next_peak - peak_t < t_bout:
                end_t = next_start
        final.append((start_t, end_t, prob))

    return final


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

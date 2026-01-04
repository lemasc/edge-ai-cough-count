# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for edge-AI cough counting using multimodal biosignals (acoustic and kinematic sensors). The repository provides tools for working with a public dataset of ~4 hours of biosignal data containing 4,300 annotated cough events, along with various non-cough sounds and motion scenarios.

Dataset available at: https://zenodo.org/record/7562332

## Core Architecture

### Dataset Structure

The `public_dataset/` directory is organized hierarchically:

```
public_dataset/
├── {subject_id}/                    # 15 subjects total
│   ├── biodata.json                 # Subject metadata (Gender, BMI)
│   └── trial_{1,2,3}/               # 3 trials per subject
│       └── mov_{sit,walk}/          # Movement conditions
│           └── background_noise_{music,nothing,someone_else_cough,traffic}/  # Audio noise conditions
│               └── {cough,laugh,deep_breathing,throat_clearing}/  # Sound types
│                   ├── outward_facing_mic.wav     # Audio from air-facing microphone
│                   ├── body_facing_mic.wav        # Audio from skin-facing microphone
│                   ├── imu.csv                    # IMU data (accel x,y,z + gyro Y,P,R)
│                   └── ground_truth.json          # Cough annotations (only for cough recordings)
```

### Sampling Frequencies

- Audio: 16000 Hz (`FS_AUDIO`)
- IMU: 100 Hz (`FS_IMU`)

### Key Modules

**`src/helpers.py`** - Core utilities for dataset navigation and signal processing:

- Enums for accessing dataset hierarchy: `Trial`, `Movement`, `Noise`, `Sound`, `IMU_Signal`
- `load_audio()`: Load dual-microphone audio signals (air and skin mics)
- `load_annotation()`: Load ground-truth cough start/end times from JSON
- `load_imu()`: Load IMU data into an `IMU` object with 6 channels (accel x,y,z + gyro Y,P,R)
- `IMU` class: Container for IMU signals with normalization and plotting methods
- `segment_cough()`: Hysteresis-based cough segmentation using signal power
- `delineate_imu()`: Peak/valley detection in IMU signals using derivatives

**`src/dataset_gen.py`** - Dataset generation for ML training:

- `get_cough_windows()`: Extract fixed-length windows around labeled coughs with data augmentation via random temporal shifts
- `get_non_cough_windows()`: Extract random non-cough segments from other sound types
- `get_samples_for_subject()`: Generate balanced cough/non-cough dataset for a single subject across all conditions
- Returns shape: audio `(N, window_len*16000, 2)`, IMU `(N, window_len*100, 6)`, labels `(N,)`

### Jupyter Notebooks

**`notebooks/Segmentation_Augmentation.ipynb`**:

- Demonstrates segmenting raw biosignals into ML-ready windows
- Shows data augmentation by randomly shifting coughs within windows
- Example: `get_samples_for_subject(data_folder, "14287", window_len=0.7, aug_factor=2)`

**`notebooks/Cough_Annotation.ipynb`**:

- Explains semi-automatic fine-grained cough labeling methodology
- Useful for teams wanting to merge datasets with consistent labeling

**`notebooks/Compute_Success_Metrics.ipynb`**:

- Event-based evaluation using the `timescoring` library
- Computes TP, FP, FN, Sensitivity, Precision, F1 scores
- Recommended parameters:
  - `toleranceStart = 0.25`
  - `toleranceEnd = 0.25`
  - `minOverlap = 0.1`
  - `maxEventDuration = 0.6`
  - `minDurationBetweenEvents = 0`

## Working with the Dataset

### Loading a Recording

```python
from helpers import *

# Load audio (returns air_mic, skin_mic)
air, skin = load_audio(data_folder, subject_id="14287",
                       trial=Trial.ONE, mov=Movement.SIT,
                       noise=Noise.NONE, sound=Sound.COUGH)

# Load ground truth (only for cough recordings)
start_times, end_times = load_annotation(data_folder, "14287",
                                         Trial.ONE, Movement.SIT,
                                         Noise.NONE, Sound.COUGH)

# Load IMU data
imu = load_imu(data_folder, "14287", Trial.ONE,
               Movement.SIT, Noise.NONE, Sound.COUGH)
```

### Iterating Through Dataset

Use the Enum classes to systematically iterate:

```python
for trial in Trial:
    for mov in Movement:
        for noise in Noise:
            for sound in Sound:
                # Process recordings
```

### Ground Truth Format

`ground_truth.json` structure:

```json
{
  "start_times": [0.638, 0.9626875, ...],
  "end_times": [0.9626875, 1.32, ...]
}
```

Times are in seconds from the start of the recording.

## Citation

If using this dataset/code, cite:

```
Orlandic, L., Thevenot, J., Teijeiro, T., & Atienza, D. (2023).
A Multimodal Dataset for Automatic Edge-AI Cough Detection.
2023 45th Annual International Conference of the IEEE Engineering
in Medicine & Biology Society (EMBC), 1-7.
```

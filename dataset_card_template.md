---
task_categories:
  - audio-classification
  - other
language:
  - en
tags:
  - healthcare
  - biosignals
  - cough-detection
  - edge-ai
  - multimodal
size_categories:
  - n<1K
---

# Edge-AI Cough Counting Dataset

## Dataset Description

### Dataset Summary

The Edge-AI Cough Counting Dataset is the first publicly accessible multimodal biosignal dataset for automatic cough detection and counting. The dataset contains nearly 4 hours of synchronized acoustic and kinematic sensor data, covering **4,300 annotated cough events** from **15 subjects**.

This dataset was created by the Embedded Systems Laboratory (ESL) at EPFL to advance research in wearable edge-AI systems for privacy-preserving automatic cough counting - an essential biomarker for antitussive therapy efficacy and personalized patient care.

**Key Features:**

- Multimodal: Dual-microphone audio (outward-facing and body-facing) + 6-channel IMU
- Comprehensive: 1,440 recordings covering diverse noise and movement conditions
- Fine-grained: Precise cough start/end timestamps for all cough events
- Realistic: Includes non-cough sounds (laughing, throat clearing, deep breathing) and daily life scenarios

### Supported Tasks

- **Cough Detection:** Binary classification of audio/IMU segments as cough or non-cough
- **Cough Counting:** Event-based detection and counting of cough occurrences
- **Multimodal Fusion:** Research on combining acoustic and kinematic modalities
- **Edge-AI Development:** Training lightweight models for on-device inference

### Languages

Recordings are from English-speaking subjects, though coughing is language-agnostic.

## Dataset Structure

### Data Instances

Each row represents one complete recording (~11-50 seconds, average ~47s) containing:

```python
{
    "outward_facing_mic": {"path": "...", "array": [...], "sampling_rate": 16000},
    "body_facing_mic": {"path": "...", "array": [...], "sampling_rate": 16000},
    "imu": [[...], ...],  # Shape: (n_samples, 6) - accel x,y,z + gyro Y,P,R
    "ground_truth": {
        "start_times": [0.638, 0.962, ...],  # Seconds from recording start
        "end_times": [0.962, 1.32, ...]
    },
    "subject_id": "14287",
    "gender": "Female",
    "bmi": 18.28,
    "trial": 1,  # 1, 2, or 3
    "movement": "sit",  # "sit" or "walk"
    "noise": "nothing",  # "music", "nothing", "someone_else_cough", "traffic"
    "sound": "cough",  # "cough", "laugh", "deep_breathing", "throat_clearing"
    "duration_seconds": 11.25,
    "num_coughs": 10,
    "has_ground_truth": true
}
```

### Data Fields

**Audio:**

- `outward_facing_mic`: Audio from air-facing microphone (16kHz)
- `body_facing_mic`: Audio from skin-facing microphone (16kHz)

**IMU:**

- `imu`: 6-channel IMU data (100Hz) - columns: accel x,y,z, gyro Y,P,R

**Ground Truth:**

- `ground_truth.start_times`: List of cough start times (seconds)
- `ground_truth.end_times`: List of cough end times (seconds)
- Note: Empty lists for non-cough recordings

**Metadata:**

- `subject_id`: Subject identifier (string)
- `gender`: "Male" or "Female"
- `bmi`: Body Mass Index (float)
- `trial`: Trial number (1-3)
- `movement`: Kinematic condition ("sit" or "walk")
- `noise`: Acoustic noise condition ("music", "nothing", "someone_else_cough", "traffic")
- `sound`: Sound type being performed ("cough", "laugh", "deep_breathing", "throat_clearing")

**Computed:**

- `duration_seconds`: Recording duration (float)
- `num_coughs`: Number of coughs in recording (0 for non-cough sounds)
- `has_ground_truth`: Boolean indicating if ground truth annotations are available

### Data Splits

Dataset is split by subject to prevent data leakage:

| Split | Subjects | Recordings | Cough Events | Duration |
| ----- | -------- | ---------- | ------------ | -------- |
| Train | 10       | ~960       | ~3,000       | ~2.7 hrs |
| Val   | 3        | ~288       | ~900         | ~0.8 hrs |
| Test  | 2        | ~192       | ~600         | ~0.5 hrs |

**Split Strategy:** Deterministic subject-level split (seed=42) ensuring no subject appears in multiple splits.

## Dataset Creation

### Curation Rationale

This dataset was created to address the lack of publicly accessible multimodal biosignal data for cough detection research. Existing datasets are either proprietary, audio-only, or lack fine-grained annotations. By providing synchronized audio and IMU data with precise cough timestamps, this dataset enables:

1. Development of privacy-preserving edge-AI algorithms (kinematic sensors don't record speech)
2. Research on multimodal sensor fusion
3. Benchmarking of cough detection algorithms with standardized evaluation metrics

### Source Data

#### Data Collection

- **Equipment:**

  - Dual-microphone setup: outward-facing (air mic) + body-facing (skin mic)
  - 6-axis IMU: 3-axis accelerometer + 3-axis gyroscope
  - Sampling rates: 16kHz (audio), 100Hz (IMU)

- **Protocol:**
  - 15 subjects performed 96 recordings each
  - Each recording: ~47 seconds of specific sound in controlled conditions
  - Variables:
    - 3 trials (repeated measurements)
    - 2 movement conditions: sitting, walking
    - 4 noise conditions: music, silence, someone else coughing, traffic
    - 4 sound types: coughing, laughing, deep breathing, throat clearing

#### Annotations

- **Cough Segmentation:** Semi-automatic fine-grained labeling
  - Automatic detection using hysteresis-based power thresholding
  - Manual refinement for precise start/end times
- **Format:** JSON with start_times and end_times arrays (in seconds)
- **Coverage:** All cough recordings have ground truth annotations

### Personal and Sensitive Information

- Only anonymized subject IDs are provided
- Gender and BMI are included as metadata (relevant for biosignal analysis)
- Audio recordings contain only controlled sounds (no speech/conversations)
- No other personally identifiable information is included

## Considerations for Using the Data

### Social Impact

**Positive Applications:**

- Improved telemedicine and remote patient monitoring
- Objective assessment of cough treatment efficacy
- Privacy-preserving health monitoring (using IMU instead of continuous audio)

**Potential Misuse:**

- Health data should not be used for discrimination
- Models trained on this dataset may not generalize to all populations (limited diversity)

### Limitations

- **Limited Diversity:** 15 subjects may not capture full population variability
- **Controlled Conditions:** Recordings in lab settings may not reflect real-world complexity
- **Sensor Specifics:** Performance may vary with different hardware
- **Language/Culture:** All subjects are from similar demographic (EPFL community)

### Recommendations

- Use subject-level cross-validation or provided splits to prevent data leakage
- Evaluate models using event-based metrics (not just frame-level accuracy)
- Consider combining with other datasets for robustness
- Validate on real-world deployment scenarios before clinical use

## Additional Information

### Dataset Curators

Lara Orlandic, Jérôme Thevenot, Tomas Teijeiro, David Atienza (Embedded Systems Laboratory, EPFL)

### Licensing Information

[Please specify - likely CC BY 4.0 or similar permissive license]

### Citation Information

If you use this dataset, please cite:

```bibtex
@inproceedings{orlandic_multimodal_2023,
    address = {Sydney, Australia},
    title = {A {Multimodal} {Dataset} for {Automatic} {Edge}-{AI} {Cough} {Detection}},
    url = {https://ieeexplore.ieee.org/document/10340413/},
    doi = {10.1109/EMBC40787.2023.10340413},
    booktitle = {2023 45th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
    publisher = {IEEE},
    author = {Orlandic, Lara and Thevenot, Jérôme and Teijeiro, Tomas and Atienza, David},
    month = jul,
    year = {2023},
    pages = {1--7},
}
```

### Contact

For questions or suggestions: lara.orlandic@epfl.ch

### Acknowledgements

This dataset was created by the Embedded Systems Laboratory (ESL) at EPFL. Original dataset available at: https://zenodo.org/record/7562332

## Usage Example

```python
from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset("username/edge-ai-cough-count")

# Access splits
train_data = dataset["train"]
val_data = dataset["val"]
test_data = dataset["test"]

# Get a single recording
recording = train_data[0]

# Access audio (automatically loaded as numpy arrays)
outward_mic = recording["outward_facing_mic"]["array"]  # Shape: (n_samples,)
body_mic = recording["body_facing_mic"]["array"]        # Shape: (n_samples,)

# Access IMU data
imu_data = recording["imu"]  # Shape: (n_samples, 6)
accel_x = imu_data[:, 0]
accel_y = imu_data[:, 1]
accel_z = imu_data[:, 2]
gyro_Y = imu_data[:, 3]
gyro_P = imu_data[:, 4]
gyro_R = imu_data[:, 5]

# Access ground truth for cough recordings
if recording["has_ground_truth"]:
    cough_starts = recording["ground_truth"]["start_times"]
    cough_ends = recording["ground_truth"]["end_times"]
    print(f"Recording has {len(cough_starts)} coughs")

# Filter dataset
cough_only = train_data.filter(lambda x: x["sound"] == "cough")
sitting_only = train_data.filter(lambda x: x["movement"] == "sit")

# Extract cough windows for training
def extract_cough_window(example, window_sec=0.7):
    """Extract fixed-length windows around each cough."""
    if not example["has_ground_truth"]:
        return None

    windows = []
    fs_audio = 16000
    fs_imu = 100

    for start, end in zip(example["ground_truth"]["start_times"],
                          example["ground_truth"]["end_times"]):
        # Center window on cough
        cough_center = (start + end) / 2
        win_start = max(0, cough_center - window_sec / 2)
        win_end = min(example["duration_seconds"], win_start + window_sec)

        # Extract audio window
        audio_start = int(win_start * fs_audio)
        audio_end = int(win_end * fs_audio)
        audio_window = example["outward_facing_mic"]["array"][audio_start:audio_end]

        # Extract IMU window
        imu_start = int(win_start * fs_imu)
        imu_end = int(win_end * fs_imu)
        imu_window = example["imu"][imu_start:imu_end]

        windows.append({"audio": audio_window, "imu": imu_window})

    return windows

# Apply to dataset
cough_windows = []
for example in train_data.filter(lambda x: x["has_ground_truth"]):
    windows = extract_cough_window(example)
    if windows:
        cough_windows.extend(windows)

print(f"Extracted {len(cough_windows)} cough windows for training")
```

## Recommended Evaluation Metrics

For cough detection, use **event-based metrics** (not frame-level):

```python
# Install: pip install timescoring
from timescoring import TimeScoring

# Recommended parameters for this dataset
scorer = TimeScoring(
    toleranceStart=0.25,      # 250ms tolerance at cough start
    toleranceEnd=0.25,        # 250ms tolerance at cough end
    minOverlap=0.1,           # 10% minimum overlap
    maxEventDuration=0.6,     # Maximum cough duration: 600ms
    minDurationBetweenEvents=0  # No minimum gap
)

# Compute metrics
tp, fp, fn = scorer.count(predictions, ground_truth)
sensitivity = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
```

See `notebooks/Compute_Success_Metrics.ipynb` in the original repository for detailed examples.

## Advanced Usage

### Working with Google Colab

```python
# Install dependencies in Colab
!pip install datasets soundfile librosa

from datasets import load_dataset
import IPython.display as ipd

# Load dataset
dataset = load_dataset("username/edge-ai-cough-count", split="train")

# Listen to a cough recording
example = dataset.filter(lambda x: x["sound"] == "cough")[0]
ipd.Audio(example["outward_facing_mic"]["array"], rate=16000)
```

### Multimodal Model Training

```python
import torch
from torch.utils.data import Dataset

class MultimodalCoughDataset(Dataset):
    """Custom PyTorch dataset for multimodal cough detection."""

    def __init__(self, hf_dataset, window_len=0.7):
        self.data = hf_dataset
        self.window_len = window_len
        self.fs_audio = 16000
        self.fs_imu = 100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Get audio (use outward-facing mic)
        audio = torch.tensor(example["outward_facing_mic"]["array"], dtype=torch.float32)

        # Get IMU
        imu = torch.tensor(example["imu"], dtype=torch.float32)

        # Label: 1 if cough, 0 otherwise
        label = 1 if example["sound"] == "cough" else 0

        return {"audio": audio, "imu": imu, "label": label}

# Create PyTorch datasets
from datasets import load_dataset
hf_data = load_dataset("username/edge-ai-cough-count")

train_dataset = MultimodalCoughDataset(hf_data["train"])
val_dataset = MultimodalCoughDataset(hf_data["val"])
test_dataset = MultimodalCoughDataset(hf_data["test"])

# Use with DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## Version History

- **v1.0.0** (2026-01): Initial release with 15 subjects, stratified splits, multimodal biosignals

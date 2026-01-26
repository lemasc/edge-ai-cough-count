# Cough Detection Model Training Documentation (Edge-AI, Multimodal)

This document describes an ML-developer-focused training pipeline for a **cough vs. non-cough** classifier that is designed to be **edge-friendly** (short inference windows, classical features + lightweight ML). It follows the dataset organization and training/validation procedure described by the paper, and assumes **all dataset files live under `dataset/`**.

---

## 1) Project overview

### 1.1 Goal

Build a model that detects whether a short multimodal time window contains a cough, using:

* **Audio** (two microphones)
* **Kinematics** (IMU: accelerometer + gyroscope)

The paper’s motivation is accurate cough counting while preserving privacy and enabling on-device inference. 

### 1.2 What you will reproduce

The paper demonstrates a complete “classical ML” pipeline:

* Segment signals into **0.4-second windows**
* Extract **handcrafted features** from audio and IMU
* Train and select a classifier with **subject-wise cross-validation**
* Report **ROC-AUC** and pick an operating point to compute sensitivity/specificity/precision

Key reported performance on an unseen (private) test set:

* **AUC ≈ 0.97**, max avg **F1 ≈ 0.914**
* **Sensitivity 91%**, **Specificity 92%**, **Precision 80%** 
  Cross-validation AUC (public subjects) for best multimodal model:
* **0.96 ± 0.01** (all features) 

---

## 2) Dataset layout and semantics

### 2.1 Directory structure

Each subject contains multiple trials and conditions; each condition includes:

* `body-facing-mic.wav`
* `outward-facing-mic.wav`
* `imu.csv`
* and for cough recordings only: `ground-truth.json` 

Example structure (simplified) :

```
dataset/
  Subject ID/
    Trial 1/
      Sit/
        No noise/
          Cough/  (has ground-truth.json)
          Laugh/
          Deep breathing/
          Throat clearing/
        Traffic/
        Music/
        Bystander cough/
        Walk/
    biodata.json
```

### 2.2 Annotation file format

For cough recordings, `ground-truth.json` contains **start** and **end** times for each cough, stored as keys:

* `start times`
* `end times` 

### 2.3 Signal encoding details to handle correctly

* Audio is stored as **32-bit signed integers** representing **24-bit PCM shifted left by 8 bits**. You should convert to float (e.g., `float32`) and normalize carefully (e.g., divide by `2**31` or use peak normalization per file) before feature extraction. 
* IMU data is in `imu.csv` (paper device sampled IMU at 100 Hz; your loader should not assume constant column order—inspect headers).

---

## 3) Environment and recommended libraries (Python)

You can implement this pipeline with:

* Core: `numpy`, `pandas`, `scipy`
* Audio: `soundfile` or `scipy.io.wavfile`, plus `librosa` for MFCC/spectral features
* ML: `scikit-learn`
* Imbalance: `imbalanced-learn` (SMOTE)
* Model: `xgboost` (XGBoost classifier)
* Explainability (optional): `shap`

---

## 4) Data preparation: segmentation into training examples

### 4.1 Windowing strategy (edge-friendly)

The paper trains on **0.4 s** windows to reduce memory footprint for on-device inference. 

Let:

* Audio sample rate = 16 kHz → 0.4 s = 6400 samples
* IMU sample rate ≈ 100 Hz → 0.4 s = 40 samples

### 4.2 Positive (cough) segments

For each cough event in `ground-truth.json`:

1. Convert cough start/end times to sample indices for audio and IMU.
2. Extract a **0.4 s window around the cough**.
3. Apply augmentation by **randomly shifting the window** around the cough; the paper does this **twice**. 

   * If cough duration < window: shift window so cough occurs at varying positions within the window.
   * If cough duration > window: sample windows randomly within the cough span.

### 4.3 Negative (non-cough) segments

From non-cough recordings (laugh, throat clearing, deep breathing, etc.):

* Randomly sample fixed-length windows (0.4 s) from each recording
* Keep a balanced mixture of negative sound types (the paper mentions generating equal number of segments for each produced sound in their helper functions). 

### 4.4 Alignment between modalities

You must ensure each training sample contains synchronized:

* Audio window(s) (choose which mic(s) you use)
* IMU window

Practical approach:

* Use timestamps or assume recordings are aligned from time 0 and rely on sample rates.
* If you observe drift, resample IMU to a fixed grid per window (e.g., linear interpolation) so feature extraction is stable.

### 4.5 Minimal preprocessing (as in paper)

* **No filtering** was applied.
* **IMU mean is subtracted per segment** to center it at zero. 

---

## 5) Feature extraction (matches the paper)

The paper uses classical features to stay edge-friendly.

### 5.1 Audio features (per microphone)

Per mic signal, compute **65 audio features** used in prior cough detection work. 
Table III lists categories including: 

* **Frequency-domain / spectral**: MFCC, PSD, dominant frequency, spectral centroid/rolloff/spread/skewness/kurtosis, spectral decrease/slope/flatness/std
* **Time-domain**: energy-envelope peak detection features, zero-crossing rate, crest factor, RMS power

Implementation notes:

* MFCC: typical choice `n_mfcc=13` (or more) + summary stats (mean/std/min/max) to reach a stable fixed-length vector.
* PSD and spectral descriptors: use `scipy.signal.welch` and compute descriptors on the magnitude spectrum.

### 5.2 IMU features

The paper uses **5 time-domain features** per IMU channel: 

* Line length
* Zero-crossing rate
* Kurtosis
* Crest factor
* RMS power

They compute these on:

* Each accelerometer axis (x, y, z)
* Accelerometer **L2 norm**
* Each gyroscope axis (yaw, pitch, roll)
* Gyroscope **L2 norm**

Result: **40 total IMU features** (8 signals × 5 features). 

---

## 6) Train/validation design (avoid leakage)

### 6.1 Subject-wise splitting is mandatory

Your dataset contains repeated trials and multiple conditions per subject. If you randomly split windows, you will leak subject identity and inflate metrics.

Use **Leave-n-Subjects-Out** cross-validation:

* **5 folds** in the paper. 
  With 15 public subjects, a practical mapping is ~3 subjects per fold.

### 6.2 Class imbalance handling

Cough vs non-cough segments will be imbalanced. The paper uses **SMOTE** within each CV fold. 
Important: apply SMOTE **only on the training split** of each fold.

### 6.3 Feature scaling

Within each CV fold:

* Fit scaler on training features only (e.g., `StandardScaler`)
* Transform train and validation with that scaler 

---

## 7) Model selection pipeline (as in paper)

### 7.1 Candidate models

The paper evaluates six classifiers: 

* Logistic Regression
* Gaussian Naive Bayes
* Linear Discriminant Analysis
* Decision Tree
* Random Forest
* **XGBoost (best)**

### 7.2 Selection criterion

Primary selection metric: **mean ROC-AUC across folds**. 

### 7.3 Feature selection

After choosing the best-performing model family, apply:

* **RFECV** (recursive feature elimination with CV) to remove non-contributing features 

### 7.4 Hyperparameter optimization

Finally, tune hyperparameters (e.g., `RandomizedSearchCV` or Bayesian optimization) and report final CV AUC. 

### 7.5 Modalities to evaluate

The paper compares:

1. IMU-only
2. Outer mic-only
3. All features (audio + IMU)

Reported CV AUC: 

* IMU only: **0.90 ± 0.02**
* Outer mic only: **0.92 ± 0.01**
* All features: **0.96 ± 0.01**

---

## 8) Practical training steps (suggested project structure)

### 8.1 Suggested repository layout

```
project/
  dataset/                       # provided
  src/
    data/
      scan_dataset.py            # enumerate subjects/trials/conditions
      load_audio.py              # wav -> float
      load_imu.py                # csv -> dataframe/array
      segment.py                 # build 0.4s windows + augmentation
    features/
      audio_features.py
      imu_features.py
      build_feature_table.py     # X matrix + y labels + subject_id
    modeling/
      cv_split.py                # leave-n-subjects-out folds
      train_eval.py              # scaler + SMOTE + model training
      rfecv.py
      tune.py
    reports/
      metrics.py                 # ROC, PR, confusion matrix
      plots.py
  configs/
    train.yaml                   # window_len, augmentation, feature params, model params
```

### 8.2 Pseudocode outline (developer to implement)

```python
# 1) Index dataset
records = scan_dataset("dataset/")  # yields paths + metadata including subject_id and label types

# 2) Build segments
segments = []
for rec in records:
    audio_body, sr = load_wav(rec.body_mic_path)
    audio_outer, sr = load_wav(rec.outer_mic_path)
    imu = load_imu(rec.imu_csv_path)

    if rec.is_cough:
        ann = load_json(rec.ground_truth_path)
        segs = build_positive_segments(audio_outer, audio_body, imu, ann,
                                       window_s=0.4, shifts=2)
    else:
        segs = build_negative_segments(audio_outer, audio_body, imu,
                                       window_s=0.4, n_segments=K)
    segments.extend(segs)

# 3) Extract features
X, y, groups = [], [], []
for seg in segments:
    f_audio = audio_features(seg.audio_outer, seg.audio_body)
    f_imu = imu_features(seg.imu_centered)  # subtract mean per segment
    X.append(concat([f_audio, f_imu]))
    y.append(seg.label)
    groups.append(seg.subject_id)

# 4) Subject-wise CV with scaling + SMOTE in-fold
for train_idx, val_idx in leave_n_subjects_out(groups, n_folds=5):
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]

    scaler.fit(Xtr); Xtr = scaler.transform(Xtr); Xva = scaler.transform(Xva)
    Xtr, ytr = SMOTE().fit_resample(Xtr, ytr)

    model.fit(Xtr, ytr)
    yhat = model.predict_proba(Xva)[:, 1]
    aucs.append(roc_auc_score(yva, yhat))

# 5) Select model, RFECV, hyperparameter tune, final metrics
```

---

## 9) Model validation and reporting

### 9.1 Primary metric: ROC-AUC

Report:

* Mean ± std ROC-AUC over folds
* ROC curve aggregated or per fold

### 9.2 Operating point selection

To match the paper’s reporting, choose the threshold that maximizes F1 on the ROC (or on a validation set), then report:

* Sensitivity (recall for cough)
* Specificity
* Precision
* F1 

### 9.3 Suggested additional checks (practical, not required by paper)

* Confusion matrix per fold
* PR-AUC (useful under imbalance)
* Error slices: “bystander cough” condition vs others (to ensure robustness)

---

## 10) Expected results (targets to sanity-check your reproduction)

If you follow the paper closely, your results on the public dataset via subject-wise CV should be in the neighborhood of: 

* **All features (audio + IMU): ~0.96 ± 0.01 ROC-AUC**
* **Outer mic only: ~0.92 ± 0.01**
* **IMU only: ~0.90 ± 0.02**

For the paper’s final selected multimodal model evaluated on withheld test subjects, they report: 

* **AUC 0.97**
* **Sensitivity 91%**, **Specificity 92%**, **Precision 80%**
* Max avg **F1 0.914**

If you fall substantially short, the usual causes are:

* subject leakage (random split instead of subject-wise)
* incorrect windowing (not 0.4 s, or misaligned modalities)
* feature mismatch (e.g., MFCC/stat aggregation differences)
* SMOTE/scaling applied incorrectly (must be in-fold, train-only)

---

## 11) Notes on explainability (optional, mirrors paper)

The paper uses SHAP to identify influential features; top contributors include IMU line-length features and audio MFCC/spectral stats. 
This is optional, but useful for debugging and for edge feature budgeting (keeping only the most valuable features).
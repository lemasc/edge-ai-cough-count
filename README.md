# Edge Artificial Intelligence (edge-AI) Cough Counting

## Background

Counting the number of times a patient coughs per day is an essential biomarker in determining treatment efficacy for novel antitussive therapies and personalizing patient care. There is a need for wearable devices that employ multimodal sensors to perform accurate, privacy-preserving, automatic cough counting algorithms directly on the device in an Edge-AI fashion. To advance this research field, our team from the Embedded Systems Laboratory (ESL) of EPFL contributed the first publicly accessible cough counting dataset of multimodal biosignals. The database contains nearly 4 hours of biosignal data, with both acoustic and kinematic modalities, covering 4,300 annotated cough events. Furthermore, a variety of non-cough sounds and motion scenarios mimicking daily life activities are also present, which the research community can use to accelerate ML algorithm development.

This repository contains useful functions that researchers can use to use our public dataset, including:

- Code for generating biosignal segments from the public dataset to streamline ML algorithm development
- Testing your predicted cough locations against the ground truth in an event-based manner
- A FastAPI inference server exposing trained models via a REST API
- A browser-based recording tool for collecting new labelled samples using the trained model as a real-time annotator

## Data access

The edge-AI cough counting dataset can be found at the following Zenodo link: https://zenodo.org/record/7562332#.Y87MenbMKUm

Additionally, the dataset has been downloaded and extracted locally to ./public_dataset, apart from the above link distributed by the repository author.

## Getting started

### Python dependencies

Install Python dependencies (Python 3.10 required). Using uv is recommended.

```
uv sync
```

## Usage

### Dataset generation

Several Jupyter notebooks are available to illustrate the functionality of the code.

In `Segmentation_Augmentation.ipynb`, we demonstrate how to turn the raw biosignals and annotations provided in the dataset into segmented biosignals ready to input into ML models.

In `Cough_Annotation.ipynb`, we explain how the fine-grained cough labeling was performed in a semi-automatic manner, in case other teams wish to merge their datasets and keep the labeling scheme consistent.

### Testing your predictions

Once you use our datset to develop a ML model for counting coughs, the next step is to validate your predicted cough locations against the ground-truth cough locations.

The `Compute_Success_Metrics.ipynb` notebook demonstrates how to use [the timescoring event-based success evaluation library](https://pypi.org/project/timescoring/) to extract clinically meaningful success metrics from your algorithms.

### Functions

The `helpers.py` file contains useful functions for quickly iterating through the database structure, performing some signal processing, and loading the biosignals and annotations.

The `dataset_gen.py` file segments the raw biosignals and contains useful functions for creating a cough detection database for training edge-AI Machine Learning Models.

## Inference server

A FastAPI server exposes the trained audio-only XGBoost model via a REST API, enabling real-time cough detection from audio recordings.

**Prerequisites:** train and save the model by running `notebooks/Model_Training_XGBoost.ipynb` first. The server loads `notebooks/models/xgb_audio.pkl` at startup.

```bash
uvicorn server.main:app --reload
```

Endpoints:

- `GET /api/health` — health check
- `POST /api/predict` — accepts a multipart audio file upload, returns detected cough events:

```json
{
  "cough_count": 3,
  "start_times": [1.25, 4.80, 7.10],
  "end_times": [1.75, 5.20, 7.55],
  "window_times": [...],
  "probabilities": [...]
}
```

Audio is automatically resampled to 16 kHz and peak-normalized to match training preprocessing. Any format supported by librosa (WAV, MP3, OGG, WebM, MP4) is accepted.

## Dataset collector

A browser-based PWA in `dataset-collector/` for recording new labelled audio samples. It records audio via the browser microphone, sends it to the inference server for automatic cough annotation, displays the detected cough events for review, then packages the recording and its annotations into a ZIP file ready for inclusion in future dataset releases.

**Prerequisites:** the inference server must be running.

```bash
cd dataset-collector
pnpm install
pnpm dev      # opens on http://localhost:5173
```

Each downloaded ZIP contains:
- `outward_facing_mic.{webm,mp4,ogg}` — the raw audio recording
- `ground_truth.json` — predicted cough start/end times (used as pseudo-ground-truth)
- `metadata.json` — subject ID, sound type, movement condition, background noise, trial number, and device info

The Vite dev server proxies `/api/*` requests to the backend automatically.

# Citations

If you use the open-source dataset in your work, please cite our publication:

```
@inproceedings{orlandic_multimodal_2023,
	address = {Sydney, Australia},
	title = {A {Multimodal} {Dataset} for {Automatic} {Edge}-{AI} {Cough} {Detection}},
	copyright = {https://doi.org/10.15223/policy-029},
	url = {https://ieeexplore.ieee.org/document/10340413/},
	doi = {10.1109/EMBC40787.2023.10340413},
	language = {en},
	urldate = {2024-04-10},
	booktitle = {2023 45th {Annual} {International} {Conference} of the {IEEE} {Engineering} in {Medicine} \& {Biology} {Society} ({EMBC})},
	publisher = {IEEE},
	author = {Orlandic, Lara and Thevenot, Jérôme and Teijeiro, Tomas and Atienza, David},
	month = jul,
	year = {2023},
	pages = {1--7},
}
```

# Contact

For questions or suggestions, please contact lara.orlandic@epfl.ch.

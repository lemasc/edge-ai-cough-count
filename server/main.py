import io
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import extract_audio_features

FS = 16000
WINDOW_SAMPLES = 6400   # 0.4 s × 16000
STRIDE_SAMPLES = 1600   # 0.1 s × 16000
GAP_FILL_SEC = 0.2      # merge events separated by ≤ this

model_data: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(__file__).parent.parent / "notebooks" / "models" / "xgb_audio.pkl"
    with open(model_path, "rb") as f:
        model_data.update(pickle.load(f))
    yield
    model_data.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(audio: UploadFile):
    audio_bytes = await audio.read()

    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=FS, mono=True)

    # Build sliding windows
    n_samples = len(y)
    windows = []
    window_times = []

    start = 0
    while start + WINDOW_SAMPLES <= n_samples:
        windows.append(y[start : start + WINDOW_SAMPLES])
        window_times.append(start / FS)
        start += STRIDE_SAMPLES

    if not windows:
        return {
            "cough_count": 0,
            "start_times": [],
            "end_times": [],
            "window_times": [],
            "probabilities": [],
        }

    # Extract features
    features = np.array([extract_audio_features(w, fs=FS) for w in windows])

    scaler = model_data["scaler"]
    model = model_data["model"]
    threshold = model_data.get("threshold", 0.5)

    X_scaled = scaler.transform(features)
    proba = model.predict_proba(X_scaled)[:, 1]
    predictions = (proba >= threshold).astype(int)

    # Merge adjacent positive windows into events
    window_duration = WINDOW_SAMPLES / FS   # 0.4 s
    stride_sec = STRIDE_SAMPLES / FS        # 0.1 s
    gap_fill_windows = int(GAP_FILL_SEC / stride_sec)

    start_times: list[float] = []
    end_times: list[float] = []

    i = 0
    n = len(predictions)
    while i < n:
        if predictions[i] == 1:
            event_start = window_times[i]
            event_end = window_times[i] + window_duration
            j = i + 1
            while j < n:
                if predictions[j] == 1:
                    event_end = window_times[j] + window_duration
                    j += 1
                elif j + gap_fill_windows < n and any(
                    predictions[j : j + gap_fill_windows + 1]
                ):
                    # Gap smaller than threshold — fill and continue
                    j += 1
                else:
                    break
            start_times.append(round(event_start, 3))
            end_times.append(round(event_end, 3))
            i = j
        else:
            i += 1

    return {
        "cough_count": len(start_times),
        "start_times": start_times,
        "end_times": end_times,
        "window_times": [round(t, 3) for t in window_times],
        "probabilities": [round(float(p), 4) for p in proba],
    }

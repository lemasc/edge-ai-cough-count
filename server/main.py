import io
import logging
import pickle
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import librosa
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("server")

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src")) 
from helpers import FS_AUDIO
from predict import sliding_window_predict, merge_detections, create_dummy_imu

model_data: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(__file__).parent.parent / "notebooks" / "models" / "xgb_audio.pkl"
    log.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        model_data.update(pickle.load(f))
    log.info("Model loaded — keys: %s, threshold: %s", list(model_data.keys()), model_data.get("threshold"))
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
    log.info("Received file: name=%s content_type=%s", audio.filename, audio.content_type)
    try:
        audio_bytes = await audio.read()
        log.debug("Read %d bytes", len(audio_bytes))

        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=FS_AUDIO, mono=True)
        log.debug("Decoded audio: %d samples (%.2f s)", len(y), len(y) / FS_AUDIO)

        duration = len(y) / FS_AUDIO
        dummy_imu = create_dummy_imu(duration)

        threshold = model_data.get("threshold", 0.5)
        raw_preds, all_probs, window_times, _ = sliding_window_predict(
            y, dummy_imu, model_data, modality='audio',
            window_len=0.4, hop_size=0.05, threshold=threshold
        )
        log.debug("Built %d windows, %d above threshold", len(all_probs), len(raw_preds))

        if len(all_probs) == 0:
            log.warning("Audio too short for even one window (%d samples)", len(y))
            return {
                "cough_count": 0,
                "start_times": [],
                "end_times": [],
                "window_times": [],
                "probabilities": [],
            }

        predictions = merge_detections(raw_preds, gap_threshold=0.3)

        start_times = [round(s, 3) for s, e, p in predictions]
        end_times = [round(e, 3) for s, e, p in predictions]

        log.info("Detected %d cough event(s): %s", len(start_times), list(zip(start_times, end_times)))

        return {
            "cough_count": len(start_times),
            "start_times": start_times,
            "end_times": end_times,
            "window_times": [round(float(t), 3) for t in window_times],
            "probabilities": [round(float(p), 4) for p in all_probs],
        }

    except HTTPException:
        raise
    except Exception:
        log.error("Unhandled exception in /api/predict:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())

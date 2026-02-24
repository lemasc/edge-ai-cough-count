import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()

logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("server")

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
from helpers import FS_AUDIO
from predict import (
    sliding_window_predict,
    merge_detections,
    refine_cough_events,
    create_dummy_audio,
    create_dummy_imu,
)

model_data: dict[str, Any] = {}


def load_audio_bytes(data: bytes, sr: int) -> np.ndarray:
    """Decode audio bytes via ffmpeg using a temp file for seekable inputs."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            tmp_path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()}")
        return np.frombuffer(result.stdout, dtype=np.float32).copy()
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                log.warning("Failed to delete temp file: %s", tmp_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_model_dir = Path(__file__).parent.parent / "notebooks" / "models"
    model_dir = Path(os.environ.get("MODEL_DIR", str(default_model_dir)))
    log.info("Using MODEL_DIR=%s", model_dir)
    model_path = model_dir / "xgb_audio.pkl"
    log.info("Loading model from %s", model_path)
    with open(model_path, "rb") as f:
        model_data.update(pickle.load(f))
    log.info(
        "Model loaded — keys: %s, threshold: %s",
        list(model_data.keys()),
        model_data.get("threshold"),
    )
    warmup_start = time.perf_counter()
    warmup_duration = 1.0
    warmup_audio = create_dummy_audio(warmup_duration)
    warmup_imu = create_dummy_imu(warmup_duration)
    sliding_window_predict(
        warmup_audio,
        warmup_imu,
        model_data,
        modality="audio",
        window_len=0.4,
        hop_size=0.05,
        threshold=model_data.get("threshold", 0.5),
    )
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    log.info("Warmup inference complete (%.1f ms)", warmup_ms)
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
    log.info(
        "Received file: name=%s content_type=%s", audio.filename, audio.content_type
    )
    try:
        request_start = time.perf_counter()
        audio_bytes = await audio.read()
        read_ms = (time.perf_counter() - request_start) * 1000
        log.debug("Read %d bytes", len(audio_bytes))

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload")

        decode_start = time.perf_counter()
        y = load_audio_bytes(audio_bytes, sr=FS_AUDIO)
        decode_ms = (time.perf_counter() - decode_start) * 1000
        if y.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Decoded audio is empty; check ffmpeg input format",
            )
        # Peak normalization matching training preprocessing (load_audio normalize_1=True)
        y = y - y.mean()
        y = y / (np.abs(y).max() + 1e-17)
        log.debug("Decoded audio: %d samples (%.2f s)", len(y), len(y) / FS_AUDIO)

        duration = len(y) / FS_AUDIO
        dummy_imu = create_dummy_imu(duration)

        threshold = model_data.get("threshold", 0.5)
        predict_start = time.perf_counter()
        raw_preds, all_probs, window_times, _ = sliding_window_predict(
            y,
            dummy_imu,
            model_data,
            modality="audio",
            window_len=0.4,
            hop_size=0.05,
            threshold=threshold,
        )
        predict_ms = (time.perf_counter() - predict_start) * 1000
        log.debug(
            "Built %d windows, %d above threshold", len(all_probs), len(raw_preds)
        )

        if len(all_probs) == 0:
            log.warning("Audio too short for even one window (%d samples)", len(y))
            return {
                "cough_count": 0,
                "start_times": [],
                "end_times": [],
                "window_times": [],
                "probabilities": [],
            }

        post_start = time.perf_counter()
        candidate_segments = merge_detections(raw_preds, gap_threshold=0.5)
        predictions = refine_cough_events(y, candidate_segments)
        post_ms = (time.perf_counter() - post_start) * 1000

        start_times = [round(s, 3) for s, e, p in predictions]
        end_times = [round(e, 3) for s, e, p in predictions]

        log.info(
            "Detected %d cough event(s): %s",
            len(start_times),
            list(zip(start_times, end_times)),
        )
        total_ms = (time.perf_counter() - request_start) * 1000
        log.info(
            "Timing (ms): read=%.1f decode=%.1f predict=%.1f post=%.1f total=%.1f",
            read_ms,
            decode_ms,
            predict_ms,
            post_ms,
            total_ms,
        )

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

React PWA for recording audio samples, sending them to the FastAPI inference server for cough detection, and packaging results as labeled ZIP files for dataset collection.

## Commands

```bash
pnpm dev        # dev server on http://localhost:5173 (proxies /api/* to localhost:8000)
pnpm build      # tsc -b && vite build → dist/
pnpm lint       # ESLint check
pnpm preview    # preview production build
```

The FastAPI backend must be running at `localhost:8000` for API calls to work. Start it from the repo root with:
```bash
uvicorn server.main:app --reload
```

There is no test suite.

## Tech Stack

- React 19, TypeScript, Vite 7, Tailwind CSS v4 (via `@tailwindcss/vite` plugin)
- `fflate` for in-browser ZIP creation
- Package manager: **pnpm** (not npm or yarn)

## Architecture

### State Machine (`App.tsx`)

The app is a single-page state machine using a discriminated union `AppState` (defined in `types.ts`). State flows strictly forward:

```
idle → requesting-permissions → ready → recording → stopped → predicting → results → labeling
```

Each phase maps to a screen component. `App.tsx` owns all state transitions and passes only the relevant handlers down as props — screens are stateless display components (except `LabelingScreen`, which holds its own form state).

### Key Files

- **`types.ts`** — All shared types: `AppState` union, `PredictionResult`, `RecordingResult`, `RecordingLabel`, and the option arrays for label dropdowns
- **`hooks/useAudioRecorder.ts`** — Wraps `MediaRecorder` API. Requests mic with `echoCancellation: false`, `noiseSuppression: false`, `autoGainControl: false`, and 16 kHz ideal sample rate to match training data preprocessing
- **`utils/buildZip.ts`** — Builds ZIP containing `outward_facing_mic.{webm,mp4,ogg}`, `ground_truth.json` (predicted `{start_times, end_times}`), and `metadata.json` (subject ID, label fields, device info). Filename format: `{subjectId}_trial{trial}_{movement}_noise-{noise}_{sound}.zip`
- **`components/`** — One component per state phase: `PermissionScreen`, `RecordingScreen`, `PredictionScreen`, `LabelingScreen`

### API Integration

`POST /api/predict` — sends `multipart/form-data` with audio blob. Response is `PredictionResult`:
```ts
{ cough_count, start_times, end_times, window_times, probabilities }
```

### Vite Config Notes

- `/api/*` is proxied to `http://localhost:8000`
- `allowedHosts` includes `*.ngrok-free.app` for exposing the dev server to mobile devices via ngrok

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

React app for recording audio samples, sending them to the FastAPI inference server for cough detection, and packaging results as labeled ZIP files for dataset collection. Deployed as a Cloudflare Worker.

## Commands

```bash
pnpm dev        # dev server on http://localhost:5173 (proxies /api/* to localhost:8000)
pnpm build      # react-router build → build/
pnpm typecheck  # generate CF types + react-router types + tsc
pnpm deploy     # build + wrangler deploy to Cloudflare Workers
pnpm preview    # build + vite preview
```

The FastAPI backend must be running at `localhost:8000` for API calls to work. Start it from the repo root with:
```bash
uvicorn server.main:app --reload
```

There is no test suite.

## Tech Stack

- React 19, TypeScript, React Router v7 (framework mode, SSR enabled), Vite 7, Tailwind CSS v4
- `fflate` for in-browser ZIP creation, `wavesurfer.js` for waveform visualization
- Deployed on Cloudflare Workers via `workers/app.ts` and `wrangler.jsonc`
- Package manager: **pnpm** (not npm or yarn)

## Architecture

### State Machine (`app/routes/_index.tsx`)

The app is a single-page state machine using a discriminated union `AppState` (defined in `app/types.ts`). State flows strictly forward:

```
idle → requesting-permissions → ready → recording → stopped → predicting → results → labeling
```

`_index.tsx` owns all state transitions and passes only the relevant handlers down as props — screens are stateless display components (except `LabelingScreen`, which holds its own form state).

### Key Files

- **`app/types.ts`** — All shared types: `AppState` union, `PredictionResult`, `RecordingResult`, `RecordingLabel`, `PermissionStatus`, and the option arrays for label dropdowns
- **`app/hooks/useAudioRecorder.ts`** — Wraps `MediaRecorder` API. Requests mic with `echoCancellation: false`, `noiseSuppression: false`, `autoGainControl: false`, and 16 kHz ideal sample rate to match training data preprocessing
- **`app/utils/buildZip.ts`** — Builds ZIP containing `outward_facing_mic.{webm,mp4,ogg}`, `ground_truth.json` (predicted `{start_times, end_times}`), and `metadata.json` (subject ID, label fields, device info). Filename format: `{subjectId}_trial{trial}_{movement}_noise-{noise}_{sound}.zip`
- **`app/components/`** — One component per state phase: `PermissionScreen`, `RecordingScreen`, `PredictionScreen`, `LabelingScreen`. `WaveformPlayer.tsx` wraps wavesurfer.js for audio playback on the results/labeling screens.
- **`workers/app.ts`** — Cloudflare Workers entry point that wraps the React Router request handler

### Routing

Routes are file-system based (`@react-router/fs-routes` with `flatRoutes()`). Currently only one route: `app/routes/_index.tsx`. Route types are auto-generated into `.react-router/types/`.

### API Integration

`POST /api/predict` — sends `multipart/form-data` with audio blob. Response is `PredictionResult`:
```ts
{ cough_count, start_times, end_times, window_times, probabilities }
```

### Vite Config Notes

- `/api/*` is proxied to `http://localhost:8000`
- `allowedHosts` includes `*.ngrok-free.app` for exposing the dev server to mobile devices via ngrok
- Uses `@cloudflare/vite-plugin` for Workers compatibility in the build

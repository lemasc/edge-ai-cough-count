# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `web` subdirectory of the edge-ai-cough-count project. It is a full-stack web application ("CoughSense") deployed on Cloudflare Workers that records audio, sends it to a FastAPI inference server, and displays cough detection results with waveform visualization.

## Commands

```bash
pnpm dev           # dev server (Wrangler + React Router, localhost:5173)
pnpm build         # production build
pnpm deploy        # build + wrangler deploy to Cloudflare
pnpm typecheck     # cf-typegen + react-router typegen + tsc -b
pnpm db:generate   # generate Drizzle migrations from schema changes
pnpm db:migrate    # apply D1 migrations locally via Wrangler
```

## Architecture

**Stack:** React Router v7 (framework/SSR mode) + Cloudflare Workers + D1 (SQLite) + R2 (object storage)

**Entry points:**
- `app/root.tsx` — HTML shell

**Route flow:**
```
/           → _index.tsx        Landing page
/record     → record.tsx        Permission + audio recording; action uploads to R2, inserts DB row, redirects to /predict/:id
/predict/:id → predict.$id.tsx  Auto-submits action that POSTs audio to FastAPI, stores results in DB, redirects to /results/:id
/results/:id → results.$id.tsx  Displays cough count + WaveformPlayer
/results/:id/audio → results.$id.audio.tsx  Streams audio from R2 (cached 1 year)
/complete   → complete.tsx      Confirmation screen
```

**Cloudflare bindings:**
- `env.DB` — D1 database, accessed via `getDb(env.DB)` from `app/db/index.ts`
- `env.STORAGE` — R2 bucket for audio blobs
- `env.PREDICT_API_URL` — FastAPI server base URL; set in `.dev.vars` locally, `wrangler.jsonc` for production

**Database schema** (`app/db/schema.ts`):
- `recordings` table: `id`, `createdAt`, `durationMs`, `audioKey` (R2 path), `status` (pending/done/error), `coughCount`, `startTimes`/`endTimes`/`windowTimes`/`probabilities` (stored as JSON strings)

**Key components:**
- `RecordingScreen` — countdown (3s) + recording phases, max 10s, 16kHz mono via `useAudioRecorder`
- `WaveformPlayer` — wavesurfer.js v7 waveform with cough region overlays; takes `src: string` (URL), not a Blob
- `PredictionScreen` — discriminated union props: `phase: "predicting"` | `phase: "results"`

**Type generation:** Route types come from `react-router typegen` and are imported as `import type { Route } from './+types/<routename>'`. Always run `pnpm typecheck` after adding routes or changing loader/action signatures.

**Local dev vars** — create `.dev.vars` at project root:
```
PREDICT_API_URL=http://localhost:8000
```
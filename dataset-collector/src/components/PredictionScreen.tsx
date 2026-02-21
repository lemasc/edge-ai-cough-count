import type { PredictionResult } from '../types.ts';

type Props =
  | {
      phase: 'predicting';
      durationMs: number;
      prediction?: undefined;
      onReset: () => void;
      onLabel: () => void;
    }
  | {
      phase: 'results';
      durationMs: number;
      prediction: PredictionResult;
      onReset: () => void;
      onLabel: () => void;
    };

function formatTime(seconds: number): string {
  return seconds.toFixed(1) + ' s';
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function ProbabilityTimeline({
  windowTimes,
  probabilities,
  durationMs,
}: {
  windowTimes: number[];
  probabilities: number[];
  durationMs: number;
}) {
  const durationSec = durationMs / 1000;
  const threshold = 0.5;

  if (windowTimes.length === 0) return null;

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
        Probability Timeline
      </p>
      <div className="relative h-10 w-full overflow-hidden rounded-lg bg-gray-800">
        {windowTimes.map((t, i) => {
          const prob = probabilities[i] ?? 0;
          const leftPct = (t / durationSec) * 100;
          // Window width based on stride (0.1s) as percentage
          const widthPct = Math.max((0.1 / durationSec) * 100, 0.5);

          // Color: green (low prob) → red (high prob)
          const r = Math.round(prob * 220);
          const g = Math.round((1 - prob) * 180);
          const color = `rgb(${r}, ${g}, 30)`;

          return (
            <div
              key={i}
              className="absolute top-0 h-full"
              style={{
                left: `${leftPct}%`,
                width: `${widthPct}%`,
                backgroundColor: color,
                opacity: 0.85,
              }}
            />
          );
        })}
        {/* Threshold line at 50% height */}
        <div
          className="absolute top-1/2 h-px w-full -translate-y-1/2 bg-white/40"
          style={{ top: `${(1 - threshold) * 100}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-600">
        <span>0 s</span>
        <span>{durationSec.toFixed(1)} s</span>
      </div>
    </div>
  );
}

export function PredictionScreen({ phase, durationMs, prediction, onReset, onLabel }: Props) {
  if (phase === 'predicting') {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
        <div className="w-full max-w-sm space-y-6 text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500" />
            <p className="text-lg font-semibold">Analyzing audio…</p>
            <p className="text-sm text-gray-500">Duration: {formatDuration(durationMs)}</p>
          </div>
        </div>
      </div>
    );
  }

  const { cough_count, start_times, end_times, window_times, probabilities } = prediction;

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <div className="mb-2 text-5xl font-bold tabular-nums">{cough_count}</div>
          <p className="text-lg text-gray-300">
            {cough_count === 1 ? 'cough detected' : 'coughs detected'}
          </p>
        </div>

        <ProbabilityTimeline
          windowTimes={window_times}
          probabilities={probabilities}
          durationMs={durationMs}
        />

        {start_times.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
              Detected Events
            </p>
            <div className="space-y-1 rounded-xl border border-gray-700 bg-gray-900 px-4 py-3">
              {start_times.map((start, i) => (
                <div key={i} className="flex justify-between text-sm">
                  <span className="text-gray-500">Cough {i + 1}</span>
                  <span className="font-mono text-gray-200">
                    {formatTime(start)} – {formatTime(end_times[i] ?? start)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {start_times.length === 0 && (
          <div className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-center text-sm text-gray-500">
            No cough events detected
          </div>
        )}

        <div className="space-y-3">
          <button
            onClick={onLabel}
            className="min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
          >
            Label &amp; Save
          </button>
          <button
            onClick={onReset}
            className="min-h-12 w-full rounded-xl border border-gray-700 px-6 py-3 text-base font-semibold text-gray-400 transition hover:border-gray-500 hover:text-white active:scale-95"
          >
            Record Another
          </button>
        </div>
      </div>
    </div>
  );
}

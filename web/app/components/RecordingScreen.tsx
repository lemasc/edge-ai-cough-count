import { useEffect, useState } from "react";

const MAX_DURATION_MS = 10_000;
const COUNTDOWN_SECONDS = 3;

type Props = {
  phase: "countdown" | "recording" | "stopped";
  startTime: number;
  durationMs?: number;
  onCountdownEnd: () => void;
  onStop: () => void;
  onPredict: () => void;
};

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export function RecordingScreen({
  phase,
  startTime,
  durationMs,
  onCountdownEnd,
  onStop,
  onPredict,
}: Props) {
  const [elapsed, setElapsed] = useState(0);
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS);

  useEffect(() => {
    if (phase !== "countdown") return;
    setCountdown(COUNTDOWN_SECONDS);

    let count = COUNTDOWN_SECONDS;
    const id = setInterval(() => {
      count -= 1;
      setCountdown(count);
      if (count <= 0) {
        clearInterval(id);
        onCountdownEnd();
      }
    }, 1000);

    return () => clearInterval(id);
  }, [phase, onCountdownEnd]);

  useEffect(() => {
    if (phase !== "recording") return;
    let stopped = false;

    const id = setInterval(() => {
      const e = performance.now() - startTime;
      setElapsed(e);
      if (e >= MAX_DURATION_MS && !stopped) {
        stopped = true;
        onStop();
      }
    }, 200);

    return () => clearInterval(id);
  }, [phase, startTime, onStop]);

  const displayMs =
    phase === "stopped" && durationMs != null ? durationMs : elapsed;
  const progress = Math.min((elapsed / MAX_DURATION_MS) * 100, 100);

  return (
    <div className="w-full max-w-sm space-y-10 text-center">
      <div className="space-y-4">
        {phase === "countdown" && (
          <>
            <span className="text-sm font-medium text-yellow-400 uppercase tracking-wider">
              Get Ready
            </span>
            <div className="font-mono text-8xl font-bold tabular-nums text-yellow-400">
              {countdown}
            </div>
          </>
        )}

        {phase === "recording" && (
          <>
            <div className="flex items-center justify-center gap-3">
              <span className="inline-block h-4 w-4 animate-pulse rounded-full bg-red-500" />
              <span className="text-sm font-medium text-red-400 uppercase tracking-wider">
                Recording
              </span>
            </div>
            <div className="font-mono text-6xl font-bold tabular-nums">
              {formatTime(displayMs)}
            </div>
            <div className="space-y-1">
              <div className="h-2 w-full overflow-hidden rounded-full bg-gray-700">
                <div
                  className="h-full rounded-full bg-red-500 transition-all duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="text-xs text-gray-500">
                max {formatTime(MAX_DURATION_MS)}
              </div>
            </div>
          </>
        )}

        {phase === "stopped" && (
          <>
            <div className="flex items-center justify-center gap-3">
              <span className="inline-block h-4 w-4 rounded-full bg-gray-500" />
              <span className="text-sm font-medium text-gray-400 uppercase tracking-wider">
                Stopped
              </span>
            </div>
            <div className="font-mono text-6xl font-bold tabular-nums">
              {formatTime(displayMs)}
            </div>
          </>
        )}
      </div>

      {phase === "recording" && (
        <button
          type="button"
          onClick={onStop}
          className="min-h-12 w-full rounded-xl bg-red-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-red-500 active:scale-95"
        >
          Stop
        </button>
      )}

      {phase === "stopped" && (
        <div className="space-y-4">
          <div className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-sm text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-500">Duration</span>
              <span className="font-mono">{formatTime(durationMs ?? 0)}</span>
            </div>
          </div>

          <button
            type="button"
            onClick={onPredict}
            className="min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
          >
            Analyze
          </button>
        </div>
      )}
    </div>
  );
}

import { useState, useEffect } from 'react';

type Props = {
  phase: 'recording' | 'stopped';
  startTime: number;
  sampleCount: number;
  durationMs?: number;
  onStop: () => void;
  onLabel: () => void;
};

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

export function RecordingScreen({
  phase,
  startTime,
  sampleCount,
  durationMs,
  onStop,
  onLabel,
}: Props) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (phase !== 'recording') return;

    const id = setInterval(() => {
      setElapsed(performance.now() - startTime);
    }, 200);

    return () => clearInterval(id);
  }, [phase, startTime]);

  const displayMs = phase === 'stopped' && durationMs != null ? durationMs : elapsed;

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
      <div className="w-full max-w-sm space-y-10 text-center">
        <div className="space-y-4">
          {phase === 'recording' && (
            <div className="flex items-center justify-center gap-3">
              <span className="inline-block h-4 w-4 animate-pulse rounded-full bg-red-500" />
              <span className="text-sm font-medium text-red-400 uppercase tracking-wider">
                Recording
              </span>
            </div>
          )}

          {phase === 'stopped' && (
            <div className="flex items-center justify-center gap-3">
              <span className="inline-block h-4 w-4 rounded-full bg-gray-500" />
              <span className="text-sm font-medium text-gray-400 uppercase tracking-wider">
                Stopped
              </span>
            </div>
          )}

          <div className="font-mono text-6xl font-bold tabular-nums">{formatTime(displayMs)}</div>

          <div className="text-sm text-gray-500">
            IMU samples: <span className="text-gray-300 font-mono">{sampleCount}</span>
          </div>
        </div>

        {phase === 'recording' && (
          <button
            onClick={onStop}
            className="min-h-12 w-full rounded-xl bg-red-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-red-500 active:scale-95"
          >
            Stop
          </button>
        )}

        {phase === 'stopped' && (
          <div className="space-y-4">
            <div className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-sm text-gray-300">
              <div className="flex justify-between">
                <span className="text-gray-500">Duration</span>
                <span className="font-mono">{formatTime(durationMs ?? 0)}</span>
              </div>
              <div className="mt-1 flex justify-between">
                <span className="text-gray-500">IMU samples</span>
                <span className="font-mono">{sampleCount}</span>
              </div>
            </div>

            <button
              onClick={onLabel}
              className="min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
            >
              Label &amp; Download
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

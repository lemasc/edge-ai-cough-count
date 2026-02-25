import { Link } from "react-router";
import type { PredictionResult } from "../types";
import { WaveformPlayer } from "./WaveformPlayer";
import { formatSeconds } from "../utils/formatTime";

type Props =
  | {
      phase: "predicting";
      durationMs: number;
      prediction?: undefined;
      onReset: () => void;
    }
  | {
      phase: "results";
      durationMs: number;
      audioSrc: string;
      prediction: PredictionResult;
      onReset: () => void;
    };

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export function PredictionScreen(props: Props) {
  if (props.phase === "predicting") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
        <div className="w-full max-w-sm space-y-6 text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500" />
            <p className="text-lg font-semibold">Analyzing audio…</p>
            <p className="text-sm text-gray-500">
              Duration: {formatDuration(props.durationMs)}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const { audioSrc, prediction, onReset } = props;
  const { cough_count, start_times, end_times } = prediction;

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <div className="mb-2 text-5xl font-bold tabular-nums">
            {cough_count}
          </div>
          <p className="text-lg text-gray-300">
            {cough_count === 1 ? "cough detected" : "coughs detected"}
          </p>
        </div>

        <WaveformPlayer
          src={audioSrc}
          startTimes={start_times}
          endTimes={end_times}
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
                    {formatSeconds(start)} – {formatSeconds(end_times[i] ?? start)}
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
          <Link
            to="/complete"
            className="flex min-h-12 w-full items-center justify-center rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
          >
            Done
          </Link>
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

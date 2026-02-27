import { useRef } from "react";
import { formatSeconds } from "../utils/formatTime";

type EvaluationTimelineProps = {
  durationSecs: number;
  detectedStartTimes: number[];
  detectedEndTimes: number[];
  missedCoughPoints: number[];
  onChange: (points: number[]) => void;
};

export function EvaluationTimeline({
  durationSecs,
  detectedStartTimes,
  detectedEndTimes,
  missedCoughPoints,
  onChange,
}: EvaluationTimelineProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const pointerDownPos = useRef<{ x: number; y: number } | null>(null);
  const duration = durationSecs > 0 ? durationSecs : 10;

  function getTimeFromPointer(e: React.PointerEvent): number {
    const bar = barRef.current;
    if (!bar) return 0;
    const rect = bar.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    return pct * duration;
  }

  function handleBarPointerDown(e: React.PointerEvent) {
    pointerDownPos.current = { x: e.clientX, y: e.clientY };
  }

  function handleBarPointerUp(e: React.PointerEvent) {
    const down = pointerDownPos.current;
    if (!down) return;
    const dx = e.clientX - down.x;
    const dy = e.clientY - down.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    pointerDownPos.current = null;
    if (dist > 8) return; // scroll gesture, skip
    const t = getTimeFromPointer(e);
    onChange([...missedCoughPoints, t].sort((a, b) => a - b));
  }

  function removePin(index: number) {
    onChange(missedCoughPoints.filter((_, i) => i !== index));
  }

  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
        Mark Missed Coughs
      </p>

      {/* Timeline bar */}
      <div
        ref={barRef}
        className="relative h-12 w-full cursor-crosshair rounded-lg bg-gray-800 select-none overflow-hidden"
        onPointerDown={handleBarPointerDown}
        onPointerUp={handleBarPointerUp}
      >
        {/* Red overlays for detected events */}
        {detectedStartTimes.map((start, i) => {
          const end = detectedEndTimes[i] ?? start;
          const left = (start / duration) * 100;
          const width = ((end - start) / duration) * 100;
          return (
            <div
              key={`det-${i}`}
              className="absolute top-0 h-full bg-red-500/30 pointer-events-none"
              style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
            />
          );
        })}

        {/* Amber pins for missed coughs */}
        {missedCoughPoints.map((t, i) => {
          const left = (t / duration) * 100;
          return (
            <div
              key={`pin-${i}`}
              className="absolute top-0 h-full"
              style={{ left: `${left}%` }}
            >
              <div className="absolute top-0 h-full w-0.5 -translate-x-1/2 bg-amber-400 pointer-events-none" />
              <button
                type="button"
                className="absolute top-1 -translate-x-1/2 h-5 w-5 rounded-full bg-amber-400 text-gray-900 text-xs flex items-center justify-center leading-none hover:bg-amber-300 active:scale-90"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  removePin(i);
                }}
                aria-label={`Remove missed cough at ${formatSeconds(t)}`}
              >
                ✕
              </button>
            </div>
          );
        })}

        {/* Empty state hint */}
        {missedCoughPoints.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-xs text-gray-500">
              Tap timeline to mark a missed cough
            </span>
          </div>
        )}
      </div>

      {/* Chips row */}
      {missedCoughPoints.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {missedCoughPoints.map((t, i) => (
            <span
              key={`chip-${i}`}
              className="flex items-center gap-1 rounded-full bg-amber-900/40 px-3 py-1 text-xs text-amber-300"
            >
              <span className="font-mono tabular-nums">{formatSeconds(t)}</span>
              <button
                type="button"
                className="ml-1 text-amber-400 hover:text-amber-200"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  removePin(i);
                }}
                aria-label={`Remove ${formatSeconds(t)}`}
              >
                ✕
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

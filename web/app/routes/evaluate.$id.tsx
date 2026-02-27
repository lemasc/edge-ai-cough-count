import { useRef, useState } from "react";
import { Form, Link, redirect } from "react-router";
import type { Route } from "./+types/evaluate.$id";
import { getDb } from "~/db";
import * as schema from "~/db/schema";
import { eq } from "drizzle-orm";
import { WaveformPlayer } from "~/components/WaveformPlayer";
import type { WaveformPlayerHandle } from "~/components/WaveformPlayer";
import { EvaluationTimeline } from "~/components/EvaluationTimeline";
import { formatSeconds } from "~/utils/formatTime";

type EventVerdict = "tp" | "mixed" | "fp";

type DetectedEventEval = {
  start: number;
  end: number;
  verdict: EventVerdict;
};

export async function loader({ params, context }: Route.LoaderArgs) {
  const db = getDb(context.cloudflare.env.DB);
  const recording = await db.query.recordings.findFirst({
    where: eq(schema.recordings.id, params.id),
  });
  if (!recording) throw new Response("Not found", { status: 404 });
  if (recording.status !== "done") return redirect(`/predict/${recording.id}`);
  return recording;
}

export async function action({ params, request, context }: Route.ActionArgs) {
  const db = getDb(context.cloudflare.env.DB);
  const recording = await db.query.recordings.findFirst({
    where: eq(schema.recordings.id, params.id),
  });
  if (!recording || recording.status !== "done") {
    throw new Response("Not found", { status: 404 });
  }

  const formData = await request.formData();
  const detectedEvents = String(formData.get("detectedEvents") ?? "[]");
  const missedCoughPoints = String(formData.get("missedCoughPoints") ?? "[]");

  await db.insert(schema.evaluations).values({
    id: crypto.randomUUID(),
    recordingId: recording.id,
    createdAt: new Date(),
    detectedEvents,
    missedCoughPoints,
  });

  return redirect("/complete");
}

const VERDICT_OPTIONS: { value: EventVerdict; label: string; active: string }[] = [
  { value: "tp", label: "Correct", active: "bg-green-600 text-white" },
  { value: "mixed", label: "Partial", active: "bg-amber-500 text-white" },
  { value: "fp", label: "False Positive", active: "bg-red-600 text-white" },
];

function VerdictToggle({
  verdict,
  onChange,
}: {
  verdict: EventVerdict;
  onChange: (v: EventVerdict) => void;
}) {
  return (
    <div className="flex w-full overflow-hidden rounded-lg border border-gray-700">
      {VERDICT_OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`flex-1 min-h-9 px-2 py-1.5 text-xs font-semibold transition ${
            verdict === opt.value
              ? opt.active
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export default function EvaluateRoute({ loaderData }: Route.ComponentProps) {
  const recording = loaderData;

  const startTimes = JSON.parse(recording.startTimes ?? "[]") as number[];
  const endTimes = JSON.parse(recording.endTimes ?? "[]") as number[];
  const durationSecs = recording.durationMs != null ? recording.durationMs / 1000 : 0;

  const [detectedEventEvals, setDetectedEventEvals] = useState<DetectedEventEval[]>(
    () => startTimes.map((s, i) => ({ start: s, end: endTimes[i] ?? s, verdict: "tp" as EventVerdict })),
  );
  const [missedCoughPoints, setMissedCoughPoints] = useState<number[]>([]);
  const waveformRef = useRef<WaveformPlayerHandle>(null);

  function updateVerdict(index: number, verdict: EventVerdict) {
    setDetectedEventEvals((prev) =>
      prev.map((ev, i) => (i === index ? { ...ev, verdict } : ev)),
    );
  }

  return (
    <div className="w-full max-w-sm space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">Evaluate Results</h1>
        <p className="mt-1 text-sm text-gray-400">
          Review detected events and mark any coughs the model missed.
        </p>
      </div>

      <WaveformPlayer
        ref={waveformRef}
        src={`/results/${recording.id}/audio`}
        startTimes={startTimes}
        endTimes={endTimes}
      />

      {/* Section 1: Detected events */}
      <div className="space-y-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
          Review Detected Events
        </p>
        {detectedEventEvals.length === 0 ? (
          <div className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-center text-sm text-gray-500">
            No detections to review
          </div>
        ) : (
          <div className="space-y-3">
            {detectedEventEvals.map((ev, i) => (
              <div
                key={`${ev.start}-${ev.end}`}
                className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Cough {i + 1}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs tabular-nums text-gray-200">
                      {formatSeconds(ev.start)} – {formatSeconds(ev.end)}
                    </span>
                    <button
                      type="button"
                      onClick={() => waveformRef.current?.seekTo(ev.start)}
                      className="flex h-7 w-7 items-center justify-center rounded-lg border border-gray-700 text-gray-400 hover:border-gray-500 hover:text-white transition active:scale-90"
                      aria-label={`Seek to cough ${i + 1}`}
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                        className="h-3.5 w-3.5"
                      >
                        <path
                          fillRule="evenodd"
                          d="M4.5 5.653c0-1.427 1.529-2.33 2.779-1.643l11.54 6.347c1.295.712 1.295 2.573 0 3.286L7.28 19.99c-1.25.687-2.779-.217-2.779-1.643V5.653Z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
                <VerdictToggle
                  verdict={ev.verdict}
                  onChange={(v) => updateVerdict(i, v)}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Section 2: Missed coughs timeline */}
      <EvaluationTimeline
        durationSecs={durationSecs}
        detectedStartTimes={startTimes}
        detectedEndTimes={endTimes}
        missedCoughPoints={missedCoughPoints}
        onChange={setMissedCoughPoints}
      />

      {/* Form submission */}
      <Form method="post" className="space-y-3">
        <input
          type="hidden"
          name="detectedEvents"
          value={JSON.stringify(detectedEventEvals)}
        />
        <input
          type="hidden"
          name="missedCoughPoints"
          value={JSON.stringify(missedCoughPoints)}
        />
        <button
          type="submit"
          className="flex min-h-12 w-full items-center justify-center rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
        >
          Submit Evaluation
        </button>
        <Link
          to="/complete"
          className="flex min-h-12 w-full items-center justify-center rounded-xl border border-gray-700 px-6 py-3 text-base font-semibold text-gray-400 transition hover:border-gray-500 hover:text-white active:scale-95"
        >
          Skip
        </Link>
      </Form>
    </div>
  );
}

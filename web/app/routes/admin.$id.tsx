import { useRef, useState } from "react";
import { Form, Link, redirect } from "react-router";
import type { Route } from "./+types/admin.$id";
import { getDb } from "~/db";
import * as schema from "~/db/schema";
import { desc, eq } from "drizzle-orm";
import { WaveformPlayer } from "~/components/WaveformPlayer";
import type { WaveformPlayerHandle } from "~/components/WaveformPlayer";
import { formatSeconds } from "~/utils/formatTime";
import { PlayIcon, XIcon } from "lucide-react";
import type { DetectedEventEval, EventVerdict } from "~/types";

export async function loader({ params, context }: Route.LoaderArgs) {
  const db = getDb(context.cloudflare.env.DB);
  const recording = await db.query.recordings.findFirst({
    where: eq(schema.recordings.id, params.id),
  });
  if (!recording) throw new Response("Not found", { status: 404 });
  if (recording.status !== "done") return redirect(`/predict/${recording.id}`);

  let displayEvaluation: typeof schema.evaluations.$inferSelect | null = null;
  let isConfirmed = false;

  if (recording.confirmedEvaluationId) {
    displayEvaluation = await db.query.evaluations.findFirst({
      where: eq(schema.evaluations.id, recording.confirmedEvaluationId),
    }) ?? null;
    isConfirmed = true;
  } else {
    displayEvaluation = await db.query.evaluations.findFirst({
      where: eq(schema.evaluations.recordingId, recording.id),
      orderBy: desc(schema.evaluations.createdAt),
    }) ?? null;
  }

  return { recording, displayEvaluation, isConfirmed };
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
  const intent = String(formData.get("intent") ?? "");

  if (intent === "confirm") {
    const evaluationId = String(formData.get("evaluationId") ?? "");
    await db
      .update(schema.recordings)
      .set({ confirmedEvaluationId: evaluationId })
      .where(eq(schema.recordings.id, recording.id));
    return redirect("/admin");
  }

  if (intent === "save") {
    const detectedEvents = String(formData.get("detectedEvents") ?? "[]");
    const missedCoughPoints = String(formData.get("missedCoughPoints") ?? "[]");
    const newId = crypto.randomUUID();
    await db.insert(schema.evaluations).values({
      id: newId,
      recordingId: recording.id,
      createdAt: new Date(),
      detectedEvents,
      missedCoughPoints,
    });
    await db
      .update(schema.recordings)
      .set({ confirmedEvaluationId: newId })
      .where(eq(schema.recordings.id, recording.id));
    return redirect("/admin");
  }

  throw new Response("Bad request", { status: 400 });
}

const VERDICT_OPTIONS: {
  value: EventVerdict;
  label: string;
  active: string;
}[] = [
  { value: "tp", label: "Correct", active: "bg-green-600 text-white" },
  { value: "mixed", label: "Partial", active: "bg-amber-500 text-white" },
  { value: "fp", label: "Wrong", active: "bg-red-600 text-white" },
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

const STATUS_CONFIG = {
  confirmed: { label: "Confirmed", className: "bg-green-900 text-green-300" },
  pending_review: {
    label: "Pending review",
    className: "bg-amber-900 text-amber-300",
  },
  none: { label: "No evaluation", className: "bg-gray-800 text-gray-400" },
};

export default function AdminReview({ loaderData }: Route.ComponentProps) {
  const { recording, displayEvaluation, isConfirmed } = loaderData;

  const startTimes = JSON.parse(recording.startTimes ?? "[]") as number[];
  const endTimes = JSON.parse(recording.endTimes ?? "[]") as number[];

  // Pre-populate state from existing evaluation, or defaults from recording
  const [detectedEventEvals, setDetectedEventEvals] = useState<
    DetectedEventEval[]
  >(() => {
    if (displayEvaluation) {
      return JSON.parse(displayEvaluation.detectedEvents) as DetectedEventEval[];
    }
    return startTimes.map((s, i) => ({
      start: s,
      end: endTimes[i] ?? s,
      verdict: "tp" as EventVerdict,
    }));
  });

  const [missedCoughPoints, setMissedCoughPoints] = useState<number[]>(() => {
    if (displayEvaluation) {
      return JSON.parse(displayEvaluation.missedCoughPoints) as number[];
    }
    return [];
  });

  const [annotating, setAnnotating] = useState(false);
  const [pendingTime, setPendingTime] = useState<number | null>(null);
  const waveformRef = useRef<WaveformPlayerHandle>(null);

  const statusKey: keyof typeof STATUS_CONFIG = isConfirmed
    ? "confirmed"
    : displayEvaluation
    ? "pending_review"
    : "none";
  const statusConfig = STATUS_CONFIG[statusKey];

  function updateVerdict(index: number, verdict: EventVerdict) {
    setDetectedEventEvals((prev) =>
      prev.map((ev, i) => (i === index ? { ...ev, verdict } : ev)),
    );
  }

  function confirmAnnotation() {
    if (pendingTime === null) return;
    setMissedCoughPoints((prev) =>
      [...prev, pendingTime].sort((a, b) => a - b),
    );
    setPendingTime(null);
    setAnnotating(false);
  }

  function cancelAnnotation() {
    setPendingTime(null);
    setAnnotating(false);
  }

  if (annotating) {
    return (
      <div className="mx-auto w-full max-w-sm space-y-4">
        <Link
          to="/admin"
          className="inline-block text-sm text-gray-400 hover:text-white transition"
        >
          ← Back to admin
        </Link>
        <div>
          <h1 className="text-xl font-bold text-white">Mark missed cough</h1>
          <p className="mt-1 text-sm text-gray-200">
            Play the audio and pause at the missed cough, then click{" "}
            <b>Mark at this time</b>.
          </p>
        </div>

        <WaveformPlayer
          ref={waveformRef}
          src={`/results/${recording.id}/audio`}
          startTimes={startTimes}
          endTimes={endTimes}
          height={128}
          annotating={true}
          pendingAnnotationTime={pendingTime}
          markerTimes={missedCoughPoints}
        />

        <div className="space-y-2">
          {pendingTime !== null && (
            <p className="text-center text-sm text-amber-300">
              {"Marked at — "}
              <span className="font-mono tabular-nums">
                {formatSeconds(pendingTime)}
              </span>
            </p>
          )}
          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={() =>
                setPendingTime(waveformRef.current?.getCurrentTime() ?? 0)
              }
              className="flex min-h-10 w-full items-center justify-center rounded-xl border border-amber-500 px-4 py-2 text-sm font-semibold text-amber-300 transition hover:bg-amber-500/10 active:scale-95"
            >
              Mark at this time
            </button>
            {pendingTime !== null && (
              <button
                type="button"
                onClick={confirmAnnotation}
                className="flex min-h-10 flex-1 items-center justify-center rounded-xl bg-amber-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-amber-400 active:scale-95"
              >
                Confirm
              </button>
            )}
            <button
              type="button"
              onClick={cancelAnnotation}
              className="flex min-h-10 flex-1 items-center justify-center rounded-xl border border-gray-700 px-4 py-2 text-sm font-semibold text-gray-400 transition hover:border-gray-500 hover:text-white active:scale-95"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-sm space-y-2">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-950 py-4 space-y-4 border-b border-gray-600">
        <div className="flex items-start justify-between gap-3">
          <div>
            <Link
              to="/admin"
              className="text-sm text-gray-400 hover:text-white transition"
            >
              ← Back to admin
            </Link>
            <h1 className="mt-1 text-xl font-bold text-white">Review recording</h1>
          </div>
          <span
            className={`mt-1 inline-block shrink-0 rounded-full px-2.5 py-0.5 text-xs font-semibold ${statusConfig.className}`}
          >
            {statusConfig.label}
          </span>
        </div>

        <WaveformPlayer
          ref={waveformRef}
          src={`/results/${recording.id}/audio`}
          startTimes={startTimes}
          endTimes={endTimes}
          height={72}
          markerTimes={missedCoughPoints}
        />
      </div>

      {/* Scrollable content */}
      <div className="space-y-6 pt-2">
        {/* Section 1: Detected events */}
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
            Detected events
          </p>
          {detectedEventEvals.length === 0 ? (
            <div className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-center text-sm text-gray-500">
              No coughs detected
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
                        onClick={() => waveformRef.current?.playFrom(ev.start)}
                        className="flex h-7 w-7 items-center justify-center rounded-lg border border-gray-700 text-gray-400 hover:border-gray-500 hover:text-white transition active:scale-90"
                        aria-label={`Seek to cough ${i + 1}`}
                      >
                        <PlayIcon className="h-3.5 w-3.5" />
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

        {/* Section 2: Missed coughs */}
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
            Missed coughs
          </p>
          {missedCoughPoints.length === 0 ? (
            <div className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-center text-sm text-gray-500">
              None recorded
            </div>
          ) : (
            <div className="space-y-3">
              {missedCoughPoints.map((t, i) => (
                <div
                  key={`missed-${t}`}
                  className="rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-gray-400">Missed {i + 1}</span>
                    <span className="font-mono text-xs tabular-nums text-gray-200">
                      {formatSeconds(t)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() =>
                        waveformRef.current?.playFrom(Math.max(t - 0.1, 0))
                      }
                      className="flex h-7 w-7 items-center justify-center rounded-lg border border-gray-700 text-gray-400 hover:border-gray-500 hover:text-white transition active:scale-90"
                      aria-label={`Seek to missed cough ${i + 1}`}
                    >
                      <PlayIcon className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setMissedCoughPoints((prev) =>
                          prev.filter((_, j) => j !== i),
                        )
                      }
                      className="flex h-7 w-7 items-center justify-center rounded-lg border border-gray-700 text-gray-400 hover:border-red-500 hover:text-red-400 transition active:scale-90"
                      aria-label={`Remove missed cough ${i + 1}`}
                    >
                      <XIcon className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
          <button
            type="button"
            onClick={() => setAnnotating(true)}
            className="flex min-h-10 w-full items-center justify-center gap-2 rounded-xl border border-gray-700 px-4 py-2 text-sm font-semibold text-gray-300 transition hover:border-gray-500 hover:text-white active:scale-95"
          >
            + Add missed cough
          </button>
        </div>

        {/* Actions */}
        <div className="space-y-3">
          {/* Confirm as-is (only when displayEvaluation exists) */}
          {displayEvaluation && (
            <Form method="post">
              <input type="hidden" name="intent" value="confirm" />
              <input
                type="hidden"
                name="evaluationId"
                value={displayEvaluation.id}
              />
              <button
                type="submit"
                className="flex min-h-12 w-full items-center justify-center rounded-xl bg-green-700 px-6 py-3 text-base font-semibold text-white transition hover:bg-green-600 active:scale-95"
              >
                Confirm as-is
              </button>
            </Form>
          )}

          {/* Save changes */}
          <Form method="post">
            <input type="hidden" name="intent" value="save" />
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
              Save changes
            </button>
          </Form>
        </div>
      </div>
    </div>
  );
}

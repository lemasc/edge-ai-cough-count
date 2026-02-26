import { Link, redirect } from "react-router";
import type { Route } from "./+types/results.$id";
import { getDb } from "~/db";
import * as schema from "~/db/schema";
import { eq } from "drizzle-orm";
import { WaveformPlayer } from "~/components/WaveformPlayer";
import { formatSeconds } from "~/utils/formatTime";

export async function loader({ params, context }: Route.LoaderArgs) {
  const db = getDb(context.cloudflare.env.DB);
  const recording = await db.query.recordings.findFirst({
    where: eq(schema.recordings.id, params.id),
  });
  if (!recording) throw new Response("Not found", { status: 404 });
  if (recording.status !== "done") return redirect(`/predict/${recording.id}`);
  return recording;
}

export default function ResultsRoute({ loaderData }: Route.ComponentProps) {
  const recording = loaderData;

  const startTimes = JSON.parse(recording.startTimes ?? "[]") as number[];
  const endTimes = JSON.parse(recording.endTimes ?? "[]") as number[];
  const coughCount = recording.coughCount ?? 0;

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="text-center">
        <div className="mb-2 text-5xl font-bold tabular-nums">{coughCount}</div>
        <p className="text-lg text-gray-300">
          {coughCount === 1 ? "cough detected" : "coughs detected"}
        </p>
      </div>

      <WaveformPlayer
        src={`/results/${recording.id}/audio`}
        startTimes={startTimes}
        endTimes={endTimes}
      />

      {startTimes.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
            Detected Events
          </p>
          <div className="space-y-1 rounded-xl border border-gray-700 bg-gray-900 px-4 py-3">
            {startTimes.map((start, i) => {
              const end = endTimes[i] ?? start;
              return (
                <div
                  key={`${start}-${end}`}
                  className="flex justify-between text-sm"
                >
                  <span className="text-gray-500">Cough {i + 1}</span>
                  <span className="font-mono text-gray-200">
                    {formatSeconds(start)} – {formatSeconds(end)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {startTimes.length === 0 && (
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
        <Link
          to="/record"
          className="flex min-h-12 w-full items-center justify-center rounded-xl border border-gray-700 px-6 py-3 text-base font-semibold text-gray-400 transition hover:border-gray-500 hover:text-white active:scale-95"
        >
          Record Another
        </Link>
      </div>
    </div>
  );
}

import { Link } from "react-router";
import type { Route } from "./+types/admin._index";
import { getDb } from "~/db";
import * as schema from "~/db/schema";
import { desc } from "drizzle-orm";

export async function loader({ context }: Route.LoaderArgs) {
  const db = getDb(context.cloudflare.env.DB);

  const [recordings, evaluations] = await Promise.all([
    db.query.recordings.findMany({ orderBy: desc(schema.recordings.createdAt) }),
    db
      .select({
        id: schema.evaluations.id,
        recordingId: schema.evaluations.recordingId,
        createdAt: schema.evaluations.createdAt,
      })
      .from(schema.evaluations),
  ]);

  // Build map of recordingId → latest evaluation id
  const latestEvalByRecording = new Map<string, string>();
  for (const ev of evaluations) {
    const existing = latestEvalByRecording.get(ev.recordingId);
    if (!existing) {
      latestEvalByRecording.set(ev.recordingId, ev.id);
    } else {
      // Keep the most recent by createdAt — evaluations come in any order so compare
      const existingEv = evaluations.find((e) => e.id === existing);
      if (existingEv && ev.createdAt > existingEv.createdAt) {
        latestEvalByRecording.set(ev.recordingId, ev.id);
      }
    }
  }

  const rows = recordings.map((r) => {
    let evalStatus: "confirmed" | "pending_review" | "none";
    if (r.confirmedEvaluationId) {
      evalStatus = "confirmed";
    } else if (latestEvalByRecording.has(r.id)) {
      evalStatus = "pending_review";
    } else {
      evalStatus = "none";
    }
    return { ...r, evalStatus };
  });

  return { rows };
}

const STATUS_BADGE: Record<
  "confirmed" | "pending_review" | "none",
  { label: string; className: string }
> = {
  confirmed: {
    label: "Confirmed",
    className: "bg-green-900 text-green-300",
  },
  pending_review: {
    label: "Pending review",
    className: "bg-amber-900 text-amber-300",
  },
  none: {
    label: "No evaluation",
    className: "bg-gray-800 text-gray-400",
  },
};

export default function AdminIndex({ loaderData }: Route.ComponentProps) {
  const { rows } = loaderData;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-white">Recordings</h1>
      <div className="overflow-x-auto rounded-xl border border-gray-800">
        <table className="w-full text-sm">
          <thead className="border-b border-gray-800 bg-gray-900 text-xs uppercase tracking-wider text-gray-500">
            <tr>
              <th className="px-4 py-3 text-left">#</th>
              <th className="px-4 py-3 text-left">Date / Time (BKK)</th>
              <th className="px-4 py-3 text-right">Duration</th>
              <th className="px-4 py-3 text-right">Coughs</th>
              <th className="px-4 py-3 text-left">Status</th>
              <th className="px-4 py-3 text-left">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {rows.map((row, i) => {
              const badge = STATUS_BADGE[row.evalStatus];
              const dateStr = row.createdAt
                ? row.createdAt.toLocaleString("en-GB", {
                    timeZone: "Asia/Bangkok",
                    day: "2-digit",
                    month: "2-digit",
                    year: "2-digit",
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : "—";
              const duration =
                row.durationMs != null
                  ? (row.durationMs / 1000).toFixed(1) + "s"
                  : "—";
              return (
                <tr key={row.id} className="hover:bg-gray-900/50">
                  <td className="px-4 py-3 font-mono text-gray-400">
                    {rows.length - i}
                  </td>
                  <td className="px-4 py-3 font-mono text-gray-200">
                    {dateStr}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-gray-300">
                    {duration}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-gray-200">
                    {row.coughCount ?? "—"}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${badge.className}`}
                    >
                      {badge.label}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      to={`/admin/${row.id}`}
                      className="text-blue-400 hover:text-blue-300 transition"
                    >
                      Review →
                    </Link>
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={6}
                  className="px-4 py-8 text-center text-gray-500"
                >
                  No recordings yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

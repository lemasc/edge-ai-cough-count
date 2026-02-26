import type { PermissionStatus } from "../types";

type Props = {
  phase: "idle" | "requesting-permissions" | "ready";
  permissions?: PermissionStatus;
  onStart: () => void;
  onBeginRecording: () => void;
};

type BadgeStatus = "unknown" | "granted" | "denied" | "unavailable";

function StatusBadge({
  status,
  label,
}: {
  status: BadgeStatus;
  label: string;
}) {
  const colors: Record<BadgeStatus, string> = {
    unknown: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
    granted: "bg-green-500/20 text-green-300 border-green-500/40",
    denied: "bg-red-500/20 text-red-300 border-red-500/40",
    unavailable: "bg-gray-500/20 text-gray-400 border-gray-500/40",
  };

  const icons: Record<BadgeStatus, string> = {
    unknown: "?",
    granted: "✓",
    denied: "✗",
    unavailable: "—",
  };

  const statusText: Record<BadgeStatus, string> = {
    unknown: "Unknown",
    granted: "Granted",
    denied: "Denied",
    unavailable: "Unavailable",
  };

  return (
    <div
      className={`flex items-center justify-between rounded-lg border px-4 py-3 ${colors[status]}`}
    >
      <span className="text-sm font-medium">{label}</span>
      <span className="text-sm font-semibold">
        {icons[status]} {statusText[status]}
      </span>
    </div>
  );
}

export function PermissionScreen({
  phase,
  permissions,
  onStart,
  onBeginRecording,
}: Props) {
  return (
    <div className="w-full max-w-sm space-y-8">
      <div className="text-center">
        <div className="mb-4 text-5xl">🎤</div>
        <h1 className="text-2xl font-bold">Cough Dataset Collector</h1>
        <p className="mt-2 text-gray-400 text-sm">
          Records audio for cough detection research
        </p>
      </div>

      {phase !== "idle" && permissions && (
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
            Permissions
          </p>
          <StatusBadge status={permissions.audio} label="Microphone" />
          {permissions.audio === "denied" && (
            <p className="text-xs text-red-400">
              Microphone access denied. Please allow in browser settings and
              reload.
            </p>
          )}
        </div>
      )}

      {phase === "idle" && (
        <button
          type="button"
          onClick={onStart}
          className="min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
        >
          Start
        </button>
      )}

      {phase === "requesting-permissions" && (
        <button
          type="button"
          disabled
          className="min-h-12 w-full cursor-not-allowed rounded-xl bg-blue-600/50 px-6 py-3 text-base font-semibold text-white/50"
        >
          Requesting permissions…
        </button>
      )}

      {phase === "ready" && permissions && (
        <button
          type="button"
          onClick={onBeginRecording}
          disabled={permissions.audio !== "granted"}
          className="min-h-12 w-full rounded-xl bg-green-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-green-500 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Begin Recording
        </button>
      )}
    </div>
  );
}

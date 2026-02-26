import { useState } from "react";
import { redirect, useNavigation, useSubmit } from "react-router";
import { PermissionScreen } from "~/components/PermissionScreen";
import { RecordingScreen } from "~/components/RecordingScreen";
import { getDb } from "~/db";
import * as schema from "~/db/schema";
import { useAudioRecorder } from "~/hooks/useAudioRecorder";
import { mimeToExt } from "~/utils/audio";
import type { Route } from "./+types/record";

export async function action({ request, context }: Route.ActionArgs) {
  const { env } = context.cloudflare;
  const db = getDb(env.DB);

  const formData = await request.formData();
  const audio = formData.get("audio") as File;
  const durationMs = parseInt(formData.get("durationMs") as string, 10);

  const id = crypto.randomUUID();
  const ext = mimeToExt(audio.type);
  const audioKey = `recordings/${id}/audio.${ext}`;

  await db.insert(schema.recordings).values({
    id,
    createdAt: new Date(),
    durationMs,
    audioKey,
    status: "pending",
  });

  const buffer = await audio.arrayBuffer();

  await env.STORAGE.put(audioKey, buffer, {
    httpMetadata: { contentType: audio.type },
  });

  return redirect(`/predict/${id}`);
}

export function HydrateFallback() {
  return (
    <div className="h-10 w-10 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500" />
  );
}

export default function RecordRoute() {
  const [phase, setPhase] = useState<"idle" | "countdown" | "recording">(
    "idle",
  );
  const [startTime, setStartTime] = useState(0);
  const [durationMs, setDurationMs] = useState<number | null>(null);

  const audio = useAudioRecorder();
  const navigation = useNavigation();
  const submit = useSubmit();

  const handleBeginRecording = () => {
    setPhase("countdown");
  };

  const handleCountdownEnd = () => {
    const t = performance.now();
    audio.startRecording();
    setStartTime(t);
    setPhase("recording");
  };

  const handleStop = () => {
    void audio.stopRecording().then((blob) => {
      const ms = performance.now() - startTime;
      setDurationMs(ms);
      const ext = mimeToExt(audio.getMimeType());
      const fd = new FormData();
      fd.append("audio", blob, `recording.${ext}`);
      fd.append("durationMs", String(ms));
      submit(fd, { method: "post", encType: "multipart/form-data" });
    });
  };

  if (navigation.state !== "idle") {
    return (
      <div className="w-full max-w-sm space-y-6 text-center">
        <div className="flex flex-col items-center gap-4">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500" />
          <p className="text-lg font-semibold">Analyzing audio…</p>
          {durationMs != null && (
            <p className="text-sm text-gray-500">
              Duration:{" "}
              {String(Math.floor(durationMs / 60000)).padStart(2, "0")}:
              {String(Math.floor((durationMs % 60000) / 1000)).padStart(2, "0")}
            </p>
          )}
        </div>
      </div>
    );
  }

  if (phase === "idle") {
    return (
      <PermissionScreen
        requestPermission={audio.requestPermission}
        onBeginRecording={handleBeginRecording}
      />
    );
  }

  return (
    <RecordingScreen
      phase={phase}
      startTime={startTime}
      onCountdownEnd={handleCountdownEnd}
      onStop={handleStop}
    />
  );
}

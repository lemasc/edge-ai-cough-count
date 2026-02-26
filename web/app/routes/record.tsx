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
  const [phase, setPhase] = useState<
    "idle" | "countdown" | "recording" | "stopped"
  >("idle");
  const [startTime, setStartTime] = useState(0);
  const [stoppedResult, setStoppedResult] = useState<{
    durationMs: number;
    audioBlob: Blob;
  } | null>(null);

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
      setStoppedResult({ durationMs: ms, audioBlob: blob });
      setPhase("stopped");
    });
  };

  const handlePredict = () => {
    if (!stoppedResult) return;
    const { audioBlob, durationMs } = stoppedResult;
    const ext = mimeToExt(audio.getMimeType());
    const fd = new FormData();
    fd.append("audio", audioBlob, `recording.${ext}`);
    fd.append("durationMs", String(durationMs));
    submit(fd, { method: "post", encType: "multipart/form-data" });
  };

  if (navigation.state !== "idle") {
    return (
      <div className="w-full max-w-sm space-y-6 text-center">
        <div className="flex flex-col items-center gap-4">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500" />
          <p className="text-lg font-semibold">Analyzing audio…</p>
          {stoppedResult && (
            <p className="text-sm text-gray-500">
              Duration:{" "}
              {String(Math.floor(stoppedResult.durationMs / 60000)).padStart(
                2,
                "0",
              )}
              :
              {String(
                Math.floor((stoppedResult.durationMs % 60000) / 1000),
              ).padStart(2, "0")}
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

  if (phase === "countdown" || phase === "recording") {
    return (
      <RecordingScreen
        phase={phase}
        startTime={startTime}
        onCountdownEnd={handleCountdownEnd}
        onStop={handleStop}
        onPredict={() => {}}
      />
    );
  }

  if (phase === "stopped") {
    return (
      <RecordingScreen
        phase="stopped"
        startTime={0}
        durationMs={stoppedResult?.durationMs}
        onCountdownEnd={() => {}}
        onStop={() => {}}
        onPredict={handlePredict}
      />
    );
  }

  return null;
}

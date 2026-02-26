import { Form, Link, redirect, useNavigation, useSubmit } from "react-router";
import { useEffect } from "react";
import type { Route } from "./+types/predict.$id";
import { getDb } from "~/db";
import * as schema from "~/db/schema";
import { eq } from "drizzle-orm";
import type { PredictionResult } from "~/types";

function mimeToExt(mimeType: string): string {
  if (mimeType.includes("mp4")) return "mp4";
  if (mimeType.includes("ogg")) return "ogg";
  return "webm";
}

export async function loader({ params, context }: Route.LoaderArgs) {
  const db = getDb(context.cloudflare.env.DB);
  const recording = await db.query.recordings.findFirst({
    where: eq(schema.recordings.id, params.id),
  });
  if (!recording) throw new Response("Not found", { status: 404 });
  if (recording.status === "done") return redirect(`/results/${recording.id}`);
  return recording;
}

export async function action({ params, context }: Route.ActionArgs) {
  const { env } = context.cloudflare;
  const db = getDb(env.DB);
  const recording = await db.query.recordings.findFirst({
    where: eq(schema.recordings.id, params.id),
  });
  if (!recording) throw new Response("Not found", { status: 404 });
  if (recording.status === "done") return redirect(`/results/${recording.id}`);

  try {
    const audioObject = await env.STORAGE.get(recording.audioKey);
    if (!audioObject) throw new Error("Recording audio not found");

    const buffer = await audioObject.arrayBuffer();
    const contentType = audioObject.httpMetadata?.contentType ?? "audio/webm";
    const ext = mimeToExt(contentType);
    const predictFormData = new FormData();
    predictFormData.append(
      "audio",
      new Blob([buffer], { type: contentType }),
      `recording.${ext}`,
    );

    const response = await fetch(`${env.PREDICT_API_URL}/api/predict`, {
      method: "POST",
      body: predictFormData,
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const prediction = (await response.json()) as PredictionResult;

    await db
      .update(schema.recordings)
      .set({
        status: "done",
        coughCount: prediction.cough_count,
        startTimes: JSON.stringify(prediction.start_times),
        endTimes: JSON.stringify(prediction.end_times),
        windowTimes: JSON.stringify(prediction.window_times),
        probabilities: JSON.stringify(prediction.probabilities),
        errorMessage: null,
      })
      .where(eq(schema.recordings.id, recording.id));

    return redirect(`/results/${recording.id}`);
  } catch (err) {
    await db
      .update(schema.recordings)
      .set({
        status: "error",
        errorMessage: err instanceof Error ? err.message : "Unknown error",
      })
      .where(eq(schema.recordings.id, recording.id));

    return redirect(`/predict/${recording.id}`);
  }
}

export default function PredictRoute({ loaderData }: Route.ComponentProps) {
  const recording = loaderData;
  const submit = useSubmit();
  const navigation = useNavigation();

  useEffect(() => {
    if (recording.status !== "pending") return;
    if (navigation.state !== "idle") return;
    submit(new FormData(), { method: "post" });
  }, [recording.status, navigation.state, submit]);

  if (navigation.state !== "idle") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
        <div className="w-full max-w-sm space-y-6 text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-700 border-t-blue-500" />
            <p className="text-lg font-semibold">Analyzing audio…</p>
          </div>
        </div>
      </div>
    );
  }

  if (recording.status === "error") {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
        <div className="w-full max-w-sm space-y-6 text-center">
          <p className="text-lg font-semibold text-red-400">Analysis failed</p>
          {recording.errorMessage && (
            <p className="text-sm text-gray-500">{recording.errorMessage}</p>
          )}
          <div className="space-y-3">
            <Form method="post">
              <button
                type="submit"
                className="min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
              >
                Retry Analysis
              </button>
            </Form>
            <Link
              to="/record"
              className="block min-h-12 w-full rounded-xl border border-gray-700 px-6 py-3 text-base font-semibold text-gray-400 transition hover:border-gray-500 hover:text-white active:scale-95"
            >
              Record Again
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
      <div className="w-full max-w-sm space-y-6 text-center">
        <p className="text-lg font-semibold">Ready to analyze</p>
        <Form method="post">
          <button
            type="submit"
            className="min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
          >
            Start Analysis
          </button>
        </Form>
      </div>
    </div>
  );
}

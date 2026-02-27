import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const recordings = sqliteTable("recordings", {
  id: text("id").primaryKey(),
  createdAt: integer("created_at", { mode: "timestamp_ms" }).notNull(),
  durationMs: integer("duration_ms"),
  audioKey: text("audio_key").notNull(),
  status: text("status", { enum: ["pending", "done", "error"] })
    .notNull()
    .default("pending"),
  coughCount: integer("cough_count"),
  startTimes: text("start_times"), // JSON
  endTimes: text("end_times"), // JSON
  windowTimes: text("window_times"), // JSON
  probabilities: text("probabilities"), // JSON
  errorMessage: text("error_message"),
});

export const evaluations = sqliteTable("evaluations", {
  id: text("id").primaryKey(),
  recordingId: text("recording_id")
    .notNull()
    .references(() => recordings.id),
  createdAt: integer("created_at", { mode: "timestamp_ms" }).notNull(),
  detectedEvents: text("detected_events").notNull(), // JSON: DetectedEventEval[]
  missedCoughPoints: text("missed_cough_points").notNull(), // JSON: number[]
});

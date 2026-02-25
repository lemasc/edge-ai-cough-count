import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const recordings = sqliteTable("recordings", {
  id: text("id").primaryKey(),
  createdAt: integer("created_at", { mode: "timestamp_ms" }).notNull(),
  durationMs: integer("duration_ms"),
  audioKey: text("audio_key"),
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

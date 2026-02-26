PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_recordings` (
	`id` text PRIMARY KEY NOT NULL,
	`created_at` integer NOT NULL,
	`duration_ms` integer,
	`audio_key` text NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`cough_count` integer,
	`start_times` text,
	`end_times` text,
	`window_times` text,
	`probabilities` text,
	`error_message` text
);
--> statement-breakpoint
INSERT INTO `__new_recordings`("id", "created_at", "duration_ms", "audio_key", "status", "cough_count", "start_times", "end_times", "window_times", "probabilities", "error_message") SELECT "id", "created_at", "duration_ms", "audio_key", "status", "cough_count", "start_times", "end_times", "window_times", "probabilities", "error_message" FROM `recordings`;--> statement-breakpoint
DROP TABLE `recordings`;--> statement-breakpoint
ALTER TABLE `__new_recordings` RENAME TO `recordings`;--> statement-breakpoint
PRAGMA foreign_keys=ON;
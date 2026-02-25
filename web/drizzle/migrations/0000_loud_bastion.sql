CREATE TABLE `recordings` (
	`id` text PRIMARY KEY NOT NULL,
	`created_at` integer NOT NULL,
	`duration_ms` integer,
	`audio_key` text,
	`status` text DEFAULT 'pending' NOT NULL,
	`cough_count` integer,
	`start_times` text,
	`end_times` text,
	`window_times` text,
	`probabilities` text,
	`error_message` text
);

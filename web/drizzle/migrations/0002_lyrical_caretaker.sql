CREATE TABLE `evaluations` (
	`id` text PRIMARY KEY NOT NULL,
	`recording_id` text NOT NULL,
	`created_at` integer NOT NULL,
	`detected_events` text NOT NULL,
	`missed_cough_points` text NOT NULL,
	FOREIGN KEY (`recording_id`) REFERENCES `recordings`(`id`) ON UPDATE no action ON DELETE no action
);

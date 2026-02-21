export type Sound = 'cough' | 'laugh' | 'deep_breathing' | 'throat_clearing';
export type Movement = 'sit' | 'walk';
export type Noise = 'nothing' | 'music' | 'someone_else_cough' | 'traffic';
export type Trial = '1' | '2' | '3';

export const SOUND_OPTIONS: { value: Sound; label: string }[] = [
  { value: 'cough', label: 'Cough' },
  { value: 'laugh', label: 'Laugh' },
  { value: 'deep_breathing', label: 'Deep Breathing' },
  { value: 'throat_clearing', label: 'Throat Clearing' },
];

export const MOVEMENT_OPTIONS: { value: Movement; label: string }[] = [
  { value: 'sit', label: 'Sitting' },
  { value: 'walk', label: 'Walking' },
];

export const NOISE_OPTIONS: { value: Noise; label: string }[] = [
  { value: 'nothing', label: 'No Background Noise' },
  { value: 'music', label: 'Music' },
  { value: 'someone_else_cough', label: "Someone Else Coughing" },
  { value: 'traffic', label: 'Traffic' },
];

export const TRIAL_OPTIONS: { value: Trial; label: string }[] = [
  { value: '1', label: 'Trial 1' },
  { value: '2', label: 'Trial 2' },
  { value: '3', label: 'Trial 3' },
];

export type IMUSample = {
  t: number; // ms since recording start
  ax: number;
  ay: number;
  az: number;
  gY: number;
  gP: number;
  gR: number;
};

export type RecordingResult = {
  audioBlob: Blob;
  imuSamples: IMUSample[];
  durationMs: number;
};

export type RecordingLabel = {
  subjectId: string;
  sound: Sound;
  movement: Movement;
  noise: Noise;
  trial: Trial;
};

export type PermissionStatus = {
  audio: 'unknown' | 'granted' | 'denied' | 'unavailable';
  imu: 'unknown' | 'granted' | 'denied' | 'unavailable';
};

export type AppState =
  | { phase: 'idle' }
  | { phase: 'requesting-permissions' }
  | { phase: 'ready'; permissions: PermissionStatus }
  | { phase: 'recording'; startTime: number; permissions: PermissionStatus }
  | { phase: 'stopped'; result: RecordingResult; permissions: PermissionStatus }
  | { phase: 'labeling'; result: RecordingResult };

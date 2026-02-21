import { strToU8, zipSync } from 'fflate';
import type { RecordingLabel } from '../types.ts';

function mimeToExt(mimeType: string): string {
  if (mimeType.includes('mp4')) return '.mp4';
  if (mimeType.includes('ogg')) return '.ogg';
  return '.webm';
}

export function buildZip(
  audioBlob: Blob,
  durationMs: number,
  mimeType: string,
  label: RecordingLabel,
): Promise<{ zipBytes: Uint8Array; filename: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const audioBuffer = new Uint8Array(reader.result as ArrayBuffer);
        const ext = mimeToExt(mimeType);

        const groundTruth = JSON.stringify({ start_times: [], end_times: [] }, null, 2);

        const metadata = JSON.stringify(
          {
            subject_id: label.subjectId,
            sound: label.sound,
            movement: label.movement,
            noise: label.noise,
            trial: label.trial,
            duration_seconds: durationMs / 1000,
            audio_mime_type: mimeType,
            device: navigator.userAgent,
            recorded_at: new Date().toISOString(),
          },
          null,
          2,
        );

        const zipData = zipSync(
          {
            [`outward_facing_mic${ext}`]: [audioBuffer, { level: 0 }],
            'ground_truth.json': [strToU8(groundTruth), { level: 6 }],
            'metadata.json': [strToU8(metadata), { level: 6 }],
          },
          { level: 0 },
        );

        const { subjectId, trial, movement, noise, sound } = label;
        const filename = `${subjectId}_trial${trial}_${movement}_noise-${noise}_${sound}.zip`;

        resolve({ zipBytes: zipData, filename });
      } catch (err) {
        reject(err);
      }
    };

    reader.onerror = () => reject(new Error('Failed to read audio blob'));
    reader.readAsArrayBuffer(audioBlob);
  });
}

export function downloadZip(zipBytes: Uint8Array, filename: string): void {
  const blob = new Blob([zipBytes.buffer.slice(0) as ArrayBuffer], { type: 'application/zip' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

import { useState } from 'react';
import type {
  RecordingResult,
  RecordingLabel,
  Sound,
  Movement,
  Noise,
  Trial,
} from '../types.ts';
import {
  SOUND_OPTIONS,
  MOVEMENT_OPTIONS,
  NOISE_OPTIONS,
  TRIAL_OPTIONS,
} from '../types.ts';
import { buildZip, downloadZip } from '../utils/buildZip.ts';

type Props = {
  result: RecordingResult;
  mimeType: string;
  onReset: () => void;
};

function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T | '';
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="space-y-1">
      <label className="block text-xs font-semibold uppercase tracking-wider text-gray-500">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="min-h-12 w-full rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-white focus:border-blue-500 focus:outline-none"
      >
        <option value="">Select…</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function LabelingScreen({ result, mimeType, onReset }: Props) {
  const [subjectId, setSubjectId] = useState('');
  const [sound, setSound] = useState<Sound | ''>('');
  const [movement, setMovement] = useState<Movement | ''>('');
  const [noise, setNoise] = useState<Noise | ''>('');
  const [trial, setTrial] = useState<Trial | ''>('');
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const isComplete =
    subjectId.trim() !== '' &&
    sound !== '' &&
    movement !== '' &&
    noise !== '' &&
    trial !== '';

  const handleDownload = async () => {
    if (!isComplete) return;

    const label: RecordingLabel = {
      subjectId: subjectId.trim(),
      sound: sound as Sound,
      movement: movement as Movement,
      noise: noise as Noise,
      trial: trial as Trial,
    };

    setError(null);
    setDownloading(true);

    try {
      const { zipBytes, filename } = await buildZip(
        result.audioBlob,
        result.durationMs,
        mimeType,
        label,
      );
      downloadZip(zipBytes, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to build ZIP');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold">Label Recording</h2>
          <p className="mt-1 text-sm text-gray-400">Fill in details before downloading</p>
        </div>

        <div className="space-y-1">
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-500">
            Subject ID
          </label>
          <input
            type="text"
            value={subjectId}
            onChange={(e) => setSubjectId(e.target.value)}
            placeholder="e.g. 14287"
            className="min-h-12 w-full rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-white placeholder-gray-600 focus:border-blue-500 focus:outline-none"
          />
        </div>

        <SelectField label="Sound Type" value={sound} options={SOUND_OPTIONS} onChange={setSound} />
        <SelectField
          label="Movement"
          value={movement}
          options={MOVEMENT_OPTIONS}
          onChange={setMovement}
        />
        <SelectField
          label="Background Noise"
          value={noise}
          options={NOISE_OPTIONS}
          onChange={setNoise}
        />
        <SelectField label="Trial" value={trial} options={TRIAL_OPTIONS} onChange={setTrial} />

        {error && (
          <p className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            Error: {error}
          </p>
        )}

        <button
          onClick={() => void handleDownload()}
          disabled={!isComplete || downloading}
          className="min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {downloading ? 'Building ZIP…' : 'Download ZIP'}
        </button>

        <button
          onClick={onReset}
          className="min-h-12 w-full rounded-xl border border-gray-700 px-6 py-3 text-base font-semibold text-gray-400 transition hover:border-gray-500 hover:text-white active:scale-95"
        >
          Record Another
        </button>
      </div>
    </div>
  );
}

import { useState, useCallback } from 'react';
import type { AppState, PermissionStatus, PredictionResult } from './types.ts';
import { useAudioRecorder } from './hooks/useAudioRecorder.ts';
import { PermissionScreen } from './components/PermissionScreen.tsx';
import { RecordingScreen } from './components/RecordingScreen.tsx';
import { LabelingScreen } from './components/LabelingScreen.tsx';
import { PredictionScreen } from './components/PredictionScreen.tsx';

export default function App() {
  const [state, setState] = useState<AppState>({ phase: 'idle' });
  const [mimeType, setMimeType] = useState('');

  const audio = useAudioRecorder();

  const handleStart = useCallback(() => {
    setState({ phase: 'requesting-permissions' });

    void (async () => {
      const audioResult = await audio.requestPermission();

      const permissions: PermissionStatus = {
        audio: audioResult,
      };

      setState({ phase: 'ready', permissions });
    })();
  }, [audio]);

  const handleBeginRecording = useCallback(() => {
    if (state.phase !== 'ready') return;
    const { permissions } = state;

    const startTime = performance.now();
    audio.startRecording();

    setState({ phase: 'recording', startTime, permissions });
  }, [state, audio]);

  const handleStop = useCallback(() => {
    if (state.phase !== 'recording') return;
    const { startTime, permissions } = state;

    void (async () => {
      const durationMs = performance.now() - startTime;
      const audioBlob = await audio.stopRecording();

      setMimeType(audio.getMimeType());

      setState({
        phase: 'stopped',
        result: { audioBlob, durationMs },
        permissions,
      });
    })();
  }, [state, audio]);

  const handlePredict = useCallback(() => {
    if (state.phase !== 'stopped') return;
    const { result } = state;

    setState({ phase: 'predicting', result });

    void (async () => {
      try {
        const formData = new FormData();
        formData.append('audio', result.audioBlob, `recording${audio.getMimeType().includes('mp4') ? '.mp4' : audio.getMimeType().includes('ogg') ? '.ogg' : '.webm'}`);

        const response = await fetch('/api/predict', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const prediction = (await response.json()) as PredictionResult;

        setState({ phase: 'results', result, prediction });
      } catch (err) {
        // On error go back to stopped so user can retry
        setState({ phase: 'stopped', result, permissions: { audio: 'granted' } });
      }
    })();
  }, [state, audio]);

  const handleLabel = useCallback(() => {
    if (state.phase !== 'results') return;
    setState({ phase: 'labeling', result: state.result, prediction: state.prediction });
  }, [state]);

  const handleReset = useCallback(() => {
    setState({ phase: 'idle' });
  }, []);

  if (state.phase === 'idle') {
    return (
      <PermissionScreen
        phase="idle"
        onStart={handleStart}
        onBeginRecording={() => {}}
      />
    );
  }

  if (state.phase === 'requesting-permissions') {
    return (
      <PermissionScreen
        phase="requesting-permissions"
        onStart={handleStart}
        onBeginRecording={() => {}}
      />
    );
  }

  if (state.phase === 'ready') {
    return (
      <PermissionScreen
        phase="ready"
        permissions={state.permissions}
        onStart={handleStart}
        onBeginRecording={handleBeginRecording}
      />
    );
  }

  if (state.phase === 'recording') {
    return (
      <RecordingScreen
        phase="recording"
        startTime={state.startTime}
        onStop={handleStop}
        onPredict={() => {}}
      />
    );
  }

  if (state.phase === 'stopped') {
    return (
      <RecordingScreen
        phase="stopped"
        startTime={0}
        durationMs={state.result.durationMs}
        onStop={() => {}}
        onPredict={handlePredict}
      />
    );
  }

  if (state.phase === 'predicting') {
    return (
      <PredictionScreen
        phase="predicting"
        durationMs={state.result.durationMs}
        onReset={handleReset}
        onLabel={() => {}}
      />
    );
  }

  if (state.phase === 'results') {
    return (
      <PredictionScreen
        phase="results"
        durationMs={state.result.durationMs}
        prediction={state.prediction}
        onReset={handleReset}
        onLabel={handleLabel}
      />
    );
  }

  if (state.phase === 'labeling') {
    return (
      <LabelingScreen
        result={state.result}
        prediction={state.prediction}
        mimeType={mimeType}
        onReset={handleReset}
      />
    );
  }

  return null;
}

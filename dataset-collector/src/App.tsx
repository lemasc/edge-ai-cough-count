import { useState, useCallback } from 'react';
import type { AppState, PermissionStatus } from './types.ts';
import { useAudioRecorder } from './hooks/useAudioRecorder.ts';
import { PermissionScreen } from './components/PermissionScreen.tsx';
import { RecordingScreen } from './components/RecordingScreen.tsx';
import { LabelingScreen } from './components/LabelingScreen.tsx';

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
        result: { audioBlob, imuSamples, durationMs },
        permissions,
      });
    })();
  }, [state, audio, imu]);

  const handleLabel = useCallback(() => {
    if (state.phase !== 'stopped') return;
    setState({ phase: 'labeling', result: state.result });
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
        onLabel={() => {}}
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
        onLabel={handleLabel}
      />
    );
  }

  if (state.phase === 'labeling') {
    return (
      <LabelingScreen
        result={state.result}
        mimeType={mimeType}
        onReset={handleReset}
      />
    );
  }

  return null;
}

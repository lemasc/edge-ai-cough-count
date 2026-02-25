import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router';
import type { PermissionStatus } from '~/types';
import { useAudioRecorder } from '~/hooks/useAudioRecorder';
import { PermissionScreen } from '~/components/PermissionScreen';

type Phase = 'idle' | 'requesting-permissions' | 'ready';

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [permissions, setPermissions] = useState<PermissionStatus>({ audio: 'unknown' });

  const audio = useAudioRecorder();
  const navigate = useNavigate();

  const handleStart = useCallback(() => {
    setPhase('requesting-permissions');

    void (async () => {
      const audioResult = await audio.requestPermission();

      const perms: PermissionStatus = { audio: audioResult };
      setPermissions(perms);

      if (audioResult === 'granted') {
        void navigate('/record');
      } else {
        setPhase('ready');
      }
    })();
  }, [audio, navigate]);

  const handleBeginRecording = useCallback(() => {
    void navigate('/record');
  }, [navigate]);

  return (
    <PermissionScreen
      phase={phase}
      permissions={phase !== 'idle' ? permissions : undefined}
      onStart={handleStart}
      onBeginRecording={handleBeginRecording}
    />
  );
}

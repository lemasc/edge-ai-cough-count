import { useRef, useState, useCallback } from 'react';
import type { IMUSample } from '../types.ts';

type IMUPermission = 'granted' | 'denied' | 'unavailable';

export function useIMURecorder() {
  const samplesRef = useRef<IMUSample[]>([]);
  const handlerRef = useRef<((e: DeviceMotionEvent) => void) | null>(null);
  const [sampleCount, setSampleCount] = useState(0);

  const requestPermission = useCallback(async (): Promise<IMUPermission> => {
    if (typeof DeviceMotionEvent === 'undefined') {
      return 'unavailable';
    }

    // iOS 13+ requires explicit permission
    if (
      typeof (DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> })
        .requestPermission === 'function'
    ) {
      try {
        const result = await (
          DeviceMotionEvent as unknown as { requestPermission: () => Promise<string> }
        ).requestPermission();
        return result === 'granted' ? 'granted' : 'denied';
      } catch {
        return 'denied';
      }
    }

    // Android / desktop — permission not required but check availability
    return new Promise((resolve) => {
      let resolved = false;

      const testHandler = (e: DeviceMotionEvent) => {
        window.removeEventListener('devicemotion', testHandler);
        if (!resolved) {
          resolved = true;
          // If we get an event with at least some data it's available
          if (e.acceleration !== null || e.rotationRate !== null) {
            resolve('granted');
          } else {
            resolve('unavailable');
          }
        }
      };

      window.addEventListener('devicemotion', testHandler);

      // Timeout: if no event fires within 500 ms, declare unavailable
      setTimeout(() => {
        if (!resolved) {
          resolved = true;
          window.removeEventListener('devicemotion', testHandler);
          resolve('unavailable');
        }
      }, 500);
    });
  }, []);

  const startRecording = useCallback((startTime: number) => {
    samplesRef.current = [];
    let localCount = 0;

    const handler = (e: DeviceMotionEvent) => {
      const t = performance.now() - startTime;
      const sample: IMUSample = {
        t,
        ax: e.accelerationIncludingGravity?.x ?? 0,
        ay: e.accelerationIncludingGravity?.y ?? 0,
        az: e.accelerationIncludingGravity?.z ?? 0,
        gY: e.rotationRate?.beta ?? 0,   // beta = rotation around inferior axis = Yaw
        gP: e.rotationRate?.alpha ?? 0,  // alpha = rotation around lateral axis = Pitch
        gR: e.rotationRate?.gamma ?? 0,  // gamma = Roll (inferred, unconfirmed)
      };
      samplesRef.current.push(sample);

      localCount++;
      if (localCount % 10 === 0) {
        setSampleCount(localCount);
      }
    };

    handlerRef.current = handler;
    window.addEventListener('devicemotion', handler);
  }, []);

  const stopRecording = useCallback((): IMUSample[] => {
    if (handlerRef.current) {
      window.removeEventListener('devicemotion', handlerRef.current);
      handlerRef.current = null;
    }
    const samples = [...samplesRef.current];
    samplesRef.current = [];
    setSampleCount(0);
    return samples;
  }, []);

  return { requestPermission, startRecording, stopRecording, sampleCount };
}

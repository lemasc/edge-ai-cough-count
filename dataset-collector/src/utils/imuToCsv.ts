import type { IMUSample } from '../types.ts';

export function imuToCsv(samples: IMUSample[]): string {
  const header = 'timestamp,accel_x,accel_y,accel_z,gyro_Y,gyro_P,gyro_R';

  if (samples.length === 0) {
    return header + '\n';
  }

  const rows = samples.map((s) => {
    const t = (s.t / 1000).toFixed(3);
    const ax = s.ax.toFixed(4);
    const ay = s.ay.toFixed(4);
    const az = s.az.toFixed(4);
    const gY = s.gY.toFixed(4);
    const gP = s.gP.toFixed(4);
    const gR = s.gR.toFixed(4);
    return `${t},${ax},${ay},${az},${gY},${gP},${gR}`;
  });

  return header + '\n' + rows.join('\n') + '\n';
}

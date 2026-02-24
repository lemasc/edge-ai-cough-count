import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js";

type WaveformPlayerProps = {
  audioBlob: Blob;
  startTimes: number[];
  endTimes: number[];
};

export function WaveformPlayer({
  audioBlob,
  startTimes,
  endTimes,
}: WaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const url = URL.createObjectURL(audioBlob);
    const wsRegions = RegionsPlugin.create();
    regionsRef.current = wsRegions;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      height: 80,
      waveColor: "#4B5563",
      progressColor: "#3B82F6",
      cursorColor: "#60A5FA",
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      interact: true,
      normalize: true,
      fillParent: true,
      plugins: [wsRegions],
    });

    wavesurferRef.current = ws;

    ws.on("ready", () => {
      setIsReady(true);
      for (let i = 0; i < startTimes.length; i++) {
        wsRegions.addRegion({
          start: startTimes[i],
          end: endTimes[i],
          color: "rgba(239, 68, 68, 0.25)",
          drag: false,
          resize: false,
        });
      }
    });

    let destroyed = false;

    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));
    ws.on("error", () => {
      if (!destroyed) setError("Could not decode audio");
    });

    ws.load(url);

    return () => {
      destroyed = true;
      ws.destroy();
      URL.revokeObjectURL(url);
      setIsReady(false);
      setIsPlaying(false);
      setError(null);
    };
  }, [audioBlob, startTimes, endTimes]);

  const handlePlayPause = () => {
    wavesurferRef.current?.playPause();
  };

  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
        Recording
      </p>
      <div className="relative rounded-xl bg-gray-900 px-1 py-2">
        {!isReady && !error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-gray-700 border-t-blue-500" />
          </div>
        )}
        {error && (
          <p className="py-4 text-center text-sm text-red-400">{error}</p>
        )}
        <div ref={containerRef} className={error ? "hidden" : ""} />
      </div>
      <button
        onClick={handlePlayPause}
        disabled={!isReady}
        className="flex min-h-10 w-full items-center justify-center gap-2 rounded-xl border border-gray-700 px-6 py-2 text-base font-semibold text-gray-300 transition hover:border-gray-500 hover:text-white active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {isPlaying ? (
          <>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="h-5 w-5"
            >
              <path
                fillRule="evenodd"
                d="M6.75 5.25a.75.75 0 0 1 .75-.75H9a.75.75 0 0 1 .75.75v13.5a.75.75 0 0 1-.75.75H7.5a.75.75 0 0 1-.75-.75V5.25Zm7.5 0A.75.75 0 0 1 15 4.5h1.5a.75.75 0 0 1 .75.75v13.5a.75.75 0 0 1-.75.75H15a.75.75 0 0 1-.75-.75V5.25Z"
                clipRule="evenodd"
              />
            </svg>
            Pause
          </>
        ) : (
          <>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="h-5 w-5"
            >
              <path
                fillRule="evenodd"
                d="M4.5 5.653c0-1.427 1.529-2.33 2.779-1.643l11.54 6.347c1.295.712 1.295 2.573 0 3.286L7.28 19.99c-1.25.687-2.779-.217-2.779-1.643V5.653Z"
                clipRule="evenodd"
              />
            </svg>
            Play
          </>
        )}
      </button>
    </div>
  );
}

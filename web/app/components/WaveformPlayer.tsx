import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { formatSeconds } from "../utils/formatTime";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js";
import { PauseIcon, PlayIcon } from "lucide-react";

export interface WaveformPlayerHandle {
  seekTo: (timeSecs: number) => void;
}

type WaveformPlayerProps = {
  src: string;
  startTimes: number[];
  endTimes: number[];
};

const regionColors = [
  "rgba(239, 68, 68, 0.3)", // red
  "rgba(234, 179, 8, 0.3)", // amber
];

export const WaveformPlayer = forwardRef<
  WaveformPlayerHandle,
  WaveformPlayerProps
>(function WaveformPlayer({ src, startTimes, endTimes }, ref) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(0);
  const [minZoom, setMinZoom] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useImperativeHandle(ref, () => ({
    seekTo: (timeSecs: number) => {
      const ws = wavesurferRef.current;
      if (!ws) return;
      const dur = ws.getDuration();
      if (dur <= 0) return;
      ws.seekTo(timeSecs / dur);
    },
  }));

  const HEIGHT = 128;

  useEffect(() => {
    if (!containerRef.current) return;

    const url = src;
    const wsRegions = RegionsPlugin.create();
    regionsRef.current = wsRegions;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      height: HEIGHT,
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

    ws.on("ready", () => {
      const dur = ws.getDuration();
      const base = Math.ceil((containerRef.current?.clientWidth ?? 300) / dur);
      setMinZoom(base);
      setZoom(base);
      setDuration(dur);
      setIsReady(true);
    });

    let destroyed = false;

    ws.on("timeupdate", (t) => setCurrentTime(t));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));
    ws.on("error", () => {
      if (!destroyed) setError("Could not decode audio");
    });

    ws.load(url);

    wavesurferRef.current = ws;

    return () => {
      destroyed = true;
      ws.destroy();
      wavesurferRef.current = null;
      regionsRef.current = null;
      setIsReady(false);
      setIsPlaying(false);
      setError(null);
      setMinZoom(0);
      setZoom(0);
      setCurrentTime(0);
      setDuration(0);
    };
  }, [src]);

  useEffect(() => {
    if (!isReady) return;
    const regions = regionsRef.current;
    if (!regions) return;

    const existing = regions.getRegions();

    // Update existing regions in-place (no DOM removal, no flash)
    for (let i = 0; i < Math.min(startTimes.length, existing.length); i++) {
      existing[i].setOptions({
        start: startTimes[i],
        end: endTimes[i],
        color: regionColors[i % 2],
      });
    }
    // Add any new regions beyond what already exists
    for (let i = existing.length; i < startTimes.length; i++) {
      regions.addRegion({
        start: startTimes[i],
        end: endTimes[i],
        color: regionColors[i % 2],
        drag: false,
        resize: false,
      });
    }
    // Remove surplus regions if count shrank
    for (let i = startTimes.length; i < existing.length; i++) {
      existing[i].remove();
    }
  }, [startTimes, endTimes, isReady]);

  useEffect(() => {
    if (isReady) {
      wavesurferRef.current?.zoom(zoom);
    }
  }, [zoom, isReady]);

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
        <div
          ref={containerRef}
          className={error ? "hidden" : ""}
          style={{ minHeight: HEIGHT }}
        />
      </div>
      <div className="flex justify-end">
        <span className="font-mono text-xs tabular-nums text-gray-500">
          {isReady ? formatSeconds(currentTime) : "0:00"}
          {" / "}
          {isReady ? formatSeconds(duration) : "0:00"}
        </span>
      </div>
      <div className="flex items-center gap-3">
        <span className="shrink-0 text-xs text-gray-500">Zoom</span>
        <input
          type="range"
          min={minZoom}
          max={minZoom * 10}
          step={1}
          value={zoom}
          onChange={(e) => setZoom(Number(e.target.value))}
          disabled={!isReady}
          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500 disabled:opacity-40"
        />
        <span className="w-10 shrink-0 text-right text-xs tabular-nums text-gray-500">
          {minZoom > 0 ? (zoom / minZoom).toFixed(1) : "1.0"}×
        </span>
      </div>
      <button
        onClick={handlePlayPause}
        disabled={!isReady}
        className="flex min-h-10 w-full items-center justify-center gap-2 rounded-xl border border-gray-700 px-6 py-2 text-base font-semibold text-gray-300 transition hover:border-gray-500 hover:text-white active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {isPlaying ? (
          <>
            <PauseIcon className="h-5 w-5" />
            Pause
          </>
        ) : (
          <>
            <PlayIcon className="h-5 w-5" />
            Play
          </>
        )}
      </button>
    </div>
  );
});
WaveformPlayer.displayName = "WaveformPlayer";

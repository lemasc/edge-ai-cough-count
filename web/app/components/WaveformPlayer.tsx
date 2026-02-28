import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { formatSeconds } from "../utils/formatTime";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, {
  type RegionParams,
} from "wavesurfer.js/plugins/regions";
import { PauseIcon, PlayIcon } from "lucide-react";
import { useTrackedRegions } from "~/hooks/useTrackedRegions";
import posthog from "posthog-js";

export interface WaveformPlayerHandle {
  seekTo: (timeSecs: number) => void;
  playFrom: (timeSecs: number) => void;
  getCurrentTime: () => number;
}

type WaveformPlayerProps = {
  src: string;
  startTimes: number[];
  endTimes: number[];
  height?: number;
  annotating?: boolean;
  pendingAnnotationTime?: number | null;
  markerTimes?: number[];
};

const regionColors = [
  "rgba(239, 68, 68, 0.3)", // red
  "rgba(234, 179, 8, 0.3)", // amber
];

export const WaveformPlayer = forwardRef<
  WaveformPlayerHandle,
  WaveformPlayerProps
>(function WaveformPlayer(
  {
    src,
    startTimes,
    endTimes,
    height = 128,
    annotating = false,
    pendingAnnotationTime = null,
    markerTimes = [],
  },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<RegionsPlugin | null>(null);
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
    playFrom: (timeSecs: number) => {
      const ws = wavesurferRef.current;
      if (!ws) return;
      const dur = ws.getDuration();
      if (dur <= 0) return;
      ws.seekTo(timeSecs / dur);
      ws.play();
    },
    getCurrentTime: () => wavesurferRef.current?.getCurrentTime() ?? 0,
  }));

  useEffect(() => {
    if (!containerRef.current) return;

    const url = src;
    const wsRegions = RegionsPlugin.create();
    regionsPluginRef.current = wsRegions;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      height,
      waveColor: "#4B5563",
      progressColor: "#3B82F6",
      cursorColor: "#60A5FA",
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      interact: true,
      normalize: false,
      fillParent: true,
      dragToSeek: false,
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
    ws.on("error", (err) => {
      if (!destroyed) {
        setError("Could not decode audio");
        posthog.captureException(err instanceof Error ? err : new Error(String(err)));
      }
    });

    ws.load(url);

    wavesurferRef.current = ws;

    return () => {
      destroyed = true;
      ws.destroy();
      wavesurferRef.current = null;
      regionsPluginRef.current = null;
      setIsReady(false);
      setIsPlaying(false);
      setError(null);
      setMinZoom(0);
      setZoom(0);
      setCurrentTime(0);
      setDuration(0);
    };
  }, [src, height]);

  const coughRegions: RegionParams[] = useMemo(
    () =>
      startTimes.map((start, i) => ({
        start,
        end: endTimes[i],
        color: regionColors[i % 2],
        drag: false,
        resize: false,
      })),
    [startTimes, endTimes],
  );

  useTrackedRegions({
    regions: coughRegions,
    regionsPlugin: regionsPluginRef.current,
    enabled: isReady,
  });

  const markerRegions: RegionParams[] = useMemo(
    () =>
      markerTimes.map((start) => ({
        start,
        color: "rgba(251, 191, 36, 0.8)",
        drag: false,
        resize: false,
      })),
    [markerTimes],
  );

  useTrackedRegions({
    regions: markerRegions,
    regionsPlugin: regionsPluginRef.current,
    enabled: isReady,
  });

  const pendingMarker: RegionParams[] = useMemo(
    () =>
      pendingAnnotationTime != null
        ? [
            {
              start: pendingAnnotationTime,
              color: "rgba(251, 191, 36, 1)",
              drag: false,
              resize: false,
            },
          ]
        : [],
    [pendingAnnotationTime],
  );

  useTrackedRegions({
    regions: pendingMarker,
    regionsPlugin: regionsPluginRef.current,
    enabled: isReady,
  });

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
      <div
        className={`relative rounded-xl bg-gray-900 px-1 py-2 transition ${annotating ? "ring-2 ring-amber-500" : ""}`}
      >
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
          className={error ? "hidden" : "select-none"}
          style={{ minHeight: height }}
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

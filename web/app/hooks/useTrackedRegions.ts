import { useEffect, useRef } from "react";
import type { Region, RegionParams } from "wavesurfer.js/plugins/regions";
import type RegionsPlugin from "wavesurfer.js/plugins/regions";

export function useTrackedRegions({
  regions,
  regionsPlugin,
  enabled = true,
  debugValue,
}: {
  regions: RegionParams[];
  regionsPlugin: RegionsPlugin | null;
  enabled?: boolean;
  debugValue?: string;
}) {
  const tracked = useRef<Region[]>([]);

  useEffect(() => {
    if (!enabled) return;
    if (!regionsPlugin) return;

    if (debugValue) {
      console.log(
        `[${debugValue}] Updating regions`,
        regions,
        tracked.current.map((r) => ({ start: r.start, end: r.end })),
      );
    }

    // Update existing regions in-place (no DOM removal, no flash)
    for (let i = 0; i < Math.min(regions.length, tracked.current.length); i++) {
      tracked.current[i].setOptions(regions[i]);
    }
    // Add any new regions beyond what already exists
    for (let i = tracked.current.length; i < regions.length; i++) {
      tracked.current.push(regionsPlugin.addRegion(regions[i]));
    }

    // Remove surplus regions if count shrank
    const surplus = tracked.current.splice(regions.length);
    for (const r of surplus) r.remove();
  }, [regions, enabled, debugValue]);

  return null;
}

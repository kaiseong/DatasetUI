import { describe, expect, it } from "vitest";

import { currentValue, groupByScale, polyline } from "../../src/renderer/parity/charts.js";
import { luminance, normalizeDepth, VIRIDIS_LUT, viridis } from "../../src/renderer/parity/color.js";
import {
  adjacentEpisode, episodePage, EPISODES_PER_PAGE, pageEpisodes, PlayheadController,
  READY_TIMEOUT_MS, segmentTime, SEGMENT_END_TOLERANCE_SECONDS, shouldSynchronize,
  SYNC_DRIFT_SECONDS, toggleCamera,
} from "../../src/renderer/parity/playback.js";

describe("episode navigation", () => {
  it("episode-navigation-pagination", () => {
    expect(EPISODES_PER_PAGE).toBe(100);
    expect(episodePage(0)).toBe(1); expect(episodePage(100)).toBe(2);
    expect(pageEpisodes(205, 3)).toEqual([200, 201, 202, 203, 204]);
    expect(adjacentEpisode(0, 205, "ArrowUp")).toBe(0);
    expect(adjacentEpisode(204, 205, "ArrowDown")).toBe(204);
  });
});

describe("shared playback", () => {
  it("synchronized-multi-camera-video", () => {
    expect(SYNC_DRIFT_SECONDS).toBe(0.2); expect(READY_TIMEOUT_MS).toBe(10_000);
    expect(SEGMENT_END_TOLERANCE_SECONDS).toBe(0.05);
    expect(shouldSynchronize(1, 1.19)).toBe(false); expect(shouldSynchronize(1, 1.21)).toBe(true);
  });

  it("segmented-video-looping", () => {
    expect(segmentTime(0, 1, 3)).toBe(1);
    expect(segmentTime(1.96, 1, 3)).toBe(1);
  });

  it("camera-controls", () => {
    const controller = new PlayheadController(); controller.setPlaying(true); controller.seek(8);
    controller.beginDrag(); expect(controller.playing).toBe(false); controller.endDrag(); expect(controller.playing).toBe(true);
    controller.skip(5, 10); expect(controller.time).toBe(10);
    expect([...toggleCamera(new Set(["front"]), "front")]).toEqual(["front"]);
    expect([...toggleCamera(new Set(["front", "wrist"]), "front")]).toEqual(["wrist"]);
  });
});

describe("image/depth and charts", () => {
  it("depth-grayscale-colormaps", () => {
    expect(luminance(255, 255, 255)).toBe(255); expect(VIRIDIS_LUT).toHaveLength(256);
    expect(VIRIDIS_LUT[0]).toEqual([68, 1, 84]); expect(VIRIDIS_LUT[128]).toEqual([33, 145, 140]);
    expect(VIRIDIS_LUT[255]).toEqual([253, 231, 37]);
    expect(normalizeDepth(-1, 0, 10)).toBe(0); expect(normalizeDepth(99, 0, 10)).toBe(1);
    expect(viridis(0, 0, 10)).toEqual(VIRIDIS_LUT[0]);
  });

  it("synchronized-action-state-charts", () => {
    const series = Array.from({ length: 7 }, (_, i) => ({ name: String(i), values: [10 ** i] }));
    expect(groupByScale(series).every((group) => group.series.length <= 6 && Math.log10(group.max / group.min) <= 2)).toBe(true);
    expect(currentValue({ name: "a", values: [1, 2, 3] }, [0, 1, 2], 0.4)).toBe(2);
    expect(polyline([1, 2], 10, 10, 0, 10)).toBe("0,9 10,8");
  });
});

import { afterEach, describe, expect, it } from "vitest";

import { episodeResponse, parityApi, replayResponse } from "../../src/renderer/parity/api.js";
import { depthPixelValue, depthQuantiles } from "../../src/renderer/parity/color.js";
import { mediaSourceAt, playheadFrame } from "../../src/renderer/parity/playback.js";
import { mappedPose, mappedUnit } from "../../src/renderer/parity/replay.js";

afterEach(() => {
  delete (globalThis as unknown as { window?: unknown }).window;
});

describe("episode media adapter", () => {
  it("preserves segments, quantiles and all embedded/path image frames", () => {
    const mediaCalls: unknown[] = [];
    (globalThis as unknown as { window: unknown }).window = { datasetEditor: {
      mediaUrl: (params: unknown) => { mediaCalls.push(params); return `safe://${(params as { relative_path: string }).relative_path}`; },
    } };
    const episode = episodeResponse({
      episode_index: 2, episode_count: 4, duration: 1, fps: 2, timestamps: [0, 0.5],
      action: [], state: [], progress: [],
      media: [{ camera: "depth", kind: "depth", relative_path: "images/depth/0.png", from: 2, to: 3,
        q01: 10, q99: 200, shape: [2, 2, 1], grayscale: true, is_depth_map: true, raw_band: [0.15, 0.85],
        depth_encoding: { depth_min: 0, depth_max: 2, shift: 0.1, use_log: false, depth_unit: "m", unit_to_metres: 1 }, frames: [
          { frame: 0, data_url: "data:image/png;base64,AAAA" },
          { frame: 1, relative_path: "images/depth/1.png" },
        ] }],
    }, "project");
    expect(episode.media[0]).toMatchObject({
      camera: "depth", kind: "depth", url: "safe://images/depth/0.png", relative_path: "images/depth/0.png",
      from: 2, to: 3, q01: 10, q99: 200, shape: [2, 2, 1], grayscale: true, is_depth_map: true,
      raw_band: [0.15, 0.85], depth_encoding: { depth_min: 0, depth_max: 2, shift: 0.1, use_log: false, depth_unit: "m", unit_to_metres: 1 },
      frames: [
        { frame: 0, url: "data:image/png;base64,AAAA", data_url: "data:image/png;base64,AAAA" },
        { frame: 1, url: "safe://images/depth/1.png", relative_path: "images/depth/1.png" },
      ],
    });
    expect(mediaCalls).toEqual([
      { project_id: "project", relative_path: "images/depth/1.png" },
      { project_id: "project", relative_path: "images/depth/0.png" },
    ]);
  });

  it("maps video gallery paths to safe media URLs while retaining seek timestamps", async () => {
    (globalThis as unknown as { window: unknown }).window = { datasetEditor: {
      datasetAnalytics: async () => ({ statistics: [], histograms: [], episodes: [{ episode_index: 0, duration: 1, movement: 0 }], variance: [], autocorrelation: [], suggested_chunk_size: null, velocity: [], jerk: [], speed_cv: { value: 0, verdict: "consistent", bins: [] }, alignment: [],
        quality: { gallery: [{ key: "front", total: 1, items: [{ episode_index: 0, first_timestamp: 2, last_timestamp: 2.95, relative_path: "videos/front.mp4" }] }] } }),
      mediaUrl: ({ relative_path }: { relative_path: string }) => `safe://${relative_path}`,
    } };
    await expect(parityApi.analytics("project")).resolves.toMatchObject({ quality: { gallery: [{ items: [{
      episode_index: 0, first_timestamp: 2, last_timestamp: 2.95, relative_path: "videos/front.mp4", url: "safe://videos/front.mp4",
    }] }] } });
  });

  it("selects image-sequence frames with the chart-compatible ceiling playhead", () => {
    const track = { camera: "front", kind: "image" as const, url: "zero", frames: [
      { frame: 0, url: "zero" }, { frame: 1, url: "one" }, { frame: 2, url: "two" },
    ] };
    expect(playheadFrame([0, 0.5, 1], 0.1)).toBe(1);
    expect(mediaSourceAt(track, [0, 0.5, 1], 0.1)).toBe("one");
    expect(mediaSourceAt(track, [0, 0.5, 1], 4)).toBe("two");
    expect(depthQuantiles(Array.from({ length: 101 }, (_, index) => index))).toEqual({ q01: 1, q99: 99 });
    expect(depthPixelValue(255, 255, 255, true)).toBe(1);
  });
});

describe("backend replay adapter", () => {
  it("sends frame values and names to replay.map and validates normalized positions", async () => {
    const calls: unknown[] = [];
    (globalThis as unknown as { window: unknown }).window = { datasetEditor: {
      replayMap: async (params: unknown) => { calls.push(params); return {
        supported: true, robot_type: "so101", family: "so", scale: 10,
        positions: { shoulder_joint: Math.PI, gripper: 0.022 }, trail: { seconds: 1, max_points: 300 },
      }; },
    } };
    await expect(parityApi.replayMap("project", 3, [180, 0.5], ["shoulder", "gripper"]))
      .resolves.toMatchObject({ supported: true, positions: { shoulder_joint: Math.PI, gripper: 0.022 } });
    expect(calls).toEqual([{ project_id: "project", episode_index: 3, values: [180, 0.5], joint_names: ["shoulder", "gripper"] }]);
  });

  it("keeps backend unsupported reasons and renders mapped units/pose", () => {
    expect(replayResponse({ supported: false, robot_type: "panda", reason: "not supported" }))
      .toEqual({ supported: false, robot_type: "panda", reason: "not supported" });
    expect(mappedUnit("left_finger_joint1")).toBe("m");
    expect(mappedUnit("left_shoulder_joint")).toBe("rad");
    expect(mappedPose({ shoulder: 0, elbow: Math.PI / 2 }, 10)).toHaveLength(3);
    expect(() => replayResponse({ supported: true, robot_type: "so100", positions: [] })).toThrow("positions");
  });
});

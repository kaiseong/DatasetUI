import { describe, expect, test } from "bun:test";
import {
  armDimensionStatus,
  augmentationSummary,
  DEFAULT_JOINT_RANGES_DEG,
  sampleEpisodes,
  validRanges,
} from "../joint-offset";

const NAMES = [
  ...Array.from({ length: 7 }, (_, i) => `right_arm_${i}`),
  ...Array.from({ length: 7 }, (_, i) => `left_arm_${i}`),
  "right_gripper_0",
  "left_gripper_0",
];
const info = (names: unknown) => ({
  features: {
    "observation.state": { names },
    action: { names },
  },
});

describe("joint offset augmentation", () => {
  test("defaults are the larger bound of each joint", () => {
    expect(DEFAULT_JOINT_RANGES_DEG).toEqual([
      0.3, 0.1, 1.0, 0.3, 1.0, 0.2, 1.5,
    ]);
    expect(validRanges(DEFAULT_JOINT_RANGES_DEG)).toBe(true);
    expect(validRanges([0.3, 0.1])).toBe(false);
    expect(validRanges([...DEFAULT_JOINT_RANGES_DEG.slice(1), 11])).toBe(false);
    expect(
      validRanges([...DEFAULT_JOINT_RANGES_DEG.slice(1), Number.NaN]),
    ).toBe(false);
  });

  test("sampling is reproducible, sized by percent and sorted", () => {
    const indices = Array.from({ length: 20 }, (_, i) => i);
    const first = sampleEpisodes(indices, 30, 7);
    expect(first).toHaveLength(6);
    expect(first).toEqual([...first].sort((a, b) => a - b));
    expect(new Set(first).size).toBe(6);
    expect(sampleEpisodes(indices, 30, 7)).toEqual(first);
    expect(sampleEpisodes(indices, 30, 8)).not.toEqual(first);
    expect(sampleEpisodes(indices, 1, 0)).toHaveLength(1);
    expect(sampleEpisodes(indices, 100, 0)).toEqual(indices);
    expect(sampleEpisodes([], 50, 0)).toEqual([]);
  });

  test("arm joint names are required in state and action", () => {
    expect(armDimensionStatus(info(NAMES))).toEqual({
      ready: true,
      missing: [],
    });
    expect(armDimensionStatus(info({ motors: NAMES })).ready).toBe(true);
    const generic = armDimensionStatus(info(["joint_0"]));
    expect(generic.ready).toBe(false);
    expect(generic.missing).toContain("action.left_arm_6");
    expect(armDimensionStatus(null).missing).toHaveLength(28);
  });

  test("summary counts added episodes and video growth", () => {
    expect(augmentationSummary(10, 3, 2)).toEqual({
      output: 16,
      added: 6,
      videoFactor: 1.6,
    });
  });
});

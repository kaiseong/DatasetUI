import { describe, expect, it } from "vitest";

import { appendTrail, isSupportedRobot, jointUnit, matchJoint, TRAIL_MAX_POINTS, TRAIL_SECONDS } from "../../src/renderer/parity/replay.js";

describe("robot replay", () => {
  it("supports the defined robots and rejects unknown ones", () => {
    for (const robot of ["SO100", "so101", "so_follower", "OpenArm", "Unitree_G1"]) expect(isSupportedRobot(robot)).toBe(true);
    expect(isSupportedRobot("panda")).toBe(false);
  });

  it("converts ticks, degrees and radians and matches suffixes", () => {
    expect(jointUnit(2048)).toBe(0); expect(jointUnit(180)).toBeCloseTo(Math.PI); expect(jointUnit(1)).toBe(1);
    const series = [{ name: "observation.state.shoulder", values: [1] }];
    expect(matchJoint("shoulder", series)?.name).toBe(series[0]?.name);
  });

  it("bounds the one-second trail to 300 points", () => {
    let trail: Array<{ time: number; x: number; y: number }> = [];
    for (let i = 0; i < 500; i++) trail = appendTrail(trail, { time: i / 400, x: i, y: i });
    expect(TRAIL_SECONDS).toBe(1); expect(TRAIL_MAX_POINTS).toBe(300); expect(trail).toHaveLength(300);
    expect(trail[0]?.time).toBeGreaterThanOrEqual(trail.at(-1)!.time - 1);
  });
});

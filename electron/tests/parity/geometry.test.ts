import { describe, expect, it } from "vitest";

import { canvasToNormalized, finishDraw, letterboxRect, moveBbox, normalizedToCanvas, round4 } from "../../src/renderer/annotations/geometry.js";

describe("annotation overlay geometry", () => {
  it("computes horizontal and vertical object-contain letterboxing", () => {
    expect(letterboxRect({ width: 400, height: 400 }, { width: 16, height: 9 })).toEqual({ left: 0, top: 87.5, width: 400, height: 225 });
    expect(letterboxRect({ width: 800, height: 200 }, { width: 4, height: 3 })).toEqual({ left: 266.6666666666667, top: 0, width: 266.66666666666663, height: 200 });
  });

  it("round-trips points, clamps letterbox clicks, and rounds to four decimals", () => {
    const rect = letterboxRect({ width: 400, height: 400 }, { width: 16, height: 9 });
    expect(canvasToNormalized({ x: 200, y: 200 }, rect)).toEqual({ x: 0.5, y: 0.5 });
    expect(canvasToNormalized({ x: -10, y: 10 }, rect)).toEqual({ x: 0, y: 0 });
    expect(normalizedToCanvas({ x: 0.5, y: 0.5 }, rect)).toEqual({ x: 200, y: 200 });
    expect(round4(0.123456)).toBe(0.1235);
  });

  it("creates points for <=4px and order-independent xyxy boxes for drags", () => {
    const rect = { left: 0, top: 0, width: 100, height: 100 };
    expect(finishDraw({ x: 10, y: 10 }, { x: 14, y: 10 }, rect)).toEqual({ kind: "keypoint", point: [0.14, 0.1] });
    expect(finishDraw({ x: 90, y: 80 }, { x: 10, y: 20 }, rect)).toEqual({ kind: "bbox", bbox: [0.1, 0.2, 0.9, 0.8] });
  });

  it("moves boxes while preserving their size and clamping to the image", () => {
    expect(moveBbox([0.2, 0.3, 0.5, 0.7], { x: 0.8, y: -0.8 })).toEqual([0.7, 0, 1, 0.4]);
  });
});

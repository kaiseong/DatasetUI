import { describe, expect, it } from "vitest";

import { ACTIVE_DELTA_RANGE_RATIO, ALIGNMENT_MAX_LAG, AUTOCORRELATION_MAX_LAG, DISCRETE_UNIQUE_LIMIT, evenlySpacedIndices, HEATMAP_BINS, MAX_ANALYSIS_EPISODES, MAX_FRAMES_PER_EPISODE, populationVariance, speedVerdict, suggestedChunk, VELOCITY_BINS } from "../../src/renderer/parity/insights.js";

describe("action insight reference constants", () => {
  it("cross-episode-variance-heatmap", () => {
    expect([MAX_ANALYSIS_EPISODES, MAX_FRAMES_PER_EPISODE, HEATMAP_BINS]).toEqual([120, 2500, 50]);
    expect(evenlySpacedIndices(5, 3)).toEqual([0, 2, 4]);
    expect(populationVariance([1, 2, 3])).toBeCloseTo(2 / 3);
  });

  it("action-autocorrelation-chunk", () => {
    expect(AUTOCORRELATION_MAX_LAG).toBe(100);
    expect(suggestedChunk([.9, .7, .49, .3])).toBe(3);
  });

  it("action-velocity-activity-discrete", () => {
    expect([ACTIVE_DELTA_RANGE_RATIO, DISCRETE_UNIQUE_LIMIT, VELOCITY_BINS]).toEqual([.001, 4, 30]);
  });

  it("jerky-episode-ranking", () => {
    const ranked = [{ name: "steady", score: .01 }, { name: "jerky", score: .7 }]
      .sort((a, b) => b.score - a.score);
    expect(ranked.map((item) => item.name)).toEqual(["jerky", "steady"]);
  });

  it("demonstrator-speed-cv", () => {
    expect(speedVerdict(.19)).toBe("consistent"); expect(speedVerdict(.2)).toBe("moderate"); expect(speedVerdict(.4)).toBe("high");
  });

  it("state-action-temporal-alignment", () => expect(ALIGNMENT_MAX_LAG).toBe(30));
});

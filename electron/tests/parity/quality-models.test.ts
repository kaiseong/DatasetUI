import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { parityApi } from "../../src/renderer/parity/api.js";
import { StatisticsPanel } from "../../src/renderer/parity/statistics.js";
import { deletePreview, evenlySample, filterEpisodes, FLAG_STORAGE_KEY, GALLERY_MAX_EPISODES, GALLERY_PAGE_SIZE, histogramBinCount, LAZY_ROOT_MARGIN, lowMovement, niceWidth, populationStats } from "../../src/renderer/parity/quality.js";
import type { AnalyticsData } from "../../src/renderer/parity/types.js";

describe("quality models", () => {
  it("statistics-histogram", async () => {
    const calls: unknown[] = [];
    (globalThis as unknown as { window: unknown }).window = { datasetEditor: {
      datasetSummary: async (params: unknown) => {
        calls.push(params);
        return {
          version: "v3.0", robot_type: "so101", fps: 10, total_episodes: 3,
          total_frames: 36_600, total_tasks: 2, cameras: ["observation.images.front"],
          features: [{ name: "observation.images.front", dtype: "image", shape: [480, 640, 3] }],
        };
      },
    } };
    const summary = await parityApi.summary("project-37");
    expect(calls).toEqual([{ project_id: "project-37" }]);
    expect(summary).toMatchObject({ total_duration: 3_660, cameras: [{ name: "observation.images.front", width: 640, height: 480 }] });

    const analytics = {
      statistics: [], histograms: [], episodes: [], variance: [], autocorrelation: [],
      suggested_chunk_size: null, velocity: [], jerk: [],
      speed_cv: { value: 0, verdict: "consistent", bins: [], lo: 0, hi: 0 }, alignment: [],
      episode_lengths: {
        shortest_episodes: [{ episode_index: 0, frames: 50, length_seconds: 5 }],
        longest_episodes: [{ episode_index: 2, frames: 90, length_seconds: 9 }],
        all_episode_lengths: [], mean_episode_length: 7, median_episode_length: 7,
        std_episode_length: 1.63,
        episode_length_histogram: [{ bin_label: "5–7", count: 1 }, { bin_label: "7–9", count: 2 }],
      },
    } satisfies AnalyticsData;
    const markup = renderToStaticMarkup(createElement(StatisticsPanel, { summary, analytics }));
    expect(markup).toContain("Dataset Statistics");
    expect(markup).toContain("so101"); expect(markup).toContain("v3.0"); expect(markup).toContain("Tasks");
    expect(markup).toContain("36,600"); expect(markup).toContain('data-seconds="3660"'); expect(markup).toContain("1h 1m");
    expect(markup).toContain("observation.images.front"); expect(markup).toContain("640×480");
    expect(markup).toContain("Mean"); expect(markup).toContain("Median"); expect(markup).toContain("Std Dev");
    expect(markup).toContain("Episode length distribution histogram"); expect(markup).toContain("5–7: 1 episodes");
    expect(renderToStaticMarkup(createElement(StatisticsPanel, { loading: true }))).toContain("Computing dataset statistics");
    (globalThis as unknown as { window: unknown }).window = { datasetEditor: { datasetSummary: async () => ({
      version: "v3.0", robot_type: null, fps: 10, total_episodes: 1, total_frames: 10, cameras: [], features: [],
    }) } };
    await expect(parityApi.summary("invalid-summary")).rejects.toThrow("counts");
    expect(populationStats([1, 2, 3])).toEqual({ mean: 2, median: 2, std: Math.sqrt(2 / 3) });
    expect(histogramBinCount(1)).toBe(10); expect(histogramBinCount(2 ** 60)).toBe(50); expect(niceWidth(2.2)).toBe(2.5);
  });

  it("first-last-frame-gallery", () => {
    expect(evenlySample([0, 1, 2, 3, 4], 3)).toEqual([0, 2, 4]);
    expect(GALLERY_MAX_EPISODES).toBe(3000); expect(GALLERY_PAGE_SIZE).toBe(48); expect(LAZY_ROOT_MARGIN).toBe("200px"); expect(FLAG_STORAGE_KEY).toBe("flagged-episodes");
  });

  it("episode-flags-export", () => {
    const preview = deletePreview([5, 1, 3], "org/data");
    expect(preview).toContain("--repo_id org/data");
    expect(preview).toContain("--new_repo_id org/data_filtered");
    expect(preview).toContain('--operation.episode_indices "[1, 3, 5]"');
    expect(preview.match(/lerobot-edit-dataset/g)).toHaveLength(2);
  });

  it("length-low-movement-filtering", () => {
    const episodes = Array.from({ length: 12 }, (_, i) => ({ episode_index: i, duration: i, movement: 12 - i }));
    expect(lowMovement(episodes)).toHaveLength(10); expect(lowMovement(episodes)[0]?.episode_index).toBe(11);
    expect(filterEpisodes(episodes, 2, 8, 8).map((item) => item.episode_index)).toEqual([4, 5, 6, 7, 8]);
  });

  it("maps bounded relative thumbnail descriptors through the media protocol", async () => {
    const velocity = { name: "joint", min: -.2, max: .3, lo: -.2, hi: .3, std: .1,
      maxAbs: .3, motorRange: 2, inactive: false, discrete: false, bins: [1] };
    (globalThis as unknown as { window: unknown }).window = { datasetEditor: {
      mediaUrl: ({ relative_path }: { relative_path: string }) => `datasetui-media://fixture/${relative_path}`,
      datasetAnalytics: async () => ({ statistics: [], histograms: [],
        episodes: [{ episode_index: 0, duration: 1, movement: 0,
          first_image_relative_path: "images/front/episode_000000/frame_000000.png",
          last_image_relative_path: "images/front/episode_000000/frame_000009.png" }],
        variance: [], autocorrelation: [], suggested_chunk_size: null, velocity: [velocity], jerk: [],
        speed_cv: { value: 0, verdict: "consistent", bins: [1], lo: 0, hi: 0 }, alignment: [],
        quality: { gallery: [{ key: "front", total: 1, items: [{ episode_index: 0,
          first_timestamp: 0, last_timestamp: .95,
          first_relative_path: "images/front/episode_000000/frame_000000.png",
          last_relative_path: "images/front/episode_000000/frame_000009.png" }] }] } }),
    } };
    const analytics = await parityApi.analytics("fixture");
    expect(analytics.episodes[0]?.first_image).toContain("datasetui-media://fixture/images/front");
    expect(analytics.quality?.gallery[0]?.items[0]?.last_url).toContain("frame_000009.png");
    expect(analytics.velocity[0]).toMatchObject(velocity);
  });
});

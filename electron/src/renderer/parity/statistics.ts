import { createElement as h, Fragment, type ReactElement } from "react";

import type { AnalyticsData, DatasetCameraSummary, DatasetSummary } from "./types.js";

function formatTotalDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function card(label: string, value: string | number, key = label): ReactElement {
  return h("article", { key }, h("span", null, label), h("strong", null, value));
}

function cameraCard(camera: DatasetCameraSummary): ReactElement {
  const resolution = camera.width === undefined || camera.height === undefined
    ? "unknown"
    : `${camera.width}×${camera.height}`;
  return h("article", { key: camera.name }, h("span", { title: camera.name }, camera.name), h("strong", null, resolution));
}

export function StatisticsPanel({ summary, analytics, loading = false }: {
  summary?: DatasetSummary | undefined;
  analytics?: AnalyticsData | undefined;
  loading?: boolean;
}): ReactElement {
  const lengths = analytics?.episode_lengths;
  const maxHistogramCount = Math.max(1, ...(lengths?.episode_length_histogram.map((bin) => bin.count) ?? []));
  const overview = summary ? h(Fragment, null,
    h("div", { className: "stat-grid overview-primary" },
      card("Robot Type", summary.robot_type ?? "unknown"),
      card("Dataset Version", summary.version ?? "unknown"),
      card("Tasks", summary.total_tasks.toLocaleString()),
    ),
    h("div", { className: "stat-grid overview-secondary" },
      card("Total Frames", summary.total_frames.toLocaleString()),
      card("Total Episodes", summary.total_episodes.toLocaleString()),
      card("FPS", summary.fps),
      h("article", null, h("span", null, "Total Recording Time"), h("strong", {
        "data-testid": "total-duration", "data-seconds": summary.total_duration,
      }, formatTotalDuration(summary.total_duration))),
    ),
    summary.cameras.length ? h("div", { className: "camera-summary" },
      h("h4", null, "Camera Resolutions"),
      h("div", { className: "stat-grid" }, summary.cameras.map(cameraCard)),
    ) : null,
  ) : null;
  const episodeLengths = lengths ? h(Fragment, null,
    h("div", { className: "episode-length-summary" },
      h("h4", null, "Episode Lengths"),
      h("div", { className: "stat-grid" },
        card("Shortest", `${lengths.shortest_episodes[0]?.length_seconds ?? "–"}s`),
        card("Longest", `${lengths.longest_episodes.at(-1)?.length_seconds ?? "–"}s`),
        card("Mean", `${lengths.mean_episode_length}s`),
        card("Median", `${lengths.median_episode_length}s`),
        card("Std Dev", `${lengths.std_episode_length}s`),
      ),
    ),
    lengths.episode_length_histogram.length ? h("div", { className: "episode-length-distribution" },
      h("h4", null, "Episode Length Distribution ", h("small", null, `${lengths.episode_length_histogram.length} bins`)),
      h("svg", {
        className: "bars", viewBox: `0 0 ${Math.max(1, lengths.episode_length_histogram.length)} 100`,
        "aria-label": "Episode length distribution histogram",
      }, lengths.episode_length_histogram.map((bin, index) => {
        const height = Math.max(1, bin.count / maxHistogramCount * 100);
        return h("rect", {
          key: bin.bin_label, x: index, y: 100 - height, width: ".8", height,
        }, h("title", null, `${bin.bin_label}: ${bin.count} episodes`));
      })),
    ) : null,
  ) : null;
  return h("section", { className: "dataset-statistics", "aria-labelledby": "dataset-statistics-heading" },
    h("h3", { id: "dataset-statistics-heading" }, "Dataset Statistics"), overview,
    loading ? h("p", { role: "status" }, "Computing dataset statistics…") : null,
    episodeLengths,
  );
}

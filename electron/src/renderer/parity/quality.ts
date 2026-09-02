import type { EpisodeSummary } from "./types.js";

export const GALLERY_PAGE_SIZE = 48;
export const GALLERY_MAX_EPISODES = 3000;
export const LAZY_ROOT_MARGIN = "200px";
export const FLAG_STORAGE_KEY = "flagged-episodes";

export function populationStats(values: number[]): { mean: number; median: number; std: number } {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return { mean: 0, median: 0, std: 0 };
  const mean = clean.reduce((sum, value) => sum + value, 0) / clean.length;
  const mid = Math.floor(clean.length / 2);
  const median = clean.length % 2 ? clean[mid]! : (clean[mid - 1]! + clean[mid]!) / 2;
  const std = Math.sqrt(clean.reduce((sum, value) => sum + (value - mean) ** 2, 0) / clean.length);
  return { mean, median, std };
}

export function histogramBinCount(count: number): number {
  return Math.max(10, Math.min(50, Math.ceil(Math.log2(Math.max(1, count)) + 1)));
}

export function niceWidth(raw: number): number {
  if (!(raw > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(raw));
  const ratio = raw / power;
  const nice = ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 2.5 ? 2.5 : ratio <= 5 ? 5 : 10;
  return nice * power;
}

export function evenlySample<T>(items: T[], maximum: number): T[] {
  if (items.length <= maximum) return items;
  return Array.from({ length: maximum }, (_, i) => items[Math.round(i / (maximum - 1) * (items.length - 1))]!);
}

export function filterEpisodes(episodes: EpisodeSummary[], min: number, max: number, movementMax?: number): EpisodeSummary[] {
  return episodes.filter((episode) => episode.duration >= min && episode.duration <= max && (movementMax === undefined || episode.movement <= movementMax));
}

export function lowMovement(episodes: EpisodeSummary[]): EpisodeSummary[] {
  return [...episodes].sort((a, b) => a.movement - b.movement).slice(0, 10);
}

export function deletePreview(ids: number[], repoId: string): string {
  const indices = `[${[...ids].sort((a, b) => a - b).join(", ")}]`;
  const original = `# Delete episodes (modifies original dataset)\nlerobot-edit-dataset \\\n    --repo_id ${repoId} \\\n    --operation.type delete_episodes \\\n    --operation.episode_indices "${indices}"`;
  const copy = `# Delete episodes and save to a new dataset (preserves original)\nlerobot-edit-dataset \\\n    --repo_id ${repoId} \\\n    --new_repo_id ${repoId}_filtered \\\n    --operation.type delete_episodes \\\n    --operation.episode_indices "${indices}"`;
  return `${original}\n\n${copy}`;
}

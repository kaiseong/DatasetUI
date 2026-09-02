export const PROGRESS_FILES = ["sarm_progress.parquet", "srm_progress.parquet"] as const;
export const PROGRESS_PRIORITY = ["progress_sparse", "progress_dense", "progress"] as const;
export const PROGRESS_MAX_POINTS = 4000;

export function progressLabel(name: string): string {
  return name.startsWith("progress_") ? `progress | ${name.slice(9)}` : name;
}

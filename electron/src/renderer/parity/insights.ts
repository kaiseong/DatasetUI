export const MAX_ANALYSIS_EPISODES = 120;
export const MAX_FRAMES_PER_EPISODE = 2500;
export const HEATMAP_BINS = 50;
export const AUTOCORRELATION_MAX_LAG = 100;
export const ACTIVE_DELTA_RANGE_RATIO = 0.001;
export const DISCRETE_UNIQUE_LIMIT = 4;
export const VELOCITY_BINS = 30;
export const ALIGNMENT_MAX_LAG = 30;

export function evenlySpacedIndices(length: number, limit: number): number[] {
  if (length <= limit) return Array.from({ length }, (_, i) => i);
  return Array.from({ length: limit }, (_, i) => Math.round(i / (limit - 1) * (length - 1)));
}

export function populationVariance(values: number[]): number {
  if (!values.length) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.max(0, values.reduce((sum, value) => sum + value * value, 0) / values.length - mean * mean);
}

export function suggestedChunk(acf: number[]): number {
  const first = acf.findIndex((value) => value < 0.5);
  return first < 0 ? Math.max(1, acf.length) : first + 1;
}

export function speedVerdict(cv: number): "consistent" | "moderate" | "high" {
  return cv < 0.2 ? "consistent" : cv < 0.4 ? "moderate" : "high";
}

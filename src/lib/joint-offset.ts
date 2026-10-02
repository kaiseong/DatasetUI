/** Joint offset augmentation: per-episode arm offsets added to state and action. */

export const JOINT_COUNT = 7;
/** Absolute bound per joint j0..j6, in degrees. */
export const DEFAULT_JOINT_RANGES_DEG = [0.3, 0.1, 1.0, 0.3, 1.0, 0.2, 1.5];
export const MAX_JOINT_RANGE_DEG = 10;
export const MAX_COPIES = 10;
const ARMS = ["right", "left"] as const;
const FEATURES = ["observation.state", "action"] as const;

/** Small seeded PRNG so a sampling is reproducible from its seed. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Pick round(percent% of indices) (at least one) with a seed; sorted. */
export function sampleEpisodes(
  indices: number[],
  percent: number,
  seed: number,
): number[] {
  if (!indices.length) return [];
  const share = Math.min(100, Math.max(0, percent)) / 100;
  const count = Math.min(
    indices.length,
    Math.max(1, Math.round(indices.length * share)),
  );
  const random = mulberry32(seed);
  const pool = [...indices];
  for (let index = pool.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [pool[index], pool[other]] = [pool[other], pool[index]];
  }
  return pool.slice(0, count).sort((left, right) => left - right);
}

function featureNames(info: unknown, feature: string): string[] {
  const features = (info as { features?: Record<string, { names?: unknown }> })
    ?.features;
  let names = features?.[feature]?.names;
  if (names && typeof names === "object" && !Array.isArray(names))
    names = Object.values(names)[0];
  return Array.isArray(names) ? names.map(String) : [];
}

/** Names the augmentation needs in observation.state and action, and which are missing. */
export function armDimensionStatus(info: unknown): {
  ready: boolean;
  missing: string[];
} {
  const missing: string[] = [];
  for (const feature of FEATURES) {
    const names = new Set(featureNames(info, feature));
    for (const arm of ARMS)
      for (let joint = 0; joint < JOINT_COUNT; joint += 1) {
        const name = `${arm}_arm_${joint}`;
        if (!names.has(name)) missing.push(`${feature}.${name}`);
      }
  }
  return { ready: missing.length === 0, missing };
}

export function augmentationSummary(
  total: number,
  augmented: number,
  copies: number,
): { output: number; added: number; videoFactor: number } {
  const added = augmented * copies;
  return {
    output: total + added,
    added,
    videoFactor: total ? (total + added) / total : 1,
  };
}

export function validRanges(ranges: number[]): boolean {
  return (
    ranges.length === JOINT_COUNT &&
    ranges.every(
      (value) =>
        Number.isFinite(value) && value >= 0 && value <= MAX_JOINT_RANGE_DEG,
    )
  );
}

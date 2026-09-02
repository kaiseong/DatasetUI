import type { Series } from "./types.js";

export interface ScaleGroup { min: number; max: number; series: Series[] }

function range(series: Series): { min: number; max: number } | undefined {
  const finite = series.values.filter(Number.isFinite);
  return finite.length ? { min: Math.min(...finite), max: Math.max(...finite) } : undefined;
}

export function groupByScale(series: Series[]): ScaleGroup[] {
  const groups: ScaleGroup[] = [];
  const unused = new Set(series);
  for (const base of series) {
    if (!unused.delete(base)) continue;
    const baseRange = range(base);
    if (!baseRange) continue;
    const selected = [base];
    for (const candidate of series) {
      if (selected.length >= 6 || !unused.has(candidate)) continue;
      const candidateRange = range(candidate);
      if (!candidateRange) continue;
      const minDelta = Math.abs(
        Math.log10(Math.abs(baseRange.min) + 1e-9) -
        Math.log10(Math.abs(candidateRange.min) + 1e-9),
      );
      const maxDelta = Math.abs(
        Math.log10(Math.abs(baseRange.max) + 1e-9) -
        Math.log10(Math.abs(candidateRange.max) + 1e-9),
      );
      if (minDelta <= 2 && maxDelta <= 2) {
        selected.push(candidate);
        unused.delete(candidate);
      }
    }
    const selectedRanges = selected.map((item) => range(item)!).filter(Boolean);
    groups.push({
      min: Math.min(...selectedRanges.map((item) => item.min)),
      max: Math.max(...selectedRanges.map((item) => item.max)),
      series: selected,
    });
  }
  return groups;
}

export function currentValue(series: Series, timestamps: number[], time: number): number | undefined {
  let index = timestamps.findIndex((stamp) => stamp >= time);
  if (index < 0) index = timestamps.length - 1;
  return index >= 0 ? series.values[index] : undefined;
}

export function polyline(
  values: number[],
  width: number,
  height: number,
  scaleMin?: number,
  scaleMax?: number,
): string {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return "";
  const min = scaleMin ?? Math.min(...finite);
  const max = scaleMax ?? Math.max(...finite);
  const span = max - min || 1;
  return values.map((value, i) => `${values.length < 2 ? 0 : i / (values.length - 1) * width},${height - (value - min) / span * height}`).join(" ");
}

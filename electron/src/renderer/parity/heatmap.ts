export function parityHeatClass(value: number): string {
  const bucket = Math.round(Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0)) * 9);
  return `heat-${bucket}`;
}

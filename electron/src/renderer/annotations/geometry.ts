export interface Size { width: number; height: number }
export interface Point { x: number; y: number }
export interface RenderedRect extends Size { left: number; top: number }

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function round4(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

export function letterboxRect(container: Size, source: Size): RenderedRect {
  if (container.width <= 0 || container.height <= 0 || source.width <= 0 || source.height <= 0) {
    return { left: 0, top: 0, width: Math.max(0, container.width), height: Math.max(0, container.height) };
  }
  const sourceAspect = source.width / source.height;
  const containerAspect = container.width / container.height;
  if (containerAspect > sourceAspect) {
    const width = container.height * sourceAspect;
    return { left: (container.width - width) / 2, top: 0, width, height: container.height };
  }
  const height = container.width / sourceAspect;
  return { left: 0, top: (container.height - height) / 2, width: container.width, height };
}

export function canvasToNormalized(point: Point, rect: RenderedRect): Point {
  if (rect.width <= 0 || rect.height <= 0) return { x: 0, y: 0 };
  return {
    x: round4(clamp01((point.x - rect.left) / rect.width)),
    y: round4(clamp01((point.y - rect.top) / rect.height)),
  };
}

export function normalizedToCanvas(point: Point, rect: RenderedRect): Point {
  return { x: rect.left + clamp01(point.x) * rect.width, y: rect.top + clamp01(point.y) * rect.height };
}

export type NormalizedDraw =
  | { kind: "bbox"; bbox: [number, number, number, number] }
  | { kind: "keypoint"; point: [number, number] };

export function finishDraw(start: Point, end: Point, rect: RenderedRect, thresholdPx = 4): NormalizedDraw {
  const a = canvasToNormalized(start, rect);
  const b = canvasToNormalized(end, rect);
  if (Math.hypot(end.x - start.x, end.y - start.y) <= thresholdPx) {
    return { kind: "keypoint", point: [b.x, b.y] };
  }
  return {
    kind: "bbox",
    bbox: [round4(Math.min(a.x, b.x)), round4(Math.min(a.y, b.y)), round4(Math.max(a.x, b.x)), round4(Math.max(a.y, b.y))],
  };
}

export function moveBbox(
  bbox: [number, number, number, number],
  delta: Point,
): [number, number, number, number] {
  const [x1, y1, x2, y2] = bbox;
  const width = clamp01(x2) - clamp01(x1);
  const height = clamp01(y2) - clamp01(y1);
  const nx1 = clamp01(Math.min(1 - width, x1 + delta.x));
  const ny1 = clamp01(Math.min(1 - height, y1 + delta.y));
  return [round4(nx1), round4(ny1), round4(nx1 + width), round4(ny1 + height)];
}

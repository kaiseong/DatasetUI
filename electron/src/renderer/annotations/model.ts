import { canonicalize, isPersistent, type CanonicalAnnotationAtom, type EditableAnnotationAtom } from "./types.js";

let fallbackId = 0;
export function makeAnnotationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  fallbackId += 1;
  return `annotation-${Date.now()}-${fallbackId}`;
}

export function snapToTimestamp(timestamps: readonly number[], requested: number): number {
  if (timestamps.length === 0) return requested;
  let best = timestamps[0]!;
  let distance = Math.abs(requested - best);
  for (let index = 1; index < timestamps.length; index += 1) {
    const candidate = timestamps[index]!;
    const nextDistance = Math.abs(requested - candidate);
    if (nextDistance < distance) {
      best = candidate;
      distance = nextDistance;
    }
  }
  return best;
}

export function envelope(atom: CanonicalAnnotationAtom, id = makeAnnotationId(), endTimestamp?: number): EditableAnnotationAtom {
  return endTimestamp === undefined ? { ...atom, id } : { ...atom, id, endTimestamp };
}

export function serializeAtoms(atoms: readonly EditableAnnotationAtom[]): CanonicalAnnotationAtom[] {
  return atoms.map(canonicalize);
}

export function clampAtom(atom: EditableAnnotationAtom, duration: number, timestamps: readonly number[]): EditableAnnotationAtom {
  const safeDuration = Math.max(0, duration);
  const requested = Math.max(0, Math.min(safeDuration, atom.style === "task_aug" ? 0 : atom.timestamp));
  const timestamp = snapToTimestamp(timestamps, requested);
  if (!isPersistent(atom) || atom.endTimestamp === undefined) return { ...atom, timestamp };
  const end = snapToTimestamp(timestamps, Math.max(timestamp, Math.min(safeDuration, atom.endTimestamp)));
  return { ...atom, timestamp, endTimestamp: Math.max(timestamp, end) };
}

export function updateRange(
  atom: EditableAnnotationAtom,
  start: number,
  end: number,
  duration: number,
  timestamps: readonly number[],
): EditableAnnotationAtom {
  if (!isPersistent(atom)) throw new Error("Only persistent atoms can have a span");
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) throw new Error("Annotation range is invalid");
  return clampAtom({ ...atom, timestamp: start, endTimestamp: end }, duration, timestamps);
}

export function overlayAtoms(
  atoms: readonly EditableAnnotationAtom[],
  camera: string,
  currentTime: number,
  selectedId: string | null,
  paused: boolean,
): EditableAnnotationAtom[] {
  return atoms.filter((atom) => {
    if (atom.role !== "assistant" || atom.style !== "vqa") return false;
    if (atom.camera !== null && atom.camera !== camera) return false;
    return Math.abs(atom.timestamp - currentTime) <= 0.05 || (paused && atom.id === selectedId);
  });
}

export interface TimelineMarker {
  atom: EditableAnnotationAtom;
  start: number;
  end: number;
  kind: "span" | "tick";
}

export function timelineMarkers(atoms: readonly EditableAnnotationAtom[], duration: number): TimelineMarker[] {
  const sorted = [...atoms].sort((a, b) => a.timestamp - b.timestamp);
  return sorted.map((atom) => {
    if (atom.style === "task_aug") {
      return { atom, start: 0, end: duration, kind: "span" as const };
    }
    if (atom.style === "subtask" || atom.style === "plan") {
      const next = sorted.find((candidate) => candidate.style === atom.style && candidate.timestamp > atom.timestamp);
      return {
        atom,
        start: atom.timestamp,
        end: next?.timestamp ?? duration,
        kind: "span" as const,
      };
    }
    return { atom, start: atom.timestamp, end: atom.timestamp, kind: "tick" as const };
  });
}

export type AnnotationRole = "user" | "assistant";

export type AnnotationStyle =
  | "task_aug"
  | "subtask"
  | "plan"
  | "memory"
  | "interjection"
  | "vqa";

export interface AnnotationToolCall {
  type: "function";
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

/** The exact v3.1 atom written to language_persistent/language_events. */
export interface CanonicalAnnotationAtom {
  role: AnnotationRole;
  content: string | null;
  style: AnnotationStyle | null;
  timestamp: number;
  camera: string | null;
  tool_calls: AnnotationToolCall[] | null;
}

/** Renderer-only envelope. id/endTimestamp are never written into an atom row. */
export interface EditableAnnotationAtom extends CanonicalAnnotationAtom {
  id: string;
  endTimestamp?: number;
}

export type VqaKind = "bbox" | "keypoint" | "count" | "attribute" | "spatial";

export interface VqaBboxAnswer {
  detections: Array<{
    label: string;
    bbox_format: "xyxy";
    bbox: [number, number, number, number];
  }>;
}

export interface VqaKeypointAnswer {
  label: string;
  point_format: "xy";
  point: [number, number];
}

export interface VqaCountAnswer { label: string; count: number }
export interface VqaAttributeAnswer { label: string; attribute: string; value: string }
export interface VqaSpatialAnswer { subject: string; relation: string; object: string }
export type VqaAnswer = VqaBboxAnswer | VqaKeypointAnswer | VqaCountAnswer | VqaAttributeAnswer | VqaSpatialAnswer;

export const PERSISTENT_STYLES = new Set<AnnotationStyle>(["task_aug", "subtask", "plan", "memory"]);

export function canonicalize(atom: EditableAnnotationAtom): CanonicalAnnotationAtom {
  return {
    role: atom.role,
    content: atom.content,
    style: atom.style,
    timestamp: atom.timestamp,
    camera: atom.camera,
    tool_calls: atom.tool_calls,
  };
}

export function isPersistent(atom: CanonicalAnnotationAtom): boolean {
  return atom.style !== null && PERSISTENT_STYLES.has(atom.style);
}

export function isSpeech(atom: CanonicalAnnotationAtom): boolean {
  return atom.role === "assistant" && atom.content === null && atom.style === null &&
    atom.camera === null && atom.tool_calls?.[0]?.function.name === "say";
}

export function parseVqa(content: string | null): VqaAnswer | null {
  if (!content) return null;
  try {
    const value = JSON.parse(content) as Record<string, unknown>;
    if (Array.isArray(value.detections)) return value as unknown as VqaBboxAnswer;
    if (value.point_format === "xy" && Array.isArray(value.point)) return value as unknown as VqaKeypointAnswer;
    if (typeof value.label === "string" && typeof value.count === "number") return value as unknown as VqaCountAnswer;
    if (typeof value.label === "string" && typeof value.attribute === "string" && typeof value.value === "string") return value as unknown as VqaAttributeAnswer;
    if (typeof value.subject === "string" && typeof value.relation === "string" && typeof value.object === "string") return value as unknown as VqaSpatialAnswer;
    return null;
  } catch {
    return null;
  }
}

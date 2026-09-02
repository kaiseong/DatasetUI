import type {
  AnnotationStyle,
  CanonicalAnnotationAtom,
  VqaAnswer,
  VqaKind,
} from "./types.js";

function textAtom(style: AnnotationStyle, content: string, timestamp: number): CanonicalAnnotationAtom {
  const trimmed = content.trim();
  if (!trimmed) throw new Error("Annotation text is required");
  return {
    role: style === "task_aug" || style === "interjection" ? "user" : "assistant",
    content: trimmed,
    style,
    timestamp: style === "task_aug" ? 0 : timestamp,
    camera: null,
    tool_calls: null,
  };
}

export function buildPersistent(style: "task_aug" | "subtask" | "plan" | "memory", content: string, timestamp: number): CanonicalAnnotationAtom {
  return textAtom(style, content, timestamp);
}

export function buildInterjection(content: string, timestamp: number): CanonicalAnnotationAtom {
  return textAtom("interjection", content, timestamp);
}

export function buildSpeech(text: string, timestamp: number): CanonicalAnnotationAtom {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("Speech text is required");
  return {
    role: "assistant",
    content: null,
    style: null,
    timestamp,
    camera: null,
    tool_calls: [{ type: "function", function: { name: "say", arguments: { text: trimmed } } }],
  };
}

export function buildVqaPair(
  kind: VqaKind,
  question: string,
  answer: VqaAnswer,
  timestamp: number,
  camera: string,
): [CanonicalAnnotationAtom, CanonicalAnnotationAtom] {
  if (!question.trim()) throw new Error("VQA question is required");
  if (!camera) throw new Error("VQA annotations require a camera");
  validateVqa(kind, answer);
  return [
    { role: "user", content: question.trim(), style: "vqa", timestamp, camera, tool_calls: null },
    { role: "assistant", content: JSON.stringify(answer), style: "vqa", timestamp, camera, tool_calls: null },
  ];
}

function validateVqa(kind: VqaKind, answer: VqaAnswer): void {
  const record = answer as unknown as Record<string, unknown>;
  if (kind === "bbox" && !Array.isArray(record.detections)) throw new Error("BBox answer requires detections");
  if (kind === "keypoint" && record.point_format !== "xy") throw new Error("Keypoint answer requires xy coordinates");
  if (kind === "count" && (!Number.isInteger(record.count) || Number(record.count) < 0)) throw new Error("Count must be a non-negative integer");
  if (kind === "attribute" && !String(record.attribute ?? "").trim()) throw new Error("Attribute is required");
  if (kind === "spatial" && !String(record.relation ?? "").trim()) throw new Error("Spatial relation is required");
}

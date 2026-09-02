import { describe, expect, it } from "vitest";

import { createAnnotationAdapter, draftKey, type SessionStorageLike } from "../../src/renderer/annotations/adapter.js";
import { buildInterjection, buildPersistent, buildSpeech, buildVqaPair } from "../../src/renderer/annotations/constructors.js";
import { finishDraw, moveBbox } from "../../src/renderer/annotations/geometry.js";
import { clampAtom, envelope, overlayAtoms, serializeAtoms, snapToTimestamp, timelineMarkers, updateRange } from "../../src/renderer/annotations/model.js";
import { parseVqa } from "../../src/renderer/annotations/types.js";

class MemoryStorage implements SessionStorageLike {
  readonly values = new Map<string, string>();
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  setItem(key: string, value: string): void { this.values.set(key, value); }
}

describe("Space parity annotation feature ids", () => {
  it("annotation-task-aug", async () => {
    const atom = envelope(buildPersistent("task_aug", "pick cube", 4), "task", 8);
    const storage = new MemoryStorage();
    const adapter = createAnnotationAdapter(undefined, storage);
    expect(atom).toMatchObject({ role: "user", style: "task_aug", timestamp: 0, content: "pick cube" });
    await adapter.save("p", 0, [atom]);
    expect(storage.getItem(draftKey("p", 0))).toContain("task_aug");
  });

  it("annotation-subtask", () => {
    const atom = envelope(buildPersistent("subtask", "grasp", 0), "subtask", 1);
    expect(updateRange(atom, 0.4, 1.7, 2, [0, 0.5, 1, 1.5, 2])).toMatchObject({ timestamp: 0.5, endTimestamp: 1.5 });
  });

  it("annotation-plan", () => {
    const initial = envelope(buildPersistent("plan", "grasp then lift", 0), "p1", 1);
    const refresh = envelope(buildPersistent("plan", "release", 1), "p2", 2);
    expect(timelineMarkers([initial, refresh], 2).map((marker) => [marker.kind, marker.start, marker.end])).toEqual([["span", 0, 1], ["span", 1, 2]]);
  });

  it("annotation-memory", () => {
    expect(buildPersistent("memory", "cube grasped", 1.5)).toEqual({ role: "assistant", content: "cube grasped", style: "memory", timestamp: 1.5, camera: null, tool_calls: null });
  });

  it("annotation-interjection", () => {
    expect(buildInterjection("skip this", 1)).toMatchObject({ role: "user", style: "interjection", timestamp: 1 });
  });

  it("annotation-speech-say", () => {
    expect(buildSpeech("okay", 1)).toMatchObject({ role: "assistant", content: null, style: null, camera: null, tool_calls: [{ function: { name: "say", arguments: { text: "okay" } } }] });
  });

  it("annotation-vqa-bbox", () => {
    const [, atom] = buildVqaPair("bbox", "Where?", { detections: [{ label: "cube", bbox_format: "xyxy", bbox: [0.1, 0.2, 0.8, 0.9] }] }, 1, "top");
    expect(parseVqa(atom.content)).toEqual({ detections: [{ label: "cube", bbox_format: "xyxy", bbox: [0.1, 0.2, 0.8, 0.9] }] });
  });

  it("annotation-vqa-keypoint", () => {
    const [, atom] = buildVqaPair("keypoint", "Point?", { label: "gripper", point_format: "xy", point: [0.3, 0.7] }, 1, "top");
    expect(parseVqa(atom.content)).toEqual({ label: "gripper", point_format: "xy", point: [0.3, 0.7] });
  });

  it("annotation-vqa-count", () => {
    const [, atom] = buildVqaPair("count", "How many?", { label: "cube", count: 3 }, 1, "top");
    expect(parseVqa(atom.content)).toEqual({ label: "cube", count: 3 });
  });

  it("annotation-vqa-attribute", () => {
    const [, atom] = buildVqaPair("attribute", "Color?", { label: "cube", attribute: "color", value: "blue" }, 1, "top");
    expect(parseVqa(atom.content)).toEqual({ label: "cube", attribute: "color", value: "blue" });
  });

  it("annotation-vqa-spatial", () => {
    const [, atom] = buildVqaPair("spatial", "Where?", { subject: "cube", relation: "left of", object: "bowl" }, 1, "top");
    expect(parseVqa(atom.content)).toEqual({ subject: "cube", relation: "left of", object: "bowl" });
  });

  it("annotation-frame-snapping", () => {
    expect(snapToTimestamp([0.1, 0.37, 0.91], 0.64)).toBe(0.37);
    expect(clampAtom(envelope(buildInterjection("now", 0.64)), 1, [0.1, 0.37, 0.91]).timestamp).toBe(0.37);
  });

  it("annotation-editable-timeline", () => {
    const atom = envelope(buildPersistent("subtask", "move", 0), "editable");
    const boundary = envelope(buildPersistent("subtask", "next", 1.5), "boundary");
    const moved = updateRange(atom, 0.5, 1.5, 2, [0, 0.5, 1, 1.5, 2]);
    expect(timelineMarkers([moved, boundary], 2)[0]).toMatchObject({ start: 0.5, end: 1.5, kind: "span" });
    expect(serializeAtoms([moved, boundary])).toHaveLength(2);
    expect(serializeAtoms([])).toEqual([]);
  });

  it("annotation-camera-overlays", () => {
    const [, top] = buildVqaPair("count", "Count?", { label: "cube", count: 1 }, 1, "top");
    const [, wrist] = buildVqaPair("count", "Count?", { label: "cube", count: 1 }, 1, "wrist");
    expect(overlayAtoms([envelope(top, "top"), envelope(wrist, "wrist")], "top", 1.04, null, false).map((atom) => atom.id)).toEqual(["top"]);
  });

  it("annotation-bbox-drag", () => {
    expect(finishDraw({ x: 90, y: 80 }, { x: 10, y: 20 }, { left: 0, top: 0, width: 100, height: 100 })).toEqual({ kind: "bbox", bbox: [0.1, 0.2, 0.9, 0.8] });
    expect(moveBbox([0.1, 0.2, 0.4, 0.5], { x: 0.8, y: -0.5 })).toEqual([0.7, 0, 1, 0.3]);
  });

  it("annotation-keypoint-click", () => {
    expect(finishDraw({ x: 50, y: 50 }, { x: 53, y: 50 }, { left: 0, top: 0, width: 100, height: 100 })).toEqual({ kind: "keypoint", point: [0.53, 0.5] });
    expect(finishDraw({ x: -10, y: 120 }, { x: -10, y: 120 }, { left: 0, top: 0, width: 100, height: 100 })).toEqual({ kind: "keypoint", point: [0, 1] });
  });
});

import { describe, expect, it } from "vitest";

import { buildInterjection, buildPersistent, buildSpeech, buildVqaPair } from "../../src/renderer/annotations/constructors.js";
import { clampAtom, envelope, overlayAtoms, serializeAtoms, snapToTimestamp, timelineMarkers, updateRange } from "../../src/renderer/annotations/model.js";

describe("annotation constructors", () => {
  it("builds all persistent styles with canonical roles and task_aug pinned to t0", () => {
    expect(buildPersistent("task_aug", " do task ", 8)).toMatchObject({ role: "user", content: "do task", timestamp: 0, camera: null, tool_calls: null });
    for (const style of ["subtask", "plan", "memory"] as const) {
      expect(buildPersistent(style, style, 1.25)).toEqual({ role: "assistant", content: style, style, timestamp: 1.25, camera: null, tool_calls: null });
    }
  });

  it("builds event and speech atoms without leaking display fields", () => {
    expect(buildInterjection(" stop ", 2)).toMatchObject({ role: "user", style: "interjection", content: "stop" });
    expect(buildSpeech(" hello ", 2)).toEqual({
      role: "assistant", content: null, style: null, timestamp: 2, camera: null,
      tool_calls: [{ type: "function", function: { name: "say", arguments: { text: "hello" } } }],
    });
  });

  it("builds matching camera-grounded VQA question/answer pairs for all shapes", () => {
    const answers = {
      count: { label: "cube", count: 2 },
      attribute: { label: "cube", attribute: "color", value: "blue" },
      spatial: { subject: "cube", relation: "left of", object: "bowl" },
      bbox: { detections: [{ label: "cube", bbox_format: "xyxy" as const, bbox: [0.1, 0.2, 0.8, 0.9] as [number, number, number, number] }] },
      keypoint: { label: "gripper", point_format: "xy" as const, point: [0.4, 0.7] as [number, number] },
    };
    for (const kind of Object.keys(answers) as Array<keyof typeof answers>) {
      const [question, answer] = buildVqaPair(kind, "Where?", answers[kind], 3, "observation.images.top");
      expect(question).toMatchObject({ role: "user", style: "vqa", camera: "observation.images.top", timestamp: 3 });
      expect(answer).toMatchObject({ role: "assistant", style: "vqa", camera: "observation.images.top", timestamp: 3 });
      expect(JSON.parse(answer.content!)).toEqual(answers[kind]);
    }
  });
});

describe("annotation timeline model", () => {
  const timestamps = [0, 0.5, 1, 1.5, 2];

  it("snaps by a linear nearest scan and keeps the first timestamp on a tie", () => {
    expect(snapToTimestamp(timestamps, 0.75)).toBe(0.5);
    expect(snapToTimestamp([], 0.73)).toBe(0.73);
  });

  it("clamps and snaps moves/resizes and rejects reversed ranges", () => {
    const atom = envelope(buildPersistent("subtask", "grasp", 0.4), "s", 1.6);
    expect(clampAtom({ ...atom, timestamp: -4, endTimestamp: 8 }, 2, timestamps)).toMatchObject({ timestamp: 0, endTimestamp: 2 });
    expect(updateRange(atom, 0.8, 1.7, 2, timestamps)).toMatchObject({ timestamp: 1, endTimestamp: 1.5 });
    expect(() => updateRange(atom, 1.2, 0.5, 2, timestamps)).toThrow("range is invalid");
  });

  it("derives spans/ticks and strips renderer envelope on serialization", () => {
    const span = envelope(buildPersistent("plan", "plan", 0.5), "p", 2);
    const tick = envelope(buildInterjection("no", 1), "i");
    expect(timelineMarkers([tick, span], 2).map((item) => item.kind)).toEqual(["span", "tick"]);
    expect(serializeAtoms([span])[0]).not.toHaveProperty("id");
    expect(serializeAtoms([span])[0]).not.toHaveProperty("endTimestamp");
  });

  it("derives canonical persistent lanes from activation timestamps", () => {
    const task = envelope(buildPersistent("task_aug", "pick", 0), "task");
    const first = envelope(buildPersistent("subtask", "reach", 0.5), "first");
    const second = envelope(buildPersistent("subtask", "grasp", 1.5), "second");
    const memory = envelope(buildPersistent("memory", "seen", 1), "memory");
    expect(timelineMarkers([memory, second, task, first], 2).map(({ atom, start, end, kind }) => [atom.id, start, end, kind])).toEqual([
      ["task", 0, 2, "span"],
      ["first", 0.5, 1.5, "span"],
      ["memory", 1, 1, "tick"],
      ["second", 1.5, 2, "span"],
    ]);
  });

  it("routes assistant VQA overlays by exact camera and 0.05 second window", () => {
    const [, answer] = buildVqaPair("count", "How many?", { label: "cube", count: 1 }, 1, "top");
    const selected = envelope(answer, "selected");
    const other = envelope({ ...answer, camera: "wrist" }, "other");
    const global = envelope({ ...answer, camera: null, timestamp: 1.04 }, "global");
    expect(overlayAtoms([selected, other, global], "top", 1.049, null, false).map((a) => a.id)).toEqual(["selected", "global"]);
    expect(overlayAtoms([selected], "top", 8, "selected", true)).toEqual([selected]);
    expect(overlayAtoms([selected], "top", 8, "selected", false)).toEqual([]);
  });
});

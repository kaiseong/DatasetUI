import { describe, expect, test } from "bun:test";
import {
  addCorrectionPoint,
  candidateGuidance,
  candidateGuidanceSignature,
  setObjectTarget,
  objectForInputTarget,
  clampEdgeMargin,
  clampNormalized,
  clearFrameCorrections,
  emptySegmentationDraft,
  hasSegmentationGuidance,
  normalizeSegmentationBox,
  promptFor,
  retainRequestKey,
  updatePrompt,
  draftFromSegmentationSpec,
  orderWorkflowCameras,
  segmentationBatchCanExport,
  segmentationMaskSignature,
  invalidateChangedMaskSelection,
  moveSemanticPromptToFrame,
  objectsFromDraft,
  semanticPromptFor,
  updateSemanticPrompt,
  batchItemAttention,
  batchEstimate,
} from "../segmentation-draft";

describe("segmentation draft", () => {
  test("new drafts preserve foreground with black background and require no image", () => {
    expect(emptySegmentationDraft()).toMatchObject({
      mode: "object_selection",
      render_mode: "black",
      camera_mode: "fixed",
      background_base64: "",
      manual_regions: [],
      selected_candidate_ids: [],
    });
  });

  test("object keyframes and corrections never overwrite other tracks", () => {
    let draft = updatePrompt(
      emptySegmentationDraft(),
      4,
      "protect",
      (prompt) => ({ ...prompt, text: "black plug" }),
      1,
    );
    draft = updatePrompt(
      draft,
      4,
      "protect",
      (prompt) => ({ ...prompt, text: "board" }),
      2,
    );
    expect(promptFor(draft, 4, "protect", 1).text).toBe("black plug");
    expect(promptFor(draft, 4, "protect", 2).text).toBe("board");
    draft = addCorrectionPoint(
      draft,
      4,
      "protect",
      0.03,
      { x: 0.2, y: 0.3 },
      "add",
      1,
    );
    draft = addCorrectionPoint(
      draft,
      4,
      "protect",
      0.03,
      { x: 0.6, y: 0.7 },
      "add",
      2,
    );
    expect(draft.corrections).toHaveLength(2);
    expect(
      clearFrameCorrections(draft, 4, "protect", 1).corrections[0].object_id,
    ).toBe(2);
  });

  test("manual preservation is valid guidance without a model prompt", () => {
    expect(
      hasSegmentationGuidance(
        {
          ...emptySegmentationDraft(),
          mode: "protect_foreground",
          manual_regions: [{ box: [0.2, 0.2, 0.4, 0.4] }],
        },
        "protect",
      ),
    ).toBe(true);
  });

  test("only render and selection changes reuse cached masks", () => {
    const draft = emptySegmentationDraft();
    const signature = segmentationMaskSignature(draft);
    expect(
      segmentationMaskSignature({
        ...draft,
        render_mode: "image",
        background_base64: "new-image",
        edge_margin_px: 8,
        selected_candidate_ids: ["1-1"],
      }),
    ).toBe(signature);
    expect(
      segmentationMaskSignature(
        updatePrompt(
          draft,
          0,
          "protect",
          (prompt) => ({ ...prompt, text: "board" }),
          1,
        ),
      ),
    ).not.toBe(signature);
    expect(
      segmentationMaskSignature({ ...draft, camera_mode: "wrist" }),
    ).not.toBe(signature);
  });

  test("model guidance edits clear stale candidate choices while rerender preserves them", () => {
    const current = {
      ...emptySegmentationDraft(),
      selected_candidate_ids: ["1-1"],
    };
    const edited = updatePrompt(
      current,
      0,
      "protect",
      (prompt) => ({ ...prompt, text: "new object" }),
      1,
    );
    expect(
      invalidateChangedMaskSelection(current, edited).selected_candidate_ids,
    ).toEqual([]);
    expect(
      invalidateChangedMaskSelection(current, {
        ...current,
        render_mode: "image",
      }).selected_candidate_ids,
    ).toEqual(["1-1"]);
  });

  test("workflow cameras stack front, right, left, then the rest in order", () => {
    expect(
      orderWorkflowCameras([
        "observation.images.left",
        "observation.images.wrist",
        "observation.images.right",
        "observation.images.front",
        "observation.images.top",
      ]),
    ).toEqual([
      "observation.images.front",
      "observation.images.right",
      "observation.images.left",
      "observation.images.wrist",
      "observation.images.top",
    ]);
    expect(orderWorkflowCameras(["cam_left_wrist", "cam_front"])).toEqual([
      "cam_front",
      "cam_left_wrist",
    ]);
  });

  test("edge margin is a bounded whole pixel count that survives reload", () => {
    expect(clampEdgeMargin(3.6)).toBe(4);
    expect(clampEdgeMargin(-2)).toBe(0);
    expect(clampEdgeMargin(500)).toBe(64);
    expect(clampEdgeMargin(Number.NaN)).toBe(0);
    const spec = {
      ...emptySegmentationDraft(),
      dataset_id: "source",
      fingerprint: "hash",
      episode_index: 1,
      video_key: "top",
    };
    expect(draftFromSegmentationSpec(spec).edge_margin_px).toBe(0);
    expect(
      draftFromSegmentationSpec({ ...spec, edge_margin_px: 6 }).edge_margin_px,
    ).toBe(6);
  });

  test("loaded drafts cannot leak stale source-preview cache identifiers", () => {
    const draft = draftFromSegmentationSpec({
      ...emptySegmentationDraft(),
      dataset_id: "source",
      fingerprint: "hash",
      episode_index: 1,
      video_key: "top",
      source_preview_id: "stale-cache",
    });
    expect("source_preview_id" in draft).toBe(false);
    expect("dataset_id" in draft).toBe(false);
  });

  test("export gate requires every clip successful and explicitly approved", () => {
    const item = { job_status: "succeeded", review_status: "approved" };
    expect(
      segmentationBatchCanExport({
        ready_to_export: true,
        items: [item, item],
      }),
    ).toBe(true);
    expect(
      segmentationBatchCanExport({
        ready_to_export: true,
        items: [item, { ...item, review_status: "pending" }],
      }),
    ).toBe(false);
    expect(
      segmentationBatchCanExport({
        ready_to_export: true,
        items: [item, { ...item, job_status: "failed" }],
      }),
    ).toBe(false);
    expect(
      segmentationBatchCanExport({
        ready_to_export: true,
        items: [{ ...item, dispatch_error: "offline" }],
      }),
    ).toBe(false);
    expect(
      segmentationBatchCanExport({ ready_to_export: true, items: [] }),
    ).toBe(false);
    expect(
      segmentationBatchCanExport({ ready_to_export: false, items: [item] }),
    ).toBe(false);
  });
  test("prompt edits are isolated by frame and target", () => {
    let draft = emptySegmentationDraft();
    draft = updatePrompt(draft, 3, "replace", (prompt) => ({
      ...prompt,
      text: "counter",
    }));
    draft = updatePrompt(draft, 3, "protect", (prompt) => ({
      ...prompt,
      text: "robot hand",
    }));
    expect(promptFor(draft, 3, "replace").text).toBe("counter");
    expect(promptFor(draft, 3, "protect").text).toBe("robot hand");
    expect(promptFor(draft, 4, "replace").text).toBe("");
  });

  test("empty prompt updates remove the prompt", () => {
    const seeded = updatePrompt(
      emptySegmentationDraft(),
      0,
      "replace",
      (prompt) => ({ ...prompt, text: "floor" }),
    );
    const cleared = updatePrompt(seeded, 0, "replace", (prompt) => ({
      ...prompt,
      text: " ",
    }));
    expect(cleared.prompts).toHaveLength(0);
  });

  test("correction brushes stay per-frame and protection target", () => {
    let draft = addCorrectionPoint(
      emptySegmentationDraft(),
      8,
      "protect",
      0.02,
      { x: 1.4, y: -0.2 },
    );
    expect(draft.corrections[0]).toEqual({
      frame_index: 8,
      target: "protect",
      operation: "add",
      radius: 0.02,
      points: [{ x: 1, y: 0 }],
    });
    draft = clearFrameCorrections(draft, 8, "protect");
    expect(draft.corrections).toHaveLength(0);
  });

  test("normalization rejects non-finite values and clamps bounds", () => {
    expect(clampNormalized(Number.NaN)).toBe(0);
    expect(clampNormalized(-1)).toBe(0);
    expect(clampNormalized(2)).toBe(1);
    expect(normalizeSegmentationBox([0.8, 0.9, 0.5, 0.5])).toEqual([
      0.8, 0.9, 0.2, 0.1,
    ]);
  });

  test("legacy brush corrections alone are not a valid primary prompt", () => {
    const corrected = addCorrectionPoint(
      { ...emptySegmentationDraft(), mode: "protect_foreground" },
      0,
      "replace",
      0.02,
      { x: 0.5, y: 0.5 },
    );
    expect(hasSegmentationGuidance(corrected, "replace")).toBe(false);
  });

  test("erase brush is serialized separately from add brush", () => {
    let draft = addCorrectionPoint(
      emptySegmentationDraft(),
      2,
      "replace",
      0.03,
      { x: 0.4, y: 0.5 },
    );
    draft = addCorrectionPoint(
      draft,
      2,
      "replace",
      0.03,
      { x: 0.6, y: 0.5 },
      "erase",
    );
    expect(draft.corrections.map((item) => item.operation)).toEqual([
      "add",
      "erase",
    ]);
  });

  test("ambiguous preview retries retain a key only for the exact snapshot", () => {
    const first = retainRequestKey(null, "preview:snapshot-1", () => "key-1");
    let generated = false;
    expect(
      retainRequestKey(first, "preview:snapshot-1", () => {
        generated = true;
        return "must-not-replace";
      }),
    ).toBe(first);
    expect(generated).toBe(false);
    expect(
      retainRequestKey(first, "preview:snapshot-2", () => "key-2"),
    ).toEqual({ signature: "preview:snapshot-2", key: "key-2" });
  });

  test("export retries retain a key until the approved output tuple changes", () => {
    const first = retainRequestKey(
      null,
      "profile:preview:approval:output-a",
      () => "export-key-1",
    );
    expect(
      retainRequestKey(
        first,
        "profile:preview:approval:output-a",
        () => "must-not-replace",
      ),
    ).toBe(first);
    expect(
      retainRequestKey(
        first,
        "profile:preview:approval:output-b",
        () => "export-key-2",
      ),
    ).toEqual({
      signature: "profile:preview:approval:output-b",
      key: "export-key-2",
    });
  });
});

test("object intent changes all its keyframes without changing point polarity or another object", () => {
  const draft = emptySegmentationDraft();
  draft.prompts = [0, 5].map((frame_index) => ({
    object_id: 1,
    frame_index,
    target: "protect",
    text: frame_index ? "" : "plug",
    points: [{ x: 0.5, y: 0.5, label: 0 }],
    box: null,
  }));
  draft.prompts.push({ ...draft.prompts[0], object_id: 2 });
  draft.corrections = [
    {
      object_id: 1,
      frame_index: 5,
      target: "protect",
      radius: 0.1,
      operation: "erase",
      points: [{ x: 0.4, y: 0.5 }],
    },
  ];
  const changed = setObjectTarget(draft, 1, "replace");
  expect(
    changed.prompts
      .slice(0, 2)
      .every((item) => item.target === "replace" && item.points[0].label === 0),
  ).toBe(true);
  expect(changed.prompts[2].target).toBe("protect");
  expect(changed.corrections[0].target).toBe("replace");
  expect(changed.corrections[0].operation).toBe("erase");
  expect(segmentationMaskSignature(changed)).not.toBe(
    segmentationMaskSignature(draft),
  );
});

test("new mode accepts additive brush guidance but never manual regions", () => {
  let draft = emptySegmentationDraft();
  expect(
    hasSegmentationGuidance({
      ...draft,
      manual_regions: [{ box: [0, 0, 0.5, 0.5] }],
    }),
  ).toBe(false);
  draft = addCorrectionPoint(
    draft,
    0,
    "replace",
    0.1,
    { x: 0.5, y: 0.5 },
    "add",
    1,
  );
  expect(hasSegmentationGuidance(draft)).toBe(true);
  expect(hasSegmentationGuidance(draft, "protect")).toBe(false);
});

test("switching input intent preserves existing object hints across frames", () => {
  let draft = updatePrompt(
    emptySegmentationDraft(),
    5,
    "protect",
    (p) => ({ ...p, points: [{ x: 0.3, y: 0.5, label: 1 }] }),
    1,
  );
  draft = addCorrectionPoint(
    draft,
    8,
    "protect",
    0.02,
    { x: 0.2, y: 0.4 },
    "add",
    1,
  );
  const before = JSON.stringify(draft);
  expect(objectForInputTarget(draft, 1, "protect", "replace", [1, 2])).toBe(3);
  expect(JSON.stringify(draft)).toBe(before);
  expect(objectForInputTarget(draft, 1, "protect", "protect", [1])).toBe(1);
  expect(
    objectForInputTarget(
      emptySegmentationDraft(),
      1,
      "protect",
      "replace",
      [1],
    ),
  ).toBe(1);
  expect(
    objectForInputTarget(
      draft,
      1,
      "protect",
      "replace",
      Array.from({ length: 32 }, (_, i) => i + 1),
    ),
  ).toBeNull();
});

test("confirmed objects exclude unconfirmed annotations and member hints remain isolated", () => {
  const base = emptySegmentationDraft();
  const first = updatePrompt(
    base,
    0,
    "protect",
    (p) => ({ ...p, points: [{ x: 0.2, y: 0.3, label: 1 }] }),
    1,
    "1-2",
  );
  const next = updatePrompt(
    first,
    0,
    "protect",
    (p) => ({ ...p, points: [{ x: 0.8, y: 0.3, label: 0 }] }),
    1,
    "1-3",
  );
  expect(next.prompts).toHaveLength(2);
  expect(promptFor(next, 0, "protect", 1, "1-2").points[0].label).toBe(1);
  expect(promptFor(next, 0, "protect", 1, "1-3").points[0].label).toBe(0);
});

test("candidate preview retains same-object frame hints without other objects", () => {
  const draft = emptySegmentationDraft();
  draft.prompts = [
    {
      object_id: 1,
      target: "protect",
      frame_index: 0,
      text: "board",
      points: [{ x: 0.2, y: 0.3, label: 1 }],
      box: [0.1, 0.1, 0.3, 0.3],
    },
    {
      object_id: 2,
      target: "replace",
      frame_index: 0,
      text: "wire",
      points: [],
      box: null,
    },
  ];
  draft.corrections = [
    {
      object_id: 1,
      target: "protect",
      frame_index: 0,
      operation: "add",
      radius: 0.02,
      points: [{ x: 0.2, y: 0.3 }],
    },
    {
      object_id: 1,
      target: "protect",
      frame_index: 1,
      operation: "add",
      radius: 0.02,
      points: [{ x: 0.2, y: 0.3 }],
    },
  ];
  const input = candidateGuidance(draft, 1, 0);
  expect(input.prompts).toHaveLength(1);
  expect(input.prompts[0].points).toHaveLength(1);
  expect(input.prompts[0].box).toEqual([0.1, 0.1, 0.3, 0.3]);
  expect(input.corrections).toHaveLength(1);
  const before = candidateGuidanceSignature(draft, 1, 0);
  draft.prompts[0].points = [];
  expect(candidateGuidanceSignature(draft, 1, 0)).not.toBe(before);
});

describe("object-level Instruction", () => {
  const ref = { sample_id: "s-1", candidate_id: "1-3" };
  const base = {
    ...emptySegmentationDraft(),
    prompts: [
      {
        object_id: 1,
        frame_index: 10,
        target: "protect" as const,
        text: "board",
        confidence_threshold: 0.4,
        selected_candidates: [ref],
        points: [{ x: 0.5, y: 0.5, label: 1 as const }],
        box: null,
      },
    ],
  };

  test("the Instruction is the same on every frame", () => {
    expect(semanticPromptFor(base, 1, "protect", 300).frame_index).toBe(10);
    expect(semanticPromptFor(base, 1, "protect", 300).text).toBe("board");
    const edited = updateSemanticPrompt(base, 1, "protect", 300, (p) => ({
      ...p,
      confidence_threshold: 0.6,
    }));
    expect(edited.prompts).toHaveLength(1);
    expect(edited.prompts[0].frame_index).toBe(10);
    expect(edited.prompts[0].confidence_threshold).toBe(0.6);
  });

  test("re-detecting moves the Instruction but keeps old geometry", () => {
    const moved = moveSemanticPromptToFrame(base, 1, "protect", 300);
    const semantic = moved.prompts.find((p) => p.text);
    const geometry = moved.prompts.find((p) => !p.text);
    expect(semantic?.frame_index).toBe(300);
    expect(semantic?.confidence_threshold).toBe(0.4);
    expect(semantic?.selected_candidates).toEqual([]);
    expect(geometry?.frame_index).toBe(10);
    expect(geometry?.points).toHaveLength(1);
  });

  test("objects in effect come from the draft, named from the workspace", () => {
    const draft = {
      ...base,
      corrections: [
        {
          object_id: 2,
          frame_index: 0,
          target: "replace" as const,
          operation: "add" as const,
          radius: 0.02,
          points: [{ x: 0.1, y: 0.1 }],
        },
      ],
    };
    const objects = objectsFromDraft(draft, [
      {
        object_id: 1,
        name: "보드",
        target: "protect",
        prompts: [],
        corrections: [],
      },
    ]);
    expect(
      objects.map((item) => [item.object_id, item.name, item.target]),
    ).toEqual([
      [1, "보드", "protect"],
      [2, "객체 2", "replace"],
    ]);
  });

  test("a server null member id extends the same brush stroke", () => {
    const fromServer = {
      ...emptySegmentationDraft(),
      corrections: [
        {
          object_id: 1,
          frame_index: 0,
          target: "protect" as const,
          operation: "add" as const,
          radius: 0.02,
          member_candidate_id: null,
          points: [{ x: 0.1, y: 0.1 }],
        },
      ],
    };
    const next = addCorrectionPoint(
      fromServer,
      0,
      "protect",
      0.02,
      { x: 0.2, y: 0.2 },
      "add",
      1,
    );
    expect(next.corrections).toHaveLength(1);
    expect(next.corrections[0].points).toHaveLength(2);
  });
});

describe("batch rework list", () => {
  const base = {
    id: "i",
    episode_index: 3,
    video_key: "observation.images.wrist",
    preview_id: "p",
    review_status: "pending" as const,
  };
  const result = {
    preview_id: "p",
    recipe_hash: "h",
    fingerprint: "f",
    frame_count: 100,
    model: "SAM 3.1",
  };

  test("failed, blocked and unselected items need work", () => {
    expect(batchItemAttention({ ...base, job_status: "failed" })).toContain(
      "실패",
    );
    expect(
      batchItemAttention({
        ...base,
        job_status: "succeeded",
        result: { ...result, selection_required: true },
      }),
    ).toContain("후보 선택");
    expect(batchItemAttention({ ...base, job_status: "running" })).toBeNull();
  });

  test("an object missing on some frames is reported with its range", () => {
    const covered = {
      ...base,
      job_status: "succeeded",
      result: {
        ...result,
        object_coverage: [
          {
            object_id: 2,
            target: "protect" as const,
            visible_frames: 100,
            frame_count: 100,
            missing_ranges: [],
          },
          {
            object_id: 4,
            target: "protect" as const,
            visible_frames: 70,
            frame_count: 100,
            missing_ranges: [
              [40, 59],
              [90, 99],
            ] as Array<[number, number]>,
          },
        ],
      },
    };
    expect(batchItemAttention(covered)).toBe(
      "객체 4 누락 30/100프레임 (40~59 외 1구간)",
    );
    expect(
      batchItemAttention({
        ...covered,
        result: {
          ...covered.result,
          object_coverage: [covered.result.object_coverage[0]],
        },
      }),
    ).toBeNull();
    expect(batchItemAttention(covered, true)).toContain("재생성");
  });
});

describe("progress estimates", () => {
  test("whole-job remaining time extrapolates from elapsed time", async () => {
    const { wholeJobEstimate, formatDuration } =
      await import("../job-presentation");
    expect(wholeJobEstimate(0.25, 60)).toEqual({
      percent: 25,
      remainingSeconds: 180,
    });
    expect(wholeJobEstimate(0.01, 60)?.remainingSeconds).toBeNull();
    expect(wholeJobEstimate(undefined, 60)).toBeNull();
    expect(formatDuration(3725)).toBe("1시간 2분");
    expect(formatDuration(200)).toBe("3분 20초");
    expect(formatDuration(9)).toBe("9초");
  });

  test("batch estimate uses finished items and the running item's fraction", () => {
    const item = (
      job_status: string,
      started?: string,
      finished?: string,
      overall?: number,
    ) => ({
      id: job_status,
      episode_index: 0,
      video_key: "wrist",
      preview_id: "p",
      review_status: "pending" as const,
      job_status,
      started_at: started,
      finished_at: finished,
      progress: overall === undefined ? null : { overall },
    });
    const now = Date.parse("2026-10-02T00:10:00Z");
    const estimate = batchEstimate(
      [
        item("succeeded", "2026-10-02T00:00:00Z", "2026-10-02T00:03:00Z"),
        item("failed", "2026-10-02T00:03:00Z", "2026-10-02T00:06:00Z"),
        item("running", "2026-10-02T00:06:00Z", undefined, 0.5),
        item("queued"),
      ],
      now,
    );
    expect(estimate).toEqual({
      done: 2,
      total: 4,
      percent: 62,
      remainingSeconds: 270,
    });
    const first = batchEstimate(
      [
        item("running", "2026-10-02T00:08:00Z", undefined, 0.25),
        item("queued"),
      ],
      now,
    );
    expect(first.remainingSeconds).toBe(480 * 1.75);
  });
});

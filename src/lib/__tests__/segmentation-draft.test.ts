import { describe, expect, test } from "bun:test";
import {
  addCorrectionPoint,
  clampNormalized,
  clearFrameCorrections,
  emptySegmentationDraft,
  hasSegmentationGuidance,
  normalizeSegmentationBox,
  promptFor,
  retainRequestKey,
  updatePrompt,
  draftFromSegmentationSpec,
  segmentationBatchCanExport,
  segmentationMaskSignature,
  invalidateChangedMaskSelection,
  moveSemanticPromptToFrame,
  objectsFromDraft,
  semanticPromptFor,
  updateSemanticPrompt,
} from "../segmentation-draft";

describe("segmentation draft", () => {
  test("new drafts preserve foreground with black background and require no image", () => {
    expect(emptySegmentationDraft()).toMatchObject({
      mode: "protect_foreground",
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

  test("brush corrections alone are not a valid primary prompt", () => {
    const corrected = addCorrectionPoint(
      emptySegmentationDraft(),
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
      { object_id: 1, name: "보드", target: "protect", prompts: [], corrections: [] },
    ]);
    expect(objects.map((item) => [item.object_id, item.name, item.target])).toEqual([
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
    const next = addCorrectionPoint(fromServer, 0, "protect", 0.02, { x: 0.2, y: 0.2 }, "add", 1);
    expect(next.corrections).toHaveLength(1);
    expect(next.corrections[0].points).toHaveLength(2);
  });
});

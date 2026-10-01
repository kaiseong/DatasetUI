import type {
  SegmentationCorrection,
  SegmentationPrompt,
  SegmentationSpec,
  SegmentationTarget,
} from "./segmentation-api";

export type SegmentationDraft = Omit<
  SegmentationSpec,
  | "dataset_id"
  | "fingerprint"
  | "episode_index"
  | "video_key"
  | "source_preview_id"
>;

export type RetainedRequestKey = {
  signature: string;
  key: string;
};

export function retainRequestKey(
  current: RetainedRequestKey | null,
  signature: string,
  createKey: () => string,
): RetainedRequestKey {
  return current?.signature === signature
    ? current
    : { signature, key: createKey() };
}

export function emptySegmentationDraft(): SegmentationDraft {
  return {
    mode: "object_selection",
    render_mode: "black",
    camera_mode: "fixed",
    background_base64: "",
    manual_regions: [],
    selected_candidate_ids: [],
    prompts: [],
    corrections: [],
  };
}

export function clampNormalized(value: number): number {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}

export function normalizeSegmentationBox(
  box: [number, number, number, number],
): [number, number, number, number] {
  const x = Math.min(0.999, clampNormalized(box[0]));
  const y = Math.min(0.999, clampNormalized(box[1]));
  const width = Number(
    Math.max(0.001, Math.min(clampNormalized(box[2]), 1 - x)).toFixed(6),
  );
  const height = Number(
    Math.max(0.001, Math.min(clampNormalized(box[3]), 1 - y)).toFixed(6),
  );
  return [x, y, width, height];
}

export function promptFor(
  draft: SegmentationDraft,
  frameIndex: number,
  target: SegmentationTarget,
  objectId?: number,
  memberCandidateId?: string,
): SegmentationPrompt {
  return (
    draft.prompts.find(
      (item) =>
        item.frame_index === frameIndex &&
        item.target === target &&
        (item.object_id ?? 1) === (objectId ?? 1) &&
        (item.member_candidate_id ?? undefined) === memberCandidateId,
    ) ?? {
      frame_index: frameIndex,
      target,
      ...(objectId ? { object_id: objectId } : {}),
      text: "",
      member_candidate_id: memberCandidateId,
      points: [],
      box: null,
    }
  );
}

export function updatePrompt(
  draft: SegmentationDraft,
  frameIndex: number,
  target: SegmentationTarget,
  update: (prompt: SegmentationPrompt) => SegmentationPrompt,
  objectId?: number,
  memberCandidateId?: string,
): SegmentationDraft {
  const current = promptFor(draft, frameIndex, target, objectId, memberCandidateId);
  const next = update(current);
  const prompts = draft.prompts.filter(
    (item) =>
      item.frame_index !== frameIndex ||
      item.target !== target ||
      (item.object_id ?? 1) !== (objectId ?? 1) ||
      (item.member_candidate_id ?? undefined) !== memberCandidateId,
  );
  const hasContent =
    next.text.trim() || next.points.length > 0 || next.box !== null;
  return { ...draft, prompts: hasContent ? [...prompts, next] : prompts };
}

export function addCorrectionPoint(
  draft: SegmentationDraft,
  frameIndex: number,
  target: SegmentationTarget,
  radius: number,
  point: { x: number; y: number },
  operation: "add" | "erase" = "add",
  objectId?: number,
  memberCandidateId?: string,
): SegmentationDraft {
  const corrections = draft.corrections.filter(
    (item) =>
      item.frame_index !== frameIndex ||
      item.target !== target ||
      (item.object_id ?? 1) !== (objectId ?? 1) ||
      item.operation !== operation ||
      (item.member_candidate_id ?? undefined) !== memberCandidateId,
  );
  const current = draft.corrections.find(
    (item) =>
      item.frame_index === frameIndex &&
      item.target === target &&
      (item.object_id ?? 1) === (objectId ?? 1) &&
      item.operation === operation &&
      (item.member_candidate_id ?? undefined) === memberCandidateId,
  );
  const next: SegmentationCorrection = {
    frame_index: frameIndex,
    target,
    ...(objectId ? { object_id: objectId } : {}),
    operation,
    member_candidate_id: memberCandidateId,
    radius: Math.min(0.25, Math.max(0.001, radius)),
    points: [
      ...(current?.points ?? []),
      { x: clampNormalized(point.x), y: clampNormalized(point.y) },
    ],
  };
  return { ...draft, corrections: [...corrections, next] };
}

export function clearFrameCorrections(
  draft: SegmentationDraft,
  frameIndex: number,
  target: SegmentationTarget,
  objectId?: number,
  memberCandidateId?: string,
): SegmentationDraft {
  return {
    ...draft,
    corrections: draft.corrections.filter(
      (item) =>
        item.frame_index !== frameIndex ||
        item.target !== target ||
        (item.object_id ?? 1) !== (objectId ?? 1) ||
        (item.member_candidate_id ?? undefined) !== memberCandidateId,
    ),
  };
}

export function hasSegmentationGuidance(
  draft: SegmentationDraft,
  target?: SegmentationTarget,
): boolean {
  return (
    draft.prompts.some((prompt) => !target || prompt.target === target) ||
    (draft.mode === "object_selection" &&
      draft.corrections.some(
        (item) =>
          item.operation === "add" && (!target || item.target === target),
      )) ||
    (draft.mode !== "object_selection" &&
      (target === "protect" || !target) &&
      (draft.manual_regions?.length ?? 0) > 0)
  );
}

/** Render choices never invalidate cached model inference; geometry edits do. */
export function segmentationMaskSignature(draft: SegmentationDraft): string {
  return JSON.stringify({
    mode: draft.mode,
    camera_mode: draft.camera_mode ?? "fixed",
    prompts: draft.prompts,
    corrections: draft.corrections,
    manual_regions: draft.manual_regions ?? [],
  });
}

export function invalidateChangedMaskSelection(
  current: SegmentationDraft,
  next: SegmentationDraft,
): SegmentationDraft {
  return segmentationMaskSignature(current) === segmentationMaskSignature(next)
    ? next
    : { ...next, selected_candidate_ids: [] };
}

export function draftFromSegmentationSpec(
  spec: SegmentationSpec,
): SegmentationDraft {
  return {
    mode: spec.mode,
    render_mode:
      spec.render_mode ?? (spec.background_base64 ? "image" : "black"),
    camera_mode: spec.camera_mode ?? "fixed",
    background_base64: spec.background_base64 ?? "",
    prompts: spec.prompts,
    corrections: spec.corrections,
    manual_regions: spec.manual_regions ?? [],
    selected_candidate_ids: spec.selected_candidate_ids ?? [],
  };
}

export function segmentationBatchCanExport(batch: {
  ready_to_export: boolean;
  items: Array<{
    job_status: string;
    review_status: string;
    dispatch_error?: string | null;
  }>;
}): boolean {
  return (
    batch.ready_to_export &&
    batch.items.length > 0 &&
    batch.items.every(
      (item) =>
        item.job_status === "succeeded" &&
        item.review_status === "approved" &&
        !item.dispatch_error,
    )
  );
}

export function setObjectTarget(
  draft: SegmentationDraft,
  objectId: number,
  target: SegmentationTarget,
): SegmentationDraft {
  return {
    ...draft,
    prompts: draft.prompts.map((item) =>
      (item.object_id ?? 1) === objectId ? { ...item, target } : item,
    ),
    corrections: draft.corrections.map((item) =>
      (item.object_id ?? 1) === objectId ? { ...item, target } : item,
    ),
    selected_candidate_ids: [],
  };
}

/** Switching drawing intent never relabels existing annotations. */
export function objectForInputTarget(
  draft: SegmentationDraft,
  objectId: number,
  currentTarget: SegmentationTarget,
  nextTarget: SegmentationTarget,
  reservedIds: number[],
): number | null {
  const annotations = [...draft.prompts, ...draft.corrections];
  if (currentTarget === nextTarget || !annotations.some(item => (item.object_id ?? 1) === objectId)) {
    return objectId;
  }
  const used = new Set([objectId, ...reservedIds, ...annotations.map(item => item.object_id ?? 1)]);
  return Array.from({length: 32}, (_, i) => i + 1).find(id => !used.has(id)) ?? null;
}

/** Merge only the object explicitly confirmed, never another object's draft. */
export function confirmedObjectDraft(
  draft: SegmentationDraft,
  objects: import("./segmentation-api").ConfirmedSegmentationObject[],
): SegmentationDraft {
  return {...draft, prompts: objects.flatMap(o => o.prompts),
    corrections: objects.flatMap(o => o.corrections), selected_candidate_ids: []};
}

/** Candidate preview uses the same object/frame guidance as the saved sample. */
export function candidateGuidance(draft: SegmentationDraft, objectId: number, frameIndex: number) {
  return {
    prompts: draft.prompts.filter(p => (p.object_id ?? 1) === objectId &&
      (p.frame_index === frameIndex || !!p.text)).map(p => ({...p,
        confidence_threshold: p.text ? (p.confidence_threshold ?? 0.5) : p.confidence_threshold})),
    corrections: draft.corrections.filter(c => c.object_id === objectId && c.frame_index === frameIndex),
  };
}

export function candidateGuidanceSignature(draft: SegmentationDraft, objectId: number, frameIndex: number) {
  const guidance = candidateGuidance(draft, objectId, frameIndex);
  return JSON.stringify([objectId, frameIndex, {
    ...guidance, prompts: guidance.prompts.map(p => ({...p, selected_candidates: []})),
  }]);
}

/** An object has one Instruction; it lives on its detection frame. */
export function semanticPromptFor(
  draft: SegmentationDraft,
  objectId: number,
  target: SegmentationTarget,
  frameIndex: number,
): SegmentationPrompt {
  return (
    draft.prompts.find(
      (item) =>
        (item.object_id ?? 1) === objectId &&
        !!item.text.trim() &&
        !item.member_candidate_id,
    ) ?? promptFor(draft, frameIndex, target, objectId)
  );
}

/** Edit the object's Instruction wherever it lives (never per viewed frame). */
export function updateSemanticPrompt(
  draft: SegmentationDraft,
  objectId: number,
  target: SegmentationTarget,
  frameIndex: number,
  update: (prompt: SegmentationPrompt) => SegmentationPrompt,
): SegmentationDraft {
  const semantic = semanticPromptFor(draft, objectId, target, frameIndex);
  return updatePrompt(draft, semantic.frame_index, target, update, objectId);
}

/**
 * Detect on a new frame: the Instruction (with fresh candidates) moves there;
 * geometry drawn on the old detection frame stays on that frame.
 */
export function moveSemanticPromptToFrame(
  draft: SegmentationDraft,
  objectId: number,
  target: SegmentationTarget,
  frameIndex: number,
): SegmentationDraft {
  const semantic = semanticPromptFor(draft, objectId, target, frameIndex);
  if (!semantic.text.trim() || semantic.frame_index === frameIndex) {
    return updatePrompt(draft, frameIndex, target, (current) => ({
      ...current,
      selected_candidates: [],
    }), objectId);
  }
  const { text, confidence_threshold } = semantic;
  let next = updatePrompt(draft, semantic.frame_index, target, (current) => ({
    ...current,
    text: "",
    confidence_threshold: null,
    selected_candidates: [],
  }), objectId);
  next = updatePrompt(next, frameIndex, target, (current) => ({
    ...current,
    text,
    confidence_threshold,
    selected_candidates: [],
  }), objectId);
  return next;
}

/** Objects in effect for this video, named from the saved workspace. */
export function objectsFromDraft(
  draft: SegmentationDraft,
  saved: import("./segmentation-api").ConfirmedSegmentationObject[] = [],
): import("./segmentation-api").ConfirmedSegmentationObject[] {
  const ids = new Map<number, SegmentationTarget>();
  for (const item of [...draft.prompts, ...draft.corrections]) {
    if (!ids.has(item.object_id ?? 1)) ids.set(item.object_id ?? 1, item.target);
  }
  return [...ids.entries()]
    .sort(([left], [right]) => left - right)
    .map(([objectId, target]) => ({
      object_id: objectId,
      name: saved.find((item) => item.object_id === objectId)?.name ?? `객체 ${objectId}`,
      target,
      prompts: draft.prompts
        .filter((item) => (item.object_id ?? 1) === objectId)
        .map((item) => ({ ...item, object_id: objectId })),
      corrections: draft.corrections
        .filter((item) => (item.object_id ?? 1) === objectId)
        .map((item) => ({ ...item, object_id: objectId })),
    }));
}

/** Why a batch item needs more work, or null when it can simply be reviewed. */
export function batchItemAttention(
  item: import("./segmentation-api").SegmentationBatchItem,
  dirty = false,
): string | null {
  if (dirty) return "수정됨 · 미리보기 재생성 필요";
  if (item.job_status === "failed" || item.dispatch_error) {
    return "실패 · 열어서 객체 지정을 보정하세요";
  }
  if (item.job_status !== "succeeded" || !item.result) return null;
  if (item.result.review_blocked) return "보존 영역 없음 · 보정 필요";
  if (item.result.selection_required) return "객체 후보 선택 필요";
  const gaps = (item.result.object_coverage ?? []).filter(
    (entry) => entry.missing_ranges.length > 0,
  );
  if (!gaps.length) return null;
  return gaps
    .map((entry) => {
      const [start, end] = entry.missing_ranges[0];
      const more = entry.missing_ranges.length > 1 ? ` 외 ${entry.missing_ranges.length - 1}구간` : "";
      return `객체 ${entry.object_id} 누락 ${entry.frame_count - entry.visible_frames}/${entry.frame_count}프레임 (${start}~${end}${more})`;
    })
    .join(" · ");
}

/**
 * Whole-batch estimate for one GPU worker processing items in sequence:
 * average finished-item duration x remaining items (the running item counts
 * by its reported whole-job fraction).
 */
export function batchEstimate(
  items: import("./segmentation-api").SegmentationBatchItem[],
  nowMs: number,
): { done: number; total: number; percent: number; remainingSeconds: number | null } {
  const terminal = new Set(["succeeded", "failed", "cancelled", "interrupted"]);
  const durations: number[] = [];
  let done = 0;
  let running = 0;
  let runningElapsed: number | null = null;
  for (const item of items) {
    if (terminal.has(item.job_status)) {
      done += 1;
      const start = item.started_at ? Date.parse(item.started_at) : NaN;
      const end = item.finished_at ? Date.parse(item.finished_at) : NaN;
      if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
        durations.push((end - start) / 1000);
      }
    } else if (item.job_status === "running") {
      running += Math.min(0.99, Math.max(0, item.progress?.overall ?? 0));
      const start = item.started_at ? Date.parse(item.started_at) : NaN;
      if (Number.isFinite(start)) runningElapsed = (nowMs - start) / 1000;
    }
  }
  const total = items.length;
  const progressed = done + running;
  let perItem: number | null = durations.length
    ? durations.reduce((sum, value) => sum + value, 0) / durations.length
    : null;
  if (perItem === null && running >= 0.05 && runningElapsed !== null) {
    perItem = runningElapsed / running;
  }
  return {
    done,
    total,
    percent: total ? Math.floor((progressed / total) * 100) : 0,
    remainingSeconds: perItem === null || done >= total ? null : perItem * (total - progressed),
  };
}

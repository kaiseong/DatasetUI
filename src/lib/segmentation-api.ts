import type { Job } from "./workbench-api";

export type SegmentationTarget = "replace" | "protect";

export type SegmentationPoint = {
  x: number;
  y: number;
  label: 0 | 1;
};

export type SegmentationPrompt = {
  object_id?: number;
  frame_index: number;
  target: SegmentationTarget;
  text: string;
  confidence_threshold?: number | null;
  selected_candidates?: Array<{sample_id: string; candidate_id: string}>;
  member_candidate_id?: string | null;
  points: SegmentationPoint[];
  box: [number, number, number, number] | null;
};

export type SegmentationCorrection = {
  member_candidate_id?: string | null;
  object_id?: number;
  frame_index: number;
  target: SegmentationTarget;
  operation: "add" | "erase";
  radius: number;
  points: Array<{ x: number; y: number }>;
};

export type SegmentationSpec = {
  dataset_id: string;
  fingerprint: string | null;
  frame_token?: string;
  episode_index: number;
  video_key: string;
  mode: "replace_background" | "protect_foreground" | "object_selection";
  render_mode?: "black" | "image";
  camera_mode?: "fixed" | "wrist";
  background_base64: string;
  manual_regions?: SegmentationManualRegion[];
  selected_candidate_ids?: string[];
  source_preview_id?: string;
  prompts: SegmentationPrompt[];
  corrections: SegmentationCorrection[];
};

export type SegmentationManualRegion = {
  box?: [number, number, number, number];
  points?: Array<{ x: number; y: number }>;
  frame_index?: number;
};

export type SegmentationCapabilities = {
  configured: boolean;
  model: string;
  message: string;
};

export type SegmentationScope = {
  metadata_revision?: string;
  fingerprint: string;
  frame_token: string;
  video_keys: string[];
  episodes: Array<{ episode_index: number; length: number }>;
};

export type SegmentationPreviewResult = {
  preview_id: string;
  recipe_hash: string;
  fingerprint: string;
  frame_count: number;
  model: string;
  model_provenance?: Record<string, unknown>;
  candidates?: Array<{
    candidate_id: string;
    detection_score?: number | null;
    score_source?: "text_detection" | null;
    manually_refined?: boolean;
    object_id: number;
    sam_object_id: number;
    target: SegmentationTarget;
    frame_index: number;
    area_pixels: number;
    artifact_name: string;
    /** Chosen automatically: the only plausible object for its Instruction. */
    auto_selected?: boolean;
    /** First appeared after the prompt frame (entered or re-entered view). */
    late_track?: boolean;
    reentry?: boolean;
    first_visible_frame?: number;
    visible_ranges?: Array<[number, number]>;
  }>;
  selection_required?: boolean;
  warnings?: string[];
  review_signals?: Array<{
    frame_index: number;
    reason:
      | "empty_mask"
      | "full_frame_mask"
      | "abrupt_area_change"
      | "reentry"
      | "auto_selected"
      | "object_gap";
    candidate_id?: string;
    object_id?: number;
    missing_frames?: number;
  }>;
  /** Frames on which each kept/removed object was actually detected. */
  object_coverage?: Array<{
    object_id: number;
    target: SegmentationTarget;
    visible_frames: number;
    frame_count: number;
    missing_ranges: Array<[number, number]>;
  }>;
  review_blocked?: boolean;
};

export function isSegmentationPreviewResult(
  value: unknown,
): value is SegmentationPreviewResult {
  if (!value || typeof value !== "object") return false;
  const result = value as Record<string, unknown>;
  return (
    typeof result.preview_id === "string" &&
    typeof result.recipe_hash === "string" &&
    typeof result.fingerprint === "string" &&
    typeof result.frame_count === "number" &&
    typeof result.model === "string"
  );
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `요청을 완료하지 못했습니다 (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      if (Array.isArray(body.detail)) {
        const messages = body.detail
          .filter((item): item is { msg: string } =>
            item !== null && typeof item === "object" && typeof item.msg === "string")
          .map((item) => item.msg.replace(/^Value error, /, ""));
        if (messages.length) message = [...new Set(messages)].join("\n");
      }
      if (
        body.detail &&
        typeof body.detail === "object" &&
        "message" in body.detail &&
        typeof body.detail.message === "string"
      ) {
        message = body.detail.message;
      }
    } catch {
      // Keep the status-only fallback for a non-JSON response.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export function getSegmentationCapabilities() {
  return requestJson<SegmentationCapabilities>(
    "/api/v1/segmentation/capabilities",
  );
}

export function getSegmentationScope(datasetId: string) {
  return requestJson<SegmentationScope>(
    `/api/v1/segmentation/datasets/${encodeURIComponent(datasetId)}/scope`,
  );
}

export function segmentationFrameUrl(
  datasetId: string,
  episodeIndex: number,
  videoKey: string,
  frameIndex: number,
  frameToken: string,
) {
  const query = new URLSearchParams({
    episode_index: String(episodeIndex),
    video_key: videoKey,
    frame_index: String(frameIndex),
    frame_token: frameToken,
  });
  return `/api/v1/segmentation/datasets/${encodeURIComponent(datasetId)}/frame?${query}`;
}

export function createSegmentationPreview(
  profileId: string,
  idempotencyKey: string,
  spec: SegmentationSpec,
) {
  const { background_base64, ...maskSpec } = spec;
  const wireSpec =
    spec.render_mode === "black" || (!spec.render_mode && !background_base64)
      ? maskSpec
      : { ...maskSpec, background_base64 };
  return requestJson<Job>("/api/v1/segmentation/previews", {
    method: "POST",
    body: JSON.stringify({
      profile_id: profileId,
      idempotency_key: idempotencyKey,
      spec: wireSpec,
    }),
  });
}

export function segmentationArtifactUrl(previewId: string, artifact: string) {
  return `/api/v1/segmentation/previews/${encodeURIComponent(previewId)}/artifacts/${encodeURIComponent(artifact)}`;
}

export function approveSegmentationPreview(
  previewId: string,
  profileId: string,
  recipeHash: string,
) {
  return requestJson<{ approval_token: string }>(
    `/api/v1/segmentation/previews/${encodeURIComponent(previewId)}/approve`,
    {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId, recipe_hash: recipeHash }),
    },
  );
}

export function exportSegmentationDataset(
  profileId: string,
  idempotencyKey: string,
  previewId: string,
  approvalToken: string,
  outputName: string,
  recomputeStatistics = false,
) {
  return requestJson<Job>("/api/v1/segmentation/exports", {
    method: "POST",
    body: JSON.stringify({
      profile_id: profileId,
      idempotency_key: idempotencyKey,
      preview_id: previewId,
      approval_token: approvalToken,
      output_name: outputName,
      recompute_statistics: recomputeStatistics,
    }),
  });
}

export type SegmentationCameraTemplate = {
  reuse_policy?: "legacy" | "text_by_default";
  source_episode_index?: number | null;
  mode?: SegmentationSpec["mode"];
  video_key: string;
  camera_mode: "fixed" | "wrist";
  prompts: SegmentationPrompt[];
  corrections: SegmentationCorrection[];
  manual_regions: SegmentationManualRegion[];
};

export type SegmentationTemplate = {
  id: string;
  profile_id: string;
  name: string;
  dataset_id: string;
  fingerprint: string | null;
  metadata_revision?: string;
  cameras: SegmentationCameraTemplate[];
  created_at: string;
  updated_at: string;
};

export type SegmentationBatchItem = {
  id: string;
  episode_index: number;
  video_key: string;
  preview_id: string | null;
  job_status: string;
  review_status: "pending" | "approved";
  dispatch_error?: string | null;
  error_code?: string | null;
  result?: SegmentationPreviewResult | null;
  spec?: SegmentationSpec;
  started_at?: string | null;
  finished_at?: string | null;
  progress?: { overall?: number; elapsed_seconds?: number } | null;
};

export type SegmentationBatch = {
  id: string;
  dataset_id: string;
  fingerprint: string;
  template_id: string;
  items: SegmentationBatchItem[];
  ready_to_export: boolean;
  created_at?: string;
  /** Objects that cannot be carried to other episodes by text re-detection. */
  warnings?: string[];
};

export function listSegmentationTemplates(profileId: string) {
  return requestJson<{ templates: SegmentationTemplate[] }>(
    `/api/v1/segmentation/templates?${new URLSearchParams({ profile_id: profileId })}`,
  );
}

export function saveSegmentationTemplate(
  input: Omit<SegmentationTemplate, "id" | "created_at" | "updated_at">,
  id?: string,
) {
  return requestJson<SegmentationTemplate>(
    `/api/v1/segmentation/templates${id ? `/${encodeURIComponent(id)}` : ""}`,
    { method: id ? "PATCH" : "POST", body: JSON.stringify(input) },
  );
}

export function createSegmentationBatch(input: {
  profile_id: string;
  idempotency_key: string;
  template_id: string;
  dataset_id: string;
  fingerprint: string;
  episode_indices: number[];
  video_keys: string[];
  same_camera_setup_confirmed: boolean;
}) {
  return requestJson<SegmentationBatch>("/api/v1/segmentation/batches", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function listSegmentationBatches(profileId: string) {
  return requestJson<{ batches: SegmentationBatch[] }>(
    `/api/v1/segmentation/batches?${new URLSearchParams({ profile_id: profileId })}`,
  );
}

export function getSegmentationBatch(id: string, profileId: string) {
  return requestJson<SegmentationBatch>(
    `/api/v1/segmentation/batches/${encodeURIComponent(id)}?${new URLSearchParams({ profile_id: profileId })}`,
  );
}

export function retrySegmentationBatch(id: string, profileId: string) {
  return requestJson<SegmentationBatch>(
    `/api/v1/segmentation/batches/${encodeURIComponent(id)}/retry`,
    { method: "POST", body: JSON.stringify({ profile_id: profileId }) },
  );
}

export function attachSegmentationBatchPreview(
  batchId: string,
  itemId: string,
  profileId: string,
  previewId: string,
) {
  return requestJson<SegmentationBatch>(
    `/api/v1/segmentation/batches/${encodeURIComponent(batchId)}/items/${encodeURIComponent(itemId)}/preview`,
    {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId, preview_id: previewId }),
    },
  );
}

export function approveSegmentationBatchItem(
  batchId: string,
  itemId: string,
  profileId: string,
  recipeHash: string,
) {
  return requestJson<{ batch: SegmentationBatch; approval_token: string }>(
    `/api/v1/segmentation/batches/${encodeURIComponent(batchId)}/items/${encodeURIComponent(itemId)}/approve`,
    {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId, recipe_hash: recipeHash }),
    },
  );
}

export function invalidateSegmentationBatchItem(
  batchId: string,
  itemId: string,
  profileId: string,
) {
  return requestJson<SegmentationBatch>(
    `/api/v1/segmentation/batches/${encodeURIComponent(batchId)}/items/${encodeURIComponent(itemId)}/invalidate`,
    { method: "POST", body: JSON.stringify({ profile_id: profileId }) },
  );
}

export function getSegmentationPreview(previewId: string, profileId: string) {
  return requestJson<{
    spec: SegmentationSpec;
    result: SegmentationPreviewResult;
  }>(
    `/api/v1/segmentation/previews/${encodeURIComponent(previewId)}?${new URLSearchParams({ profile_id: profileId })}`,
  );
}

export function exportSegmentationBatch(
  batchId: string,
  profileId: string,
  idempotencyKey: string,
  outputName: string,
  recomputeStatistics = false,
) {
  return requestJson<Job>(
    `/api/v1/segmentation/batches/${encodeURIComponent(batchId)}/exports`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        idempotency_key: idempotencyKey,
        output_name: outputName,
        recompute_statistics: recomputeStatistics,
      }),
    },
  );
}

export type SegmentationCatalog = {
  metadata_revision: string;
  total_episodes: number;
  video_keys: string[];
  fps: number;
};
export type SegmentationSelection = {
  metadata_revision: string;
  frame_token: string;
  episode_index: number;
  video_key: string;
  length: number;
  fps: number;
};
export function getSegmentationCatalog(
  datasetId: string,
  signal?: AbortSignal,
) {
  return requestJson<SegmentationCatalog>(
    `/api/v1/segmentation/datasets/${encodeURIComponent(datasetId)}/catalog`,
    { signal },
  );
}
export function getSegmentationSelection(
  datasetId: string,
  episode: number,
  camera: string,
  signal?: AbortSignal,
) {
  const query = new URLSearchParams({
    episode_index: String(episode),
    video_key: camera,
  });
  return requestJson<SegmentationSelection>(
    `/api/v1/segmentation/datasets/${encodeURIComponent(datasetId)}/selection?${query}`,
    { signal },
  );
}
export type SegmentationSampleResult = {
  selection_signature?: string;
  result_type: "segmentation.sample";
  sample_id: string;
  frame_count: 1;
  source_frame_index: number;
  candidates: NonNullable<SegmentationPreviewResult["candidates"]>;
  artifacts: Record<string, string>;
};
export function createSegmentationSample(
  profileId: string,
  key: string,
  spec: Omit<
    SegmentationSpec,
    "fingerprint" | "source_preview_id" | "selected_candidate_ids"
  > & { frame_token: string; frame_index: number },
) {
  return requestJson<Job>("/api/v1/segmentation/samples", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId, idempotency_key: key, spec }),
  });
}
export function sampleArtifactUrl(
  sampleId: string,
  name: string,
  profileId: string,
) {
  return `/api/v1/segmentation/samples/${encodeURIComponent(sampleId)}/artifacts/${encodeURIComponent(name)}?profile_id=${encodeURIComponent(profileId)}`;
}

export function prepareSegmentationBatch(
  input: Omit<Parameters<typeof createSegmentationBatch>[0], "fingerprint"> & {
    metadata_revision: string;
  },
) {
  return requestJson<Job>("/api/v1/segmentation/batches/prepare", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export type ConfirmedSegmentationObject = {
  object_id: number;
  name: string;
  target: SegmentationTarget;
  prompts: SegmentationPrompt[];
  corrections: SegmentationCorrection[];
};
export type SegmentationWorkspace = {
  revision: number;
  metadata_revision: string;
  stale: boolean;
  objects: ConfirmedSegmentationObject[];
};
export type SegmentationWorkspaceScope = {
  profile_id: string;
  dataset_id: string;
  episode_index: number;
  video_key: string;
};
export function getSegmentationWorkspace(scope: SegmentationWorkspaceScope, signal?: AbortSignal) {
  const query = new URLSearchParams({...scope, episode_index: String(scope.episode_index)});
  return requestJson<SegmentationWorkspace>(`/api/v1/segmentation/workspace?${query}`, {signal});
}
export function saveSegmentationWorkspace(scope: SegmentationWorkspaceScope, workspace: SegmentationWorkspace, objects: ConfirmedSegmentationObject[]) {
  return requestJson<SegmentationWorkspace>("/api/v1/segmentation/workspace", {
    method: "PUT",
    body: JSON.stringify({...scope, expected_revision: workspace.revision, metadata_revision: workspace.metadata_revision, objects}),
  });
}

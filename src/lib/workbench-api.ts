import type { LanguageAtom } from "@/types/language.types";

export type Profile = {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
};

export type DatasetReadiness =
  | "ready"
  | "incomplete"
  | "unsupported"
  | "invalid";

export type DatasetSummary = {
  id: string;
  storage_area: "raw" | "derived";
  relative_path: string;
  name: string;
  codebase_version: string | null;
  readiness: DatasetReadiness;
  robot_type: string | null;
  total_episodes: number | null;
  total_frames: number | null;
  total_tasks: number | null;
  fps: number | null;
  fingerprint: string;
  scan_error: string | null;
  first_seen_at: string;
  last_seen_at: string;
  available: boolean;
};

export type DatasetTrashState =
  | "moving"
  | "trashed"
  | "restoring"
  | "recovery_required"
  | "purge_queued"
  | "purging"
  | "purged";

export type DatasetTrashEntry = {
  dataset: DatasetSummary;
  original_relative_path: string;
  trashed_at: string | null;
  state: DatasetTrashState;
  requested_by_profile_id: string;
};

export type EpisodeFlags = {
  dataset_id: string;
  dataset_fingerprint: string;
  profile_id: string;
  revision: number;
  episode_indices: number[];
  updated_at: string | null;
};

export type EpisodeFlagChange = {
  episode_index: number;
  flagged: boolean;
};

export type EpisodeAnnotations = {
  dataset_id: string;
  dataset_fingerprint: string;
  profile_id: string;
  episode_index: number;
  revision: number;
  task_override: string | null;
  atoms: LanguageAtom[];
  updated_at: string | null;
};

export type CurationSelectionMode = "all" | "flagged" | "unflagged";
export type CurationOperation =
  | "subset"
  | "delete_flagged"
  | "train_eval_split";

export type TrimConfig = {
  enabled: boolean;
  recompute_statistics?: boolean;
  method: "legacy_motion" | "stationary";
  state_epsilon: number;
  threshold: number;
  hold_time_s: number;
  margin_s: number;
  start_hold_time_s?: number | null;
  end_hold_time_s?: number | null;
  start_margin_s?: number | null;
  end_margin_s?: number | null;
  dimensions: string[];
  episode_overrides: Record<string, { start_frame: number; end_frame: number }>;
};

export type RelativeActionConfig = {
  enabled: boolean;
  dimensions: string[];
  chunk_size: number;
};

export type TrainEvalSplitConfig = {
  method: "flagged" | "random";
  eval_percent: number;
  seed: number;
};

export type CurationRecipe = {
  id: string;
  dataset_id: string;
  dataset_fingerprint: string;
  profile_id: string;
  name: string;
  selection_mode: CurationSelectionMode;
  operation: CurationOperation;
  trim_config: TrimConfig;
  include_annotations: boolean;
  relative_action: RelativeActionConfig;
  split_config: TrainEvalSplitConfig;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
};

export type CurationRecipeSnapshot = {
  id: string;
  recipe_id: string;
  dataset_id: string;
  dataset_fingerprint: string;
  profile_id: string;
  recipe_name: string;
  selection_mode: CurationSelectionMode;
  operation: CurationOperation;
  trim_config: TrimConfig;
  include_annotations: boolean;
  relative_action: RelativeActionConfig;
  split_config: TrainEvalSplitConfig;
  annotation_episode_indices: number[];
  flag_revision: number;
  flagged_episode_indices: number[];
  selected_episode_indices: number[];
  eval_episode_indices: number[];
  created_at: string;
};

export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "interrupted";

export type Job = {
  progress?: {
    stage: string;
    completed: number;
    total: number;
    unit: string;
    elapsed_seconds: number;
    current_item?: string;
    output_name?: string;
    /** Whole-job fraction 0..1 when the job reports a weighted plan. */
    overall?: number;
  } | null;
  id: string;
  kind: string;
  queue_name: string;
  status: JobStatus;
  profile_id: string;
  payload: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error_code: string | null;
  error_message: string | null;
  idempotency_key: string;
  rq_job_id: string | null;
  created_at: string;
  enqueued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  cancellation_requested: boolean;
  cancellation_requested_at: string | null;
  cancellation_guarded_at: string | null;
  validation_job_id?: string | null;
  wait_reason?: string | null;
  queue_position?: number | null;
  profile_name?: string | null;
};

export type JobEvent = {
  sequence: number;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
};

export type ValidationMode = "quick" | "full" | "export_gate";

export type ValidationCheckStatus =
  | "pending"
  | "running"
  | "passed"
  | "warning"
  | "failed"
  | "skipped";

export type ValidationCheck = {
  id: string;
  label: string;
  status: ValidationCheckStatus;
  failures: number;
  warnings: number;
  detail?: string;
};

export type ValidationProgressSnapshot = {
  stage: "metadata" | "data" | "video" | "statistics" | "complete";
  completed: number;
  total: number;
  decoded_frames: number;
  elapsed_seconds: number;
  checks: ValidationCheck[];
};

export type ValidationResult = {
  validator_policy?: string;
  mode: ValidationMode;
  passed: boolean;
  checked_episodes: number;
  total_episodes: number;
  checked_frames: number;
  failures: number;
  warnings: number;
  checks?: ValidationCheck[];
  issues: Array<{
    severity: "WARN" | "FAIL";
    code: string;
    message: string;
    episode_index?: number;
  }>;
};

export type ValidationRun = {
  job_id: string;
  dataset_id: string;
  dataset_fingerprint: string;
  mode: ValidationMode;
  created_at: string;
  started_at?: string | null;
  status: JobStatus;
  progress?: ValidationProgressSnapshot | null;
  result: ValidationResult | null;
  error_code: string | null;
  error_message: string | null;
  finished_at: string | null;
};

export type SystemHealth = {
  ok: boolean;
  service: string;
  database: "ok" | "error";
  queue: "ok" | "error";
  schema_versions: number[];
};

export type SystemResources = {
  enabled: boolean;
  healthy: boolean;
  sampled_at: string | null;
  cpu_available_percent: number | null;
  memory_available_percent: number | null;
  iowait_percent: number | null;
  active_jobs: number;
  max_parallel_jobs: number;
  queued_jobs: number;
  reserved_worker_slots: number;
  managed_running_jobs: number;
  minimum_available_percent: number;
  decision:
    | "ready"
    | "no_jobs"
    | "resource_wait"
    | "limit"
    | "warming_up"
    | "unavailable";
  message: string;
};

export type HuggingFaceDatasetStatus =
  | "not_downloaded"
  | "queued"
  | "downloading"
  | "ready"
  | "update_available"
  | "incomplete"
  | "validation_failed";

export type HuggingFaceDataset = {
  repo_id: string;
  name: string;
  private: boolean;
  gated: boolean;
  downloads: number;
  likes: number;
  last_modified: string | null;
  latest_commit_sha: string;
  current_commit_sha: string | null;
  tags: string[];
  status: HuggingFaceDatasetStatus;
};

export type HuggingFaceRevision = {
  name: string;
  kind: "branch" | "tag";
  commit_sha: string;
};

export class WorkbenchApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "WorkbenchApiError";
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (!response.ok) {
    let message = `요청을 완료하지 못했습니다 (${response.status})`;
    let code: string | undefined;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      if (body.detail && typeof body.detail === "object") {
        if (
          "message" in body.detail &&
          typeof body.detail.message === "string"
        ) {
          message = body.detail.message;
        }
        if ("code" in body.detail && typeof body.detail.code === "string") {
          code = body.detail.code;
        }
      }
    } catch {
      // Keep the safe status-only fallback for non-JSON failures.
    }
    throw new WorkbenchApiError(message, response.status, code);
  }
  if (response.status === 204) return undefined as T;
  const result = (await response.json()) as T;
  if (
    init?.method === "POST" &&
    typeof window !== "undefined" &&
    result &&
    typeof result === "object" &&
    "id" in result &&
    "kind" in result &&
    "status" in result &&
    "profile_id" in result
  ) {
    window.dispatchEvent(
      new CustomEvent("datasetui:job-accepted", { detail: result }),
    );
  }
  return result;
}

export function listProfiles(): Promise<Profile[]> {
  return requestJson("/api/v1/profiles");
}

export function createProfile(name: string): Promise<Profile> {
  return requestJson("/api/v1/profiles", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function updateProfile(
  profileId: string,
  change: { name?: string; archived?: boolean },
): Promise<Profile> {
  return requestJson(`/api/v1/profiles/${encodeURIComponent(profileId)}`, {
    method: "PATCH",
    body: JSON.stringify(change),
  });
}

export type DatasetSort = "newest" | "oldest" | "name_asc" | "name_desc";

export function listDatasets(options?: {
  sort?: DatasetSort;
  query?: string;
  storageArea?: "raw" | "derived";
  readiness?: DatasetReadiness;
  offset?: number;
  limit?: number;
  signal?: AbortSignal;
}): Promise<DatasetSummary[]> {
  const query = new URLSearchParams({ limit: String(options?.limit ?? 500) });
  if (options?.sort) query.set("sort", options.sort);
  if (options?.query?.trim()) query.set("q", options.query.trim());
  if (options?.storageArea) query.set("storage_area", options.storageArea);
  if (options?.readiness) query.set("readiness", options.readiness);
  if (options?.offset) query.set("offset", String(options.offset));
  return requestJson(`/api/v1/datasets?${query}`, { signal: options?.signal });
}

export function getDataset(datasetId: string): Promise<DatasetSummary> {
  return requestJson(`/api/v1/datasets/${encodeURIComponent(datasetId)}`);
}

export function getDatasetInfo(datasetId: string): Promise<unknown> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/files/meta/info.json`,
  );
}

export function renameDataset(
  datasetId: string,
  name: string,
  expectedName: string,
): Promise<DatasetSummary> {
  return requestJson(`/api/v1/datasets/${encodeURIComponent(datasetId)}`, {
    method: "PATCH",
    body: JSON.stringify({ name, expected_name: expectedName }),
  });
}

export function trashDataset(
  datasetId: string,
  profileId: string,
  expectedName: string,
  expectedFingerprint: string,
): Promise<DatasetTrashEntry> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/trash`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        expected_name: expectedName,
        expected_fingerprint: expectedFingerprint,
      }),
    },
  );
}

export function listDatasetTrash(
  profileId: string,
  signal?: AbortSignal,
): Promise<DatasetTrashEntry[]> {
  const query = new URLSearchParams({ profile_id: profileId, limit: "500" });
  return requestJson(`/api/v1/dataset-trash?${query}`, { signal });
}

export function restoreDatasetFromTrash(
  trashId: string,
  profileId: string,
  expectedFingerprint: string,
): Promise<DatasetSummary> {
  return requestJson(
    `/api/v1/dataset-trash/${encodeURIComponent(trashId)}/restore`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        expected_fingerprint: expectedFingerprint,
      }),
    },
  );
}

export function mergeDatasets(
  profileId: string,
  datasetIds: string[],
  outputName: string,
  robotType: string,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson("/api/v1/datasets/merge", {
    method: "POST",
    body: JSON.stringify({
      profile_id: profileId,
      dataset_ids: datasetIds,
      output_name: outputName,
      robot_type: robotType,
      idempotency_key: idempotencyKey,
    }),
  });
}

export function validateDataset(
  datasetId: string,
  profileId: string,
  mode: ValidationMode,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/validations`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        mode,
        idempotency_key: idempotencyKey,
      }),
    },
  );
}

export function listDatasetValidations(
  datasetId: string,
): Promise<ValidationRun[]> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/validations`,
  );
}

export function deleteDatasetValidation(
  datasetId: string,
  jobId: string,
): Promise<void> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/validations/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export function convertDatasetToV21(
  datasetId: string,
  profileId: string,
  outputName: string,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/conversions/v2.1`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        output_name: outputName,
        idempotency_key: idempotencyKey,
      }),
    },
  );
}

export function exportDatasetToNas(
  datasetId: string,
  profileId: string,
  outputName: string,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/deliveries/nas`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        output_name: outputName,
        idempotency_key: idempotencyKey,
      }),
    },
  );
}

export function uploadDatasetToHuggingFace(
  datasetId: string,
  profileId: string,
  repoName: string,
  visibility: "private" | "public",
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/deliveries/huggingface`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        repo_name: repoName,
        visibility,
        idempotency_key: idempotencyKey,
      }),
    },
  );
}

export type PcTransferTarget = {
  host: string;
  port: number;
  username: string;
  destination: string;
};

export function copyDatasetToPcWithKey(
  datasetId: string,
  profileId: string,
  target: PcTransferTarget,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/deliveries/pc/key`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        ...target,
        idempotency_key: idempotencyKey,
      }),
    },
  );
}

export function copyDatasetToPcWithPassword(
  datasetId: string,
  profileId: string,
  target: PcTransferTarget,
  password: string,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/deliveries/pc/password`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        ...target,
        password,
        idempotency_key: idempotencyKey,
      }),
    },
  );
}

export function getEpisodeFlags(
  datasetId: string,
  profileId: string,
): Promise<EpisodeFlags> {
  const query = new URLSearchParams({ profile_id: profileId });
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/flags?${query}`,
  );
}

export function updateEpisodeFlags(
  datasetId: string,
  profileId: string,
  expectedRevision: number,
  changes: EpisodeFlagChange[],
): Promise<EpisodeFlags> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/flags`,
    {
      method: "PATCH",
      body: JSON.stringify({
        profile_id: profileId,
        expected_revision: expectedRevision,
        changes,
      }),
    },
  );
}

export function getEpisodeAnnotations(
  datasetId: string,
  profileId: string,
  episodeIndex: number,
): Promise<EpisodeAnnotations> {
  const query = new URLSearchParams({ profile_id: profileId });
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/annotations/${episodeIndex}?${query}`,
  );
}

export function replaceEpisodeAnnotations(
  datasetId: string,
  profileId: string,
  episodeIndex: number,
  expectedRevision: number,
  taskOverride: string | null,
  atoms: LanguageAtom[],
): Promise<EpisodeAnnotations> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/annotations/${episodeIndex}`,
    {
      method: "PUT",
      body: JSON.stringify({
        profile_id: profileId,
        expected_revision: expectedRevision,
        task_override: taskOverride,
        atoms,
      }),
    },
  );
}

export function listCurationRecipes(
  datasetId: string,
  profileId: string,
): Promise<CurationRecipe[]> {
  const query = new URLSearchParams({ profile_id: profileId });
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/recipes?${query}`,
  );
}

export function createCurationRecipe(
  datasetId: string,
  profileId: string,
  name: string,
  selectionMode: CurationSelectionMode,
  operation: CurationOperation,
  trimConfig: TrimConfig,
  includeAnnotations = false,
  relativeAction: RelativeActionConfig = {
    enabled: false,
    dimensions: [],
    chunk_size: 50,
  },
  splitConfig: TrainEvalSplitConfig = {
    method: "flagged",
    eval_percent: 20,
    seed: 0,
  },
): Promise<CurationRecipe> {
  return requestJson(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/recipes`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        name,
        selection_mode: selectionMode,
        operation,
        trim_config: trimConfig,
        include_annotations: includeAnnotations,
        relative_action: relativeAction,
        split_config: splitConfig,
      }),
    },
  );
}

export function updateCurationRecipe(
  recipeId: string,
  profileId: string,
  change: {
    name?: string;
    selection_mode?: CurationSelectionMode;
    operation?: CurationOperation;
    trim_config?: TrimConfig;
    include_annotations?: boolean;
    relative_action?: RelativeActionConfig;
    split_config?: TrainEvalSplitConfig;
    archived?: boolean;
  },
): Promise<CurationRecipe> {
  return requestJson(`/api/v1/recipes/${encodeURIComponent(recipeId)}`, {
    method: "PATCH",
    body: JSON.stringify({ profile_id: profileId, ...change }),
  });
}

export function runCurationRecipe(
  recipeId: string,
  profileId: string,
  outputName: string,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(`/api/v1/recipes/${encodeURIComponent(recipeId)}/runs`, {
    method: "POST",
    body: JSON.stringify({
      profile_id: profileId,
      output_name: outputName,
      idempotency_key: idempotencyKey,
    }),
  });
}

export function snapshotCurationRecipe(
  recipeId: string,
  profileId: string,
): Promise<CurationRecipeSnapshot> {
  return requestJson(
    `/api/v1/recipes/${encodeURIComponent(recipeId)}/snapshots`,
    {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    },
  );
}

export function listJobs(
  profileId?: string,
  signal?: AbortSignal,
): Promise<Job[]> {
  const query = new URLSearchParams({ limit: "200" });
  if (profileId) query.set("profile_id", profileId);
  return requestJson(`/api/v1/jobs?${query}`, { signal });
}

export async function listActiveTeamJobs(signal?: AbortSignal): Promise<Job[]> {
  const items = new Map<string, Job>();
  for (let offset = 0; ; offset += 200) {
    const page = await requestJson<Job[]>(
      `/api/v1/jobs?active_only=true&limit=200&offset=${offset}`,
      { signal },
    );
    for (const job of page) items.set(job.id, job);
    if (page.length < 200) return [...items.values()];
  }
}

export function deleteHuggingFaceDataset(
  profileId: string,
  dataset: HuggingFaceDataset,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson(
    `/api/v1/hf/datasets/${encodeURIComponent(dataset.name)}/delete`,
    {
      method: "POST",
      body: JSON.stringify({
        profile_id: profileId,
        expected_repo_id: dataset.repo_id,
        expected_commit_sha: dataset.latest_commit_sha,
        confirmed: true,
        idempotency_key: idempotencyKey,
      }),
    },
  );
}

export function emptyDatasetTrash(
  profileId: string,
  entries: DatasetTrashEntry[],
  idempotencyKey: string,
): Promise<Job> {
  return requestJson("/api/v1/dataset-trash/empty", {
    method: "POST",
    body: JSON.stringify({
      profile_id: profileId,
      confirmed: true,
      idempotency_key: idempotencyKey,
      items: entries.map((entry) => ({
        dataset_id: entry.dataset.id,
        expected_fingerprint: entry.dataset.fingerprint,
      })),
    }),
  });
}

export function getJob(jobId: string): Promise<Job> {
  return requestJson(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
}

export function cancelJob(jobId: string, profileId: string): Promise<Job> {
  return requestJson(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId }),
  });
}

export function listMergeJobs(
  profileId: string,
  signal?: AbortSignal,
): Promise<Job[]> {
  const query = new URLSearchParams({
    profile_id: profileId,
    kind: "datasets.merge",
    limit: "50",
  });
  return requestJson(`/api/v1/jobs?${query}`, { signal });
}

export function listJobEvents(jobId: string): Promise<JobEvent[]> {
  return requestJson(`/api/v1/jobs/${encodeURIComponent(jobId)}/events`);
}

export function getSystemHealth(): Promise<SystemHealth> {
  return requestJson("/api/v1/system/health");
}

export function getSystemResources(
  signal?: AbortSignal,
): Promise<SystemResources> {
  return requestJson("/api/v1/system/resources", { signal });
}

export function getDeliveryCapabilities(): Promise<{
  hf_upload_configured: boolean;
  hf_namespace: string;
  hf_delete_configured: boolean;
  pc_password_configured: boolean;
}> {
  return requestJson("/api/v1/delivery/capabilities");
}

export function listHuggingFaceDatasets(
  query = "",
  signal?: AbortSignal,
): Promise<HuggingFaceDataset[]> {
  const parameters = new URLSearchParams({ limit: "200" });
  if (query.trim()) parameters.set("q", query.trim());
  return requestJson(`/api/v1/hf/datasets?${parameters}`, { signal });
}

export function listHuggingFaceRevisions(
  datasetName: string,
): Promise<HuggingFaceRevision[]> {
  return requestJson(
    `/api/v1/hf/datasets/${encodeURIComponent(datasetName)}/revisions`,
  );
}

export function importHuggingFaceDataset(
  profileId: string,
  datasetName: string,
  revision: HuggingFaceRevision,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson("/api/v1/hf/imports", {
    method: "POST",
    body: JSON.stringify({
      profile_id: profileId,
      dataset_name: datasetName,
      requested_revision: revision.name,
      commit_sha: revision.commit_sha,
      idempotency_key: idempotencyKey,
    }),
  });
}

export function refreshLibrary(
  profileId: string,
  idempotencyKey: string,
): Promise<Job> {
  return requestJson("/api/v1/jobs", {
    method: "POST",
    body: JSON.stringify({
      kind: "datasets.scan",
      profile_id: profileId,
      payload: { storage_areas: ["raw", "derived"] },
      idempotency_key: idempotencyKey,
    }),
  });
}

export function isActiveJob(status: JobStatus): boolean {
  return status === "queued" || status === "running";
}

export function publicJobError(errorCode: string | null): string {
  const operationErrors: Record<string, string> = {
    delivery_validation_failed:
      "선행 전체 검사가 실패하거나 취소되어 전달하지 않았습니다. 검사 보고서를 확인하세요.",
    credential_unavailable:
      "비밀번호가 만료되거나 임시 저장소를 사용할 수 없습니다. 다시 입력해 새 작업을 요청하세요.",
    hf_delete_denied: "Hugging Face 저장소 삭제 권한을 확인하세요.",
    hf_delete_uncertain:
      "원격 삭제 결과가 불명확합니다. HF에서 확인하세요. 자동 재시도하지 않습니다.",
    confirmation_mismatch:
      "확인 후 데이터셋이 변경되었습니다. 목록을 새로 확인하고 다시 요청하세요.",
  };
  if (errorCode && operationErrors[errorCode])
    return operationErrors[errorCode];
  if (errorCode === "export_gate_required") {
    return "현재 파일 내용에 대한 내보내기 검사가 필요합니다. 검증 화면에서 내보내기 검사를 다시 실행하세요.";
  }
  if (errorCode === "external_outcome_uncertain") {
    return "외부 전송 결과를 확인할 수 없습니다. 중복 실행하지 말고 대상 저장소 상태를 확인해 주세요.";
  }
  if (errorCode === "validation_interrupted") {
    return "검사 작업자가 중단되었습니다. 자동 재시도하지 않습니다. 원인을 확인한 뒤 다시 검사해 주세요.";
  }
  if (errorCode === "job_timeout") {
    return "검사 또는 작업이 제한 시간을 초과했습니다. 자동 재시도하지 않습니다.";
  }
  if (errorCode === "external_outcome_uncertain") {
    return "전송 중 연결이 끊겨 외부 저장 결과를 확정할 수 없습니다. 대상 위치를 확인한 뒤 다시 시도하세요. 자동 재시도하지 않습니다.";
  }
  if (errorCode === "hf_cleanup_required") {
    return "Hugging Face 임시 저장소를 정리하지 못했습니다. 대상 저장소를 확인한 뒤 정리하세요.";
  }
  if (errorCode === "segmentation_unavailable") {
    return "SAM GPU 작업자의 가중치·SHA256·CUDA 설정을 확인하세요. 임의의 마스크로 대체하지 않습니다.";
  }
  if (errorCode === "segmentation_failed") {
    return "SAM 결과의 프레임 또는 마스크가 완전하지 않아 중단했습니다. 입력 범위와 GPU 작업자 기록을 확인하세요.";
  }
  if (errorCode === "queue_unavailable") {
    return "작업 대기열에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.";
  }
  if (errorCode === "storage_unavailable") {
    return "공유 저장소를 읽을 수 없습니다. 관리자에게 확인해 주세요.";
  }
  if (errorCode === "hf_unavailable") {
    return "Hugging Face에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.";
  }
  if (errorCode === "hf_source_not_found") {
    return "선택한 revision을 찾을 수 없습니다. 목록을 새로 확인해 주세요.";
  }
  if (errorCode === "hf_import_too_large") {
    return "허용된 가져오기 용량을 초과했습니다. 관리자에게 확인해 주세요.";
  }
  if (errorCode === "hf_revision_conflict") {
    return "같은 revision의 저장 내용이 달라 안전하게 중단했습니다.";
  }
  if (errorCode === "hf_validation_failed") {
    return "LeRobot 데이터셋 구조 확인에 실패했습니다.";
  }
  if (errorCode === "worker_lost") {
    return "작업자가 중단되어 자동으로 다시 시도하고 있습니다.";
  }
  if (errorCode === "source_revision_changed") {
    return "원본 데이터셋 revision이 바뀌어 안전하게 중단했습니다.";
  }
  if (errorCode === "curation_failed") {
    return "데이터셋을 만드는 중 구조 또는 영상 검증에 실패했습니다.";
  }
  return "작업을 완료하지 못했습니다. 관리자에게 확인해 주세요.";
}

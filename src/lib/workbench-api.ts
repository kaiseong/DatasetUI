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

export type CurationSelectionMode = "all" | "flagged" | "unflagged";
export type CurationOperation =
  | "subset"
  | "delete_flagged"
  | "train_eval_split";

export type TrimConfig = {
  enabled: boolean;
  threshold: number;
  hold_time_s: number;
  margin_s: number;
  dimensions: string[];
  episode_overrides: Record<string, { start_frame: number; end_frame: number }>;
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
  flag_revision: number;
  flagged_episode_indices: number[];
  selected_episode_indices: number[];
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
};

export type JobEvent = {
  sequence: number;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
};

export type SystemHealth = {
  ok: boolean;
  service: string;
  database: "ok" | "error";
  queue: "ok" | "error";
  schema_versions: number[];
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
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      if (
        body.detail &&
        typeof body.detail === "object" &&
        "message" in body.detail &&
        typeof body.detail.message === "string"
      ) {
        message = body.detail.message;
      }
    } catch {
      // Keep the safe status-only fallback for non-JSON failures.
    }
    throw new WorkbenchApiError(message, response.status);
  }
  return (await response.json()) as T;
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

export function listDatasets(): Promise<DatasetSummary[]> {
  return requestJson("/api/v1/datasets?limit=500");
}

export function getDataset(datasetId: string): Promise<DatasetSummary> {
  return requestJson(`/api/v1/datasets/${encodeURIComponent(datasetId)}`);
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

export function getJob(jobId: string): Promise<Job> {
  return requestJson(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
}

export function listJobEvents(jobId: string): Promise<JobEvent[]> {
  return requestJson(`/api/v1/jobs/${encodeURIComponent(jobId)}/events`);
}

export function getSystemHealth(): Promise<SystemHealth> {
  return requestJson("/api/v1/system/health");
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

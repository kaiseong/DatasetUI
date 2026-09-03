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
  return "작업을 완료하지 못했습니다. 관리자에게 확인해 주세요.";
}

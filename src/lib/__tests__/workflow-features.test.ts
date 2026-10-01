import { afterEach, expect, mock, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { JobProgress } from "../../components/workbench/job-progress";
import {
  listDatasets,
  listActiveTeamJobs,
  deleteHuggingFaceDataset,
  emptyDatasetTrash,
  copyDatasetToPcWithPassword,
  type Job,
  type HuggingFaceDataset,
  type DatasetTrashEntry,
} from "../workbench-api";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const job = {
  id: "job",
  kind: "datasets.export_nas",
  queue_name: "io",
  status: "queued",
  profile_id: "owner",
  profile_name: "팀원",
  payload: {},
  result: null,
  progress: null,
  error_code: null,
  error_message: null,
  validation_job_id: "check",
  wait_reason: "validation",
  queue_position: null,
  cancellation_requested: false,
  cancellation_guarded_at: null,
} as Job;

test("delivery wait is explicitly a prerequisite, never a guessed queue rank or percent", () => {
  const html = renderToStaticMarkup(
    createElement(JobProgress, { job, currentProfileId: "owner" }),
  );
  expect(html).toContain("전체 검사 결과를 기다리는 중");
  expect(html).toContain("팀원");
  expect(html).toContain("선행 검사 보고서 확인");
  expect(html).toContain("대기 작업 취소");
  expect(html).not.toContain("aria-valuenow");
});

test("measured per-queue rank and owner-only cancellation are visible", () => {
  const html = renderToStaticMarkup(
    createElement(JobProgress, {
      job: { ...job, wait_reason: null, queue_name: "cpu", queue_position: 4 },
      currentProfileId: "another",
    }),
  );
  expect(html).toContain("CPU 대기열 4번째");
  expect(html).not.toContain("대기 작업 취소");
});

test("new jobs have real operation labels and partial purge does not claim complete success", () => {
  for (const kind of [
    "datasets.delivery_preflight",
    "datasets.copy_pc_password",
    "hf.delete",
    "datasets.empty_trash",
  ]) {
    const html = renderToStaticMarkup(
      createElement(JobProgress, { job: { ...job, kind } }),
    );
    expect(html).not.toContain("지원되지 않는 작업");
  }
  const html = renderToStaticMarkup(
    createElement(JobProgress, {
      job: {
        ...job,
        kind: "datasets.empty_trash",
        status: "succeeded",
        result: {
          deleted: ["a"],
          failed: [{ dataset_id: "b" }],
          skipped: [],
          partial_failure: true,
        },
      },
    }),
  );
  expect(html).toContain("일부 실패");
  expect(html).toContain("실패 1");
});

test("library sends server-side sort, search, filters and pagination", async () => {
  const fetchMock = mock<
    (_url: unknown, _init?: RequestInit) => Promise<Response>
  >(async () => Response.json([]));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  await listDatasets({
    sort: "newest",
    query: " robot ",
    storageArea: "derived",
    readiness: "ready",
    offset: 500,
    limit: 100,
  });
  const url = new URL(
    String(fetchMock.mock.calls[0]?.[0]),
    "https://test.local",
  );
  expect(Object.fromEntries(url.searchParams)).toEqual({
    sort: "newest",
    q: "robot",
    storage_area: "derived",
    readiness: "ready",
    offset: "500",
    limit: "100",
  });
});

test("all active team jobs load beyond 200 without a selected-profile filter", async () => {
  const urls: string[] = [];
  globalThis.fetch = mock(async (url: unknown) => {
    urls.push(String(url));
    return Response.json(
      urls.length === 1
        ? Array.from({ length: 200 }, (_, i) => ({ ...job, id: String(i) }))
        : [{ ...job, id: "older-active" }],
    );
  }) as unknown as typeof fetch;
  const jobs = await listActiveTeamJobs();
  expect(jobs).toHaveLength(201);
  expect(jobs.at(-1)?.id).toBe("older-active");
  expect(urls[1]).toContain("offset=200");
  expect(
    urls.every(
      (url) => url.includes("active_only=true") && !url.includes("profile_id"),
    ),
  ).toBe(true);
});

test("remote delete binds explicit confirmation and the displayed revision", async () => {
  const fetchMock = mock<
    (_url: unknown, _init?: RequestInit) => Promise<Response>
  >(async () => Response.json(job, { status: 202 }));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  await deleteHuggingFaceDataset(
    "owner",
    {
      name: "sample",
      repo_id: "rainbowrobotics/sample",
      latest_commit_sha: "a".repeat(40),
    } as HuggingFaceDataset,
    "intent",
  );
  const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
  expect(fetchMock.mock.calls[0]?.[0]).toBe(
    "/api/v1/hf/datasets/sample/delete",
  );
  expect(JSON.parse(String(init.body))).toEqual({
    profile_id: "owner",
    expected_repo_id: "rainbowrobotics/sample",
    expected_commit_sha: "a".repeat(40),
    confirmed: true,
    idempotency_key: "intent",
  });
});

test("empty trash sends immutable IDs and fingerprints, not a recursive path", async () => {
  const fetchMock = mock<
    (_url: unknown, _init?: RequestInit) => Promise<Response>
  >(async () => Response.json(job, { status: 202 }));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  await emptyDatasetTrash(
    "owner",
    [{ dataset: { id: "fixed", fingerprint: "fp" } } as DatasetTrashEntry],
    "intent",
  );
  const body = JSON.parse(
    String((fetchMock.mock.calls[0]?.[1] as RequestInit).body),
  );
  expect(body).toEqual({
    profile_id: "owner",
    items: [{ dataset_id: "fixed", expected_fingerprint: "fp" }],
    confirmed: true,
    idempotency_key: "intent",
  });
});

test("password transfer returns an asynchronous job and uses a retry identity", async () => {
  const fetchMock = mock<
    (_url: unknown, _init?: RequestInit) => Promise<Response>
  >(async () => Response.json(job, { status: 202 }));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  expect(
    await copyDatasetToPcWithPassword(
      "dataset",
      "owner",
      {
        host: "192.168.0.51",
        port: 22,
        username: "robot",
        destination: "~/copy",
      },
      "one-use",
      "intent",
    ),
  ).toEqual(job);
  expect(
    JSON.parse(String((fetchMock.mock.calls[0]?.[1] as RequestInit).body))
      .idempotency_key,
  ).toBe("intent");
});

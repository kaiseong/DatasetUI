import { afterEach, describe, expect, mock, test } from "bun:test";
import {
  WorkbenchApiError,
  createCurationRecipe,
  createProfile,
  getEpisodeAnnotations,
  getEpisodeFlags,
  getDataset,
  importHuggingFaceDataset,
  isActiveJob,
  listJobs,
  listProfiles,
  publicJobError,
  refreshLibrary,
  replaceEpisodeAnnotations,
  runCurationRecipe,
  updateEpisodeFlags,
} from "../workbench-api";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("Workbench API client", () => {
  test("uses the same-origin profile endpoint", async () => {
    const fetchMock = mock(async () =>
      Response.json([
        {
          id: "profile-1",
          name: "Researcher",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
          archived_at: null,
        },
      ]),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const profiles = await listProfiles();
    expect(profiles[0]?.name).toBe("Researcher");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/profiles");
  });

  test("creates a profile with a strict JSON body", async () => {
    const fetchMock = mock(async () =>
      Response.json(
        {
          id: "profile-2",
          name: "Kim",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
          archived_at: null,
        },
        { status: 201 },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await createProfile("Kim");
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(init.body).toBe('{"name":"Kim"}');
  });

  test("surfaces a safe API detail message", async () => {
    globalThis.fetch = mock(async () =>
      Response.json(
        { detail: "A profile with this name already exists" },
        { status: 409 },
      ),
    ) as unknown as typeof fetch;

    try {
      await createProfile("Kim");
      throw new Error("request should fail");
    } catch (error) {
      expect(error).toBeInstanceOf(WorkbenchApiError);
      expect((error as WorkbenchApiError).status).toBe(409);
      expect((error as Error).message).toBe(
        "A profile with this name already exists",
      );
    }
  });

  test("filters jobs by the selected profile", async () => {
    const fetchMock = mock(async () => Response.json([]));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await listJobs("profile/id");
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/jobs?limit=200&profile_id=profile%2Fid",
    );
  });

  test("resolves a Viewer dataset by opaque registry ID", async () => {
    const fetchMock = mock(async () =>
      Response.json({ id: "dataset-1", name: "Pick cup" }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getDataset("dataset/id");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/datasets/dataset%2Fid");
  });

  test("loads flags for only the selected profile and opaque dataset", async () => {
    const fetchMock = mock(async () =>
      Response.json({ revision: 0, episode_indices: [] }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getEpisodeFlags("dataset/id", "profile/id");
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/datasets/dataset%2Fid/flags?profile_id=profile%2Fid",
    );
  });

  test("updates flags with optimistic concurrency and no server path", async () => {
    const fetchMock = mock(async () =>
      Response.json({ revision: 4, episode_indices: [2] }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await updateEpisodeFlags("dataset-1", "profile-1", 3, [
      { episode_index: 2, flagged: true },
    ]);
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe("PATCH");
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(body).toEqual({
      profile_id: "profile-1",
      expected_revision: 3,
      changes: [{ episode_index: 2, flagged: true }],
    });
    expect(JSON.stringify(body).toLowerCase()).not.toContain("path");
    expect(JSON.stringify(body).toLowerCase()).not.toContain("password");
  });

  test("creates a strict revision-bound recipe definition", async () => {
    const fetchMock = mock(async () =>
      Response.json(
        { id: "recipe-1", selection_mode: "flagged" },
        { status: 201 },
      ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await createCurationRecipe(
      "dataset-1",
      "profile-1",
      "Review failures",
      "flagged",
      "subset",
      {
        enabled: false,
        threshold: 0.02,
        hold_time_s: 0.5,
        margin_s: 1,
        dimensions: [],
        episode_overrides: {},
      },
    );
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/datasets/dataset-1/recipes",
    );
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      profile_id: "profile-1",
      name: "Review failures",
      selection_mode: "flagged",
      operation: "subset",
      trim_config: {
        enabled: false,
        threshold: 0.02,
        hold_time_s: 0.5,
        margin_s: 1,
        dimensions: [],
        episode_overrides: {},
      },
      include_annotations: false,
    });
  });

  test("loads and replaces an episode annotation draft without paths", async () => {
    const fetchMock = mock(async () =>
      Response.json({ revision: 2, task_override: "place cup", atoms: [] }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getEpisodeAnnotations("dataset/id", "profile/id", 3);
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/datasets/dataset%2Fid/annotations/3?profile_id=profile%2Fid",
    );

    await replaceEpisodeAnnotations(
      "dataset-1",
      "profile-1",
      3,
      2,
      "place cup",
      [],
    );
    const init = fetchMock.mock.calls[1]?.[1] as RequestInit;
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(init.method).toBe("PUT");
    expect(body).toEqual({
      profile_id: "profile-1",
      expected_revision: 2,
      task_override: "place cup",
      atoms: [],
    });
    expect(JSON.stringify(body).toLowerCase()).not.toContain("path");
    expect(JSON.stringify(body).toLowerCase()).not.toContain("token");
  });

  test("starts curation with an opaque recipe and safe output name", async () => {
    const fetchMock = mock(async () =>
      Response.json({ id: "job-curate", status: "queued" }, { status: 202 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await runCurationRecipe(
      "recipe/id",
      "profile-1",
      "pick-clean",
      "curation-intent-1",
    );
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/recipes/recipe%2Fid/runs",
    );
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      profile_id: "profile-1",
      output_name: "pick-clean",
      idempotency_key: "curation-intent-1",
    });
  });

  test("library refresh cannot supply a server path", async () => {
    const fetchMock = mock(async () =>
      Response.json({ id: "job-1", status: "queued" }, { status: 202 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await refreshLibrary("profile-1", "library-refresh-intent-1");
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(body.kind).toBe("datasets.scan");
    expect(body.profile_id).toBe("profile-1");
    expect(body.payload).toEqual({ storage_areas: ["raw", "derived"] });
    expect(body.idempotency_key).toBe("library-refresh-intent-1");
    expect(JSON.stringify(body)).not.toContain("path");
  });

  test("maps job failures without exposing server messages", () => {
    expect(publicJobError("storage_unavailable")).toContain("공유 저장소");
    expect(publicJobError("UnexpectedInternalError")).not.toContain(
      "UnexpectedInternalError",
    );
  });

  test("Hugging Face import sends only the selected immutable revision", async () => {
    const fetchMock = mock(async () =>
      Response.json({ id: "job-hf", status: "queued" }, { status: 202 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await importHuggingFaceDataset(
      "profile-1",
      "pick-cup",
      { name: "main", kind: "branch", commit_sha: "a".repeat(40) },
      "import-intent-1",
    );
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/hf/imports");
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(body).toEqual({
      profile_id: "profile-1",
      dataset_name: "pick-cup",
      requested_revision: "main",
      commit_sha: "a".repeat(40),
      idempotency_key: "import-intent-1",
    });
    expect(JSON.stringify(body).toLowerCase()).not.toContain("token");
    expect(JSON.stringify(body).toLowerCase()).not.toContain("path");
  });
});

describe("job status", () => {
  test("only queued and running jobs are active", () => {
    expect(isActiveJob("queued")).toBe(true);
    expect(isActiveJob("running")).toBe(true);
    expect(isActiveJob("succeeded")).toBe(false);
    expect(isActiveJob("failed")).toBe(false);
  });
});

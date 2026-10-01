import { afterEach, describe, expect, mock, test } from "bun:test";
import {
  approveSegmentationPreview,
  createSegmentationPreview,
  isSegmentationPreviewResult,
  segmentationFrameUrl,
  createSegmentationBatch,
  exportSegmentationBatch,
  invalidateSegmentationBatchItem,
} from "../segmentation-api";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("segmentation API client", () => {
  test("black previews use no image and carry grouped prompts plus manual protection", async () => {
    const fetchMock = mock(async () =>
      Response.json({ id: "job-1", status: "queued" }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    await createSegmentationPreview("owner", "intent", {
      dataset_id: "source",
      fingerprint: "a".repeat(64),
      episode_index: 0,
      video_key: "top",
      mode: "protect_foreground",
      render_mode: "black",
      camera_mode: "fixed",
      background_base64: "",
      prompts: [
        {
          object_id: 2,
          frame_index: 0,
          target: "protect",
          text: "board",
          points: [],
          box: null,
        },
      ],
      corrections: [],
      manual_regions: [{ box: [0.1, 0.2, 0.3, 0.4] }],
      selected_candidate_ids: ["2-1"],
      source_preview_id: "cache-id",
    });
    const body = JSON.parse(
      (fetchMock.mock.calls[0]?.[1] as RequestInit).body as string,
    );
    expect(body.spec).toMatchObject({
      render_mode: "black",
      source_preview_id: "cache-id",
      selected_candidate_ids: ["2-1"],
      prompts: [{ object_id: 2 }],
    });
    expect("background_base64" in body.spec).toBe(false);
  });

  test("batch request binds the exact episode-camera selection and setup confirmation", async () => {
    const fetchMock = mock(async () => Response.json({ id: "batch" }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    await createSegmentationBatch({
      profile_id: "owner",
      idempotency_key: "intent",
      template_id: "template",
      dataset_id: "source",
      fingerprint: "bound-hash",
      episode_indices: [2, 3],
      video_keys: ["top", "wrist"],
      same_camera_setup_confirmed: true,
    });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/segmentation/batches");
    expect(
      JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string),
    ).toMatchObject({
      episode_indices: [2, 3],
      video_keys: ["top", "wrist"],
      same_camera_setup_confirmed: true,
      fingerprint: "bound-hash",
    });
  });

  test("batch export defaults distribution statistics off; edit invalidation persists", async () => {
    const fetchMock = mock(async () => Response.json({ id: "batch" }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    await exportSegmentationBatch("batch/id", "owner", "intent", "output");
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/segmentation/batches/batch%2Fid/exports",
    );
    expect(
      JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string),
    ).toMatchObject({ recompute_statistics: false });
    await invalidateSegmentationBatchItem("batch/id", "item/id", "owner");
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      "/api/v1/segmentation/batches/batch%2Fid/items/item%2Fid/invalidate",
    );
  });
  test("preview request binds fingerprint and explicit erase correction", async () => {
    const fetchMock = mock(async () =>
      Response.json({ id: "job-1", status: "queued" }, { status: 202 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await createSegmentationPreview("profile-1", "intent-1", {
      dataset_id: "dataset-1",
      fingerprint: "a".repeat(64),
      episode_index: 2,
      video_key: "observation.images.top",
      mode: "replace_background",
      background_base64: "cG5n",
      prompts: [
        {
          frame_index: 4,
          target: "replace",
          text: "counter",
          points: [],
          box: null,
        },
      ],
      corrections: [
        {
          frame_index: 4,
          target: "replace",
          operation: "erase",
          radius: 0.02,
          points: [{ x: 0.4, y: 0.5 }],
        },
      ],
    });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/segmentation/previews");
    expect(JSON.parse(init.body as string)).toMatchObject({
      profile_id: "profile-1",
      idempotency_key: "intent-1",
      spec: {
        fingerprint: "a".repeat(64),
        corrections: [{ operation: "erase" }],
      },
    });
  });

  test("approval binds the immutable recipe hash", async () => {
    const fetchMock = mock(async () =>
      Response.json({ approval_token: "token" }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    await approveSegmentationPreview("preview/id", "profile-1", "b".repeat(64));
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/segmentation/previews/preview%2Fid/approve",
    );
    expect(
      JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string),
    ).toEqual({ profile_id: "profile-1", recipe_hash: "b".repeat(64) });
  });

  test("frame URL encodes the opaque camera key and API details are surfaced", async () => {
    const frameToken = "123e4567-e89b-12d3-a456-426614174000";
    expect(
      segmentationFrameUrl("dataset/id", 3, "cam/top", 7, frameToken),
    ).toBe(
      `/api/v1/segmentation/datasets/dataset%2Fid/frame?episode_index=3&video_key=cam%2Ftop&frame_index=7&frame_token=${frameToken}`,
    );
    globalThis.fetch = mock(async () =>
      Response.json(
        { detail: { message: "미리보기가 stale 상태입니다." } },
        { status: 409 },
      ),
    ) as unknown as typeof fetch;
    await expect(
      approveSegmentationPreview("preview-1", "profile-1", "c".repeat(64)),
    ).rejects.toThrow("미리보기가 stale 상태입니다.");
  });

  test("completed preview result requires a string model identity", () => {
    const result = {
      preview_id: "preview-1",
      recipe_hash: "b".repeat(64),
      fingerprint: "a".repeat(64),
      frame_count: 42,
      model: "SAM 3.1",
      model_provenance: { checkpoint_sha256: "c".repeat(64) },
    };
    expect(isSegmentationPreviewResult(result)).toBe(true);
    expect(
      isSegmentationPreviewResult({ ...result, model: { name: "SAM 3.1" } }),
    ).toBe(false);
  });
});

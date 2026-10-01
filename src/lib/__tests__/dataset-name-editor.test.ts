import { afterEach, expect, mock, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import DatasetRow from "../../components/workbench/dataset-row";
import { renameDataset, type DatasetSummary } from "../workbench-api";
const dataset: DatasetSummary = {
  id: "dataset", name: "기존 이름", storage_area: "raw", relative_path: "lab/original",
  codebase_version: "v3.0", readiness: "ready", robot_type: "robot",
  total_episodes: 2, total_frames: 10, total_tasks: 1, fps: 30, fingerprint: "fp",
  scan_error: null, first_seen_at: "2026-09-16T00:00:00Z", last_seen_at: "2026-09-16T00:00:00Z",
  available: true,
};
const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
test("dataset name editing and detail expansion are separate accessible controls", () => {
  const html = renderToStaticMarkup(createElement(DatasetRow, { dataset, onRenamed: () => {} }));
  expect(html).toContain('aria-label="기존 이름 이름 변경"');
  expect(html).toContain('aria-label="기존 이름 상세 정보"');
  expect(html).toContain("lab/original");
  expect(html).not.toContain("<form");
});
test("rename sends display name plus concurrency precondition, not a path", async () => {
  const fetchMock = mock(async () => Response.json({ ...dataset, name: "새 이름" }));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  const updated = await renameDataset("id/encoded", "새 이름", "기존 이름");
  expect(updated.name).toBe("새 이름");
  expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/datasets/id%2Fencoded");
  expect(fetchMock.mock.calls[0]?.[1]?.method).toBe("PATCH");
  expect(JSON.parse(fetchMock.mock.calls[0]?.[1]?.body as string)).toEqual({ name: "새 이름", expected_name: "기존 이름" });
});


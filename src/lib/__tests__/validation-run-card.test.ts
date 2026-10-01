import { afterEach, expect, mock, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ValidationRunCard } from "../../components/workbench/validation-run-card";
import { deleteDatasetValidation, type ValidationRun } from "../workbench-api";

const run: ValidationRun = {
  job_id: "job",
  dataset_id: "dataset",
  dataset_fingerprint: "fp",
  mode: "quick",
  created_at: "2026-09-16T00:00:00Z",
  status: "succeeded",
  error_code: null,
  error_message: null,
  finished_at: null,
  result: {
    mode: "quick",
    passed: false,
    failures: 1,
    warnings: 0,
    checked_episodes: 2,
    total_episodes: 2,
    checked_frames: 10,
    validator_policy: "datasetui-semantic-v2",
    issues: [
      {
        severity: "FAIL",
        code: "feature_dtype_mismatch",
        message: "float32 / float64",
        episode_index: 0,
      },
    ],
  },
};
const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("completed record has independently collapsed result and problem disclosures", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationRunCard, {
      run,
      stale: false,
      onDelete: async () => {},
    }),
  );
  expect(html.match(/<details/g)?.length).toBe(2);
  expect(html).not.toContain(" open=");
  expect(html).toContain("검사 결과");
  expect(html).toContain("문제 상세");
  expect(html).toContain("float32 / float64");
  expect(html).toContain("기록 삭제");
  expect(html).not.toContain("삭제 확인");
});
test("active record preserves visible progress and disables deletion", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationRunCard, {
      run: { ...run, status: "running", result: null },
      stale: false,
      onDelete: async () => {},
    }),
  );
  expect(html).toContain('role="progressbar"');
  expect(html).toContain('disabled=""');
  expect(html).not.toContain("<details");
});
test("validation deletion sends encoded IDs and accepts empty 204", async () => {
  const fetchMock = mock(async () => new Response(null, { status: 204 }));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  await expect(
    deleteDatasetValidation("data/set", "job/id"),
  ).resolves.toBeUndefined();
  expect(fetchMock.mock.calls[0]?.[0]).toBe(
    "/api/v1/datasets/data%2Fset/validations/job%2Fid",
  );
  expect(fetchMock.mock.calls[0]?.[1]?.method).toBe("DELETE");
});
test("validation deletion surfaces server rejection", async () => {
  globalThis.fetch = mock(async () =>
    Response.json(
      { detail: "진행 중인 검사는 삭제할 수 없습니다." },
      { status: 409 },
    ),
  ) as unknown as typeof fetch;
  await expect(deleteDatasetValidation("dataset", "job")).rejects.toThrow(
    "진행 중인 검사는 삭제할 수 없습니다.",
  );
});

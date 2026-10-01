import { expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MergeJobProgress } from "../../components/workbench/merge-job-progress";
import type { Job } from "../workbench-api";

const job: Job = {
  id: "merge-job",
  kind: "datasets.merge",
  queue_name: "cpu",
  status: "running",
  profile_id: "profile",
  payload: { output_name: "merged-test", sources: [] },
  result: null,
  error_code: null,
  error_message: null,
  idempotency_key: "test",
  rq_job_id: null,
  created_at: "2026-09-16T00:00:00Z",
  enqueued_at: null,
  started_at: null,
  finished_at: null,
  cancellation_requested: false,
  cancellation_requested_at: null,
  cancellation_guarded_at: null,
  progress: {
    stage: "video",
    completed: 50,
    total: 100,
    unit: "frames",
    elapsed_seconds: 82,
  },
};
const render = (value: Job, stale = false) =>
  renderToStaticMarkup(
    createElement(MergeJobProgress, { job: value, datasets: [], stale }),
  );

test("merge shows real stage percentage, frame counts and elapsed time", () => {
  const html = render(job);
  expect(html).toContain('aria-valuenow="50"');
  expect(html).toContain("width:50%");
  expect(html).toContain("영상 처리");
  expect(html).toContain("50 / 100");
  expect(html).toContain("프레임");
  expect(html).toContain("1분 22초");
  expect(html).toContain("현재 단계 기준");
});
test("queued and legacy jobs never pretend to report measured progress", () => {
  const queued = render({ ...job, status: "queued" });
  expect(queued).toContain("작업자 배정 대기");
  expect(queued).not.toContain("aria-valuenow");
  expect(render({ ...job, progress: null })).toContain(
    "세부 처리량을 보고하지 않습니다",
  );
});
test("unknown totals are indeterminate and excessive percentages are clamped", () => {
  expect(
    render({ ...job, progress: { ...job.progress!, total: 0 } }),
  ).not.toContain("aria-valuenow");
  expect(
    render({ ...job, progress: { ...job.progress!, completed: 150 } }),
  ).toContain('aria-valuenow="100"');
});
test("stale progress is preserved but animation stops", () => {
  const html = render(job, true);
  expect(html).toContain("상태 확인 지연");
  expect(html).toContain('data-active="false"');
  expect(html).toContain('aria-valuenow="50"');
});
test("completion links to library and failures do not leak raw errors", () => {
  const success = render({
    ...job,
    status: "succeeded",
    result: { output: { episodes: 12 } },
  });
  expect(success).toContain("합치기 완료");
  expect(success).toContain("에피소드 12개");
  expect(success).toContain('href="/library"');
  expect(success).not.toContain('role="progressbar"');
  const failed = render({
    ...job,
    status: "failed",
    error_code: "internal_error",
    error_message: "SECRET",
  });
  expect(failed).toContain("합치기 실패");
  expect(failed).not.toContain("SECRET");
});

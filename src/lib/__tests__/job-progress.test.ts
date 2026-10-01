import { expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { JobProgress } from "../../components/workbench/job-progress";
import { jobLabel } from "../job-presentation";
import type { Job } from "../workbench-api";
const job: Job = {
  id: "job",
  kind: "datasets.upload_hf",
  queue_name: "io",
  status: "running",
  profile_id: "p",
  payload: { repo_name: "safe-name" },
  result: null,
  error_code: null,
  error_message: null,
  idempotency_key: "test",
  rq_job_id: null,
  created_at: "2026-09-17T00:00:00Z",
  enqueued_at: null,
  started_at: null,
  finished_at: null,
  cancellation_requested: false,
  cancellation_requested_at: null,
  cancellation_guarded_at: null,
  progress: {
    stage: "upload",
    completed: 0,
    total: 0,
    unit: "bytes",
    elapsed_seconds: 65,
  },
};
const render = (value: Job, stale = false) =>
  renderToStaticMarkup(createElement(JobProgress, { job: value, stale }));
test("all registered operations have progress UI without unsupported labels", () => {
  for (const kind of [
    "datasets.scan",
    "hf.import",
    "curation.materialize",
    "datasets.merge",
    "datasets.validate",
    "datasets.convert_v21",
    "datasets.export_nas",
    "datasets.upload_hf",
    "datasets.copy_pc_key",
    "segmentation.preview",
    "segmentation.export",
    "phase2.smoke",
  ]) {
    const html = render({ ...job, kind });
    expect(html).toContain('role="progressbar"');
    expect(html).toContain(jobLabel({ kind }));
    expect(html).not.toContain("지원되지 않는 작업");
  }
});
test("opaque HF upload stays indeterminate and shows elapsed and current stage", () => {
  const html = render(job);
  expect(html).not.toContain("aria-valuenow");
  expect(html).toContain("Hugging Face 업로드");
  expect(html).toContain("1분 5초");
  expect(html).toContain("전체 처리량을 알 수 없는");
});
test("file and byte counts use measured stage-local percent", () => {
  const html = render({
    ...job,
    progress: {
      ...job.progress!,
      stage: "copy",
      completed: 2,
      total: 4,
      unit: "files",
    },
  });
  expect(html).toContain('aria-valuenow="50"');
  expect(html).toContain("2 / 4");
  expect(html).toContain("파일");
});
test("zero progress has no visible minimum and queue ignores stale attempt counts", () => {
  const zero = render({ ...job, progress: { ...job.progress!, total: 20 } });
  expect(zero).toContain("width:0%;min-width:0");
  const queued = render({
    ...job,
    status: "queued",
    progress: { ...job.progress!, completed: 10, total: 20 },
  });
  expect(queued).not.toContain("aria-valuenow");
  expect(queued).toContain("작업자 배정 대기");
});
test("only the selected owner can cancel a queued or running job", () => {
  const queued = { ...job, status: "queued" as const };
  const owned = renderToStaticMarkup(
    createElement(JobProgress, {
      job: queued,
      currentProfileId: "p",
    }),
  );
  const other = renderToStaticMarkup(
    createElement(JobProgress, {
      job: queued,
      currentProfileId: "another-profile",
    }),
  );
  const running = renderToStaticMarkup(
    createElement(JobProgress, {
      job,
      currentProfileId: "p",
    }),
  );
  expect(owned).toContain("대기 작업 취소");
  expect(other).not.toContain("대기 작업 취소");
  expect(running).toContain("실행 작업 취소");
});
test("a running cancellation request is visible and cannot be submitted twice", () => {
  const html = renderToStaticMarkup(
    createElement(JobProgress, {
      job: {
        ...job,
        cancellation_requested: true,
        cancellation_requested_at: "2026-09-21T00:00:00Z",
      },
      currentProfileId: "p",
    }),
  );
  expect(html).toContain("취소 요청 중");
  expect(html).not.toContain("실행 작업 취소");
});
test("a job past the publication guard explains that cancellation is too late", () => {
  const html = renderToStaticMarkup(
    createElement(JobProgress, {
      job: {
        ...job,
        cancellation_guarded_at: "2026-09-21T00:00:00Z",
      },
      currentProfileId: "p",
    }),
  );
  expect(html).toContain("마무리 중");
  expect(html).toContain("취소할 수 없습니다");
  expect(html).not.toContain("실행 작업 취소");
});
test("stale freezes bar and terminal outcome is not mislabeled as disconnected", () => {
  expect(render(job, true)).toContain('data-active="false"');
  const html = render(
    {
      ...job,
      status: "succeeded",
      result: { repo_id: "rainbowrobotics/safe-name" },
    },
    true,
  );
  expect(html).not.toContain("상태 확인 지연");
  expect(html).toContain(
    'href="https://huggingface.co/datasets/rainbowrobotics/safe-name"',
  );
  expect(html).not.toContain('role="progressbar"');
});
test("error messages never render raw backend secrets", () => {
  const html = render({
    ...job,
    status: "failed",
    error_code: "job_failed",
    error_message: "SECRET_TOKEN",
  });
  expect(html).not.toContain("SECRET_TOKEN");
  expect(html).toContain("실패");
});

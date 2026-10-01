import { expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ValidationProgress } from "../../components/workbench/validation-progress";
import type {
  ValidationCheck,
  ValidationProgressSnapshot,
  ValidationRun,
} from "../workbench-api";

const checks: ValidationCheck[] = [
  {
    id: "metadata",
    label: "메타데이터와 필수 컬럼",
    status: "passed",
    failures: 0,
    warnings: 0,
    detail: "필수 필드를 확인했습니다.",
  },
  {
    id: "video",
    label: "영상 디코딩",
    status: "running",
    failures: 0,
    warnings: 1,
  },
  {
    id: "statistics",
    label: "통계 파일",
    status: "skipped",
    failures: 0,
    warnings: 0,
  },
];

const progress: ValidationProgressSnapshot = {
  stage: "video",
  completed: 2,
  total: 5,
  decoded_frames: 1234,
  elapsed_seconds: 82,
  checks,
};

const run: ValidationRun = {
  job_id: "job",
  dataset_id: "dataset",
  dataset_fingerprint: "fingerprint",
  mode: "full",
  created_at: "2026-09-14T00:00:00Z",
  started_at: null,
  status: "queued",
  progress: null,
  result: null,
  error_code: null,
  error_message: null,
  finished_at: null,
};

test("legacy queued and running jobs show an indeterminate progress bar", () => {
  for (const status of ["queued", "running"] as const) {
    const html = renderToStaticMarkup(
      createElement(ValidationProgress, { run: { ...run, status } }),
    );
    expect(html).toContain('role="progressbar"');
    expect(html).not.toContain("aria-valuenow");
    expect(html).toContain('data-determinate="false"');
    expect(html).toContain(status === "queued" ? "대기 중" : "검사 중");
  }
});

test("running job shows real episode progress, stage, frames, elapsed time, and checks", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationProgress, {
      run: { ...run, status: "running", progress },
    }),
  );
  expect(html).toContain('aria-valuenow="40"');
  expect(html).toContain('style="width:40%"');
  expect(html).toContain("에피소드 2 / 5");
  expect(html).toContain("영상");
  expect(html).toContain("1,234");
  expect(html).toContain("1분 22초");
  expect(html).toContain("메타데이터·데이터셋 구조");
  expect(html).toContain("데이터셋 통계");
  expect(html).toContain("미검사");
});

test("metadata stage shows elapsed details before the episode total is known", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationProgress, {
      run: {
        ...run,
        status: "running",
        progress: { ...progress, stage: "metadata", completed: 0, total: 0 },
      },
    }),
  );
  expect(html).not.toContain("aria-valuenow");
  expect(html).toContain("에피소드 진행률 계산 중");
  expect(html).toContain("에피소드 0 / 확인 중");
  expect(html).toContain("메타데이터");
  expect(html).toContain("1분 22초");
});

test("percent is clamped to the accessible range", () => {
  for (const [completed, expected] of [
    [-4, 0],
    [8, 100],
  ] as const) {
    const html = renderToStaticMarkup(
      createElement(ValidationProgress, {
        run: {
          ...run,
          status: "running",
          progress: { ...progress, completed, total: 5 },
        },
      }),
    );
    expect(html).toContain(`aria-valuenow="${expected}"`);
    expect(html).toContain(`style="width:${expected}%"`);
  }
});

test("stale running state stops animation and labels metrics as last seen", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationProgress, {
      run: { ...run, status: "running", progress },
      stale: true,
    }),
  );
  expect(html).toContain("상태 확인 지연");
  expect(html).toContain('data-active="false"');
  expect(html).toContain("마지막 확인 시점 기준");
  expect(html).toContain('aria-valuetext="마지막으로 확인된 진행 상태"');
});

test("completed validations show final checks including skipped items", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationProgress, {
      run: {
        ...run,
        status: "succeeded",
        result: {
          mode: "full",
          validator_policy: "datasetui-semantic-v2",
          passed: true,
          checked_episodes: 5,
          total_episodes: 5,
          checked_frames: 200,
          failures: 0,
          warnings: 1,
          issues: [],
          checks: checks.map((check) =>
            check.id === "video" ? { ...check, status: "warning" } : check,
          ),
        },
      },
    }),
  );
  expect(html).not.toContain('role="progressbar"');
  expect(html).toContain("검사 완료 · 통과");
  expect(html).toContain("영상 디코딩");
  expect(html).toContain("경고");
  expect(html).toContain("미검사");
});

test("partial failure is not animated or shown as successful completion", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationProgress, {
      run: {
        ...run,
        status: "failed",
        progress: {
          ...progress,
          completed: 3,
          checks: checks.map((check) =>
            check.id === "metadata"
              ? { ...check, status: "failed", failures: 1 }
              : check,
          ),
        },
      },
    }),
  );
  expect(html).toContain("검사 실패");
  expect(html).toContain('data-active="false"');
  expect(html).not.toContain('aria-valuenow="100"');
  expect(html).not.toContain("검사 완료 · 통과");
  expect(html).toContain("실패 1 · 경고 0");
  expect(html).toContain("완료 전 중단");
  expect(html).not.toContain(">검사 중<");
});

test("legacy completed results disclose missing detailed history", () => {
  const html = renderToStaticMarkup(
    createElement(ValidationProgress, {
      run: {
        ...run,
        status: "succeeded",
        result: {
          mode: "quick",
          passed: false,
          checked_episodes: 2,
          total_episodes: 2,
          checked_frames: 20,
          failures: 1,
          warnings: 0,
          issues: [],
        },
      },
    }),
  );
  expect(html).toContain("이전 검사 기준 · 재검사 필요");
  expect(html).toContain("상세 검사 이력을 제공하지 않는 이전 검사 기록");
  expect(html).not.toContain("메타데이터·데이터셋 구조");
});

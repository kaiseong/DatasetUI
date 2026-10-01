import { expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ResourceSchedulerStatusView } from "../../components/workbench/resource-scheduler-status";
import type { SystemResources } from "../workbench-api";

const resources: SystemResources = {
  enabled: true,
  healthy: true,
  sampled_at: "2026-09-17T00:00:00Z",
  cpu_available_percent: 54.25,
  memory_available_percent: 41.5,
  iowait_percent: 1.2,
  active_jobs: 2,
  max_parallel_jobs: 3,
  queued_jobs: 4,
  reserved_worker_slots: 3,
  managed_running_jobs: 1,
  minimum_available_percent: 30,
  decision: "ready",
  message: "리소스 여유가 있어 다음 CPU 작업을 시작할 수 있습니다.",
};

test("shows measured headroom, threshold, queue and CPU concurrency", () => {
  const html = renderToStaticMarkup(
    createElement(ResourceSchedulerStatusView, { resources }),
  );
  expect(html).toContain("병렬 실행 가능");
  expect(html).toContain("54.3%");
  expect(html).toContain("41.5%");
  expect(html).toContain("2 / 3");
  expect(html).toContain("대기 4개");
  expect(html).toContain("각각 30% 이상");
  expect(html).toContain("4 CPU·8 GiB");
  expect(html).toContain("I/O wait가 10%");
  expect(html).toContain("기존 실행 작업도 한도에 포함");
  expect(html).toContain("GPU 작업은 제외");
});

test("unhealthy samples say measurement pending instead of showing zero", () => {
  const html = renderToStaticMarkup(
    createElement(ResourceSchedulerStatusView, {
      resources: {
        ...resources,
        healthy: false,
        sampled_at: null,
        cpu_available_percent: null,
        memory_available_percent: null,
        decision: "unavailable",
        message: "호스트 측정값을 가져오지 못했습니다.",
      },
    }),
  );
  expect(html).toContain("리소스 측정 불가");
  expect(html).toContain("측정 대기");
  expect(html).not.toContain("0.0%");
});

test("stale status explains that the displayed values are not fresh", () => {
  const html = renderToStaticMarkup(
    createElement(ResourceSchedulerStatusView, {
      resources,
      stale: true,
    }),
  );
  expect(html).toContain("상태 갱신 지연");
  expect(html).toContain("마지막 측정값");
});

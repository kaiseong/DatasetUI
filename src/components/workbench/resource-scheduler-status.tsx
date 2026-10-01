"use client";

import { useEffect, useState, type ReactNode } from "react";
import { LuCpu, LuGauge, LuMemoryStick } from "react-icons/lu";
import { getSystemResources, type SystemResources } from "@/lib/workbench-api";

const DECISION_LABELS: Record<SystemResources["decision"], string> = {
  ready: "병렬 실행 가능",
  no_jobs: "대기 작업 없음",
  resource_wait: "리소스 여유 대기",
  limit: "동시 작업 한도 도달",
  warming_up: "리소스 측정 준비 중",
  unavailable: "리소스 측정 불가",
};

export function ResourceSchedulerStatus() {
  const [resources, setResources] = useState<SystemResources | null>(null);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | null = null;
    async function poll() {
      controller = new AbortController();
      try {
        const loaded = await getSystemResources(controller.signal);
        if (active) {
          setResources(loaded);
          setStale(false);
        }
      } catch {
        if (active) setStale(true);
      } finally {
        if (active) timer = setTimeout(poll, 5000);
      }
    }
    void poll();
    return () => {
      active = false;
      controller?.abort();
      clearTimeout(timer);
    };
  }, []);

  return <ResourceSchedulerStatusView resources={resources} stale={stale} />;
}

export function ResourceSchedulerStatusView({
  resources,
  stale = false,
}: {
  resources: SystemResources | null;
  stale?: boolean;
}) {
  const measured = Boolean(resources?.healthy && resources.sampled_at);
  const decision = resources
    ? DECISION_LABELS[resources.decision]
    : "리소스 확인 중";
  const waiting =
    !resources ||
    !measured ||
    resources.decision === "warming_up" ||
    resources.decision === "unavailable";

  return (
    <section
      className="resource-scheduler-status"
      data-ready={resources?.decision === "ready"}
      data-waiting={waiting}
      aria-label="자동 병렬 작업 상태"
    >
      <header>
        <div>
          <p className="workbench-eyebrow">RESOURCE SCHEDULER</p>
          <h3>
            <LuGauge aria-hidden /> 자동 병렬 실행
          </h3>
        </div>
        <strong role="status">{stale ? "상태 갱신 지연" : decision}</strong>
      </header>
      <div className="resource-scheduler-status__metrics">
        <ResourceMetric
          icon={<LuCpu aria-hidden />}
          label="CPU 여유"
          value={measured ? resources?.cpu_available_percent : null}
          threshold={resources?.minimum_available_percent ?? 30}
        />
        <ResourceMetric
          icon={<LuMemoryStick aria-hidden />}
          label="RAM 여유"
          value={measured ? resources?.memory_available_percent : null}
          threshold={resources?.minimum_available_percent ?? 30}
        />
        <div className="resource-scheduler-status__jobs">
          <small>CPU 작업</small>
          <strong>
            {resources
              ? `${resources.active_jobs} / ${resources.max_parallel_jobs}`
              : "—"}
          </strong>
          <span>
            {resources
              ? `대기 ${resources.queued_jobs}개`
              : "작업 상태 확인 중"}
          </span>
        </div>
      </div>
      <p className="resource-scheduler-status__message">
        {stale
          ? "마지막 측정값을 표시하고 있습니다. 서버 연결 후 자동 갱신합니다."
          : resources?.message ||
            "CPU와 RAM 여유를 측정한 뒤 대기 작업의 실행 여부를 결정합니다."}
      </p>
      <small className="resource-scheduler-status__note">
        새 작업의 예약치(4 CPU·8 GiB)를 반영한 뒤에도 CPU와 RAM이 각각{" "}
        {resources?.minimum_available_percent ?? 30}% 이상 남고 I/O wait가 10%
        이하일 때 CPU 작업을 최대 {resources?.max_parallel_jobs ?? 3}개까지 자동
        실행합니다. 기존 실행 작업도 한도에 포함되고 GPU 작업은 제외됩니다.
      </small>
    </section>
  );
}

function ResourceMetric({
  icon,
  label,
  value,
  threshold,
}: {
  icon: ReactNode;
  label: string;
  value: number | null | undefined;
  threshold: number;
}) {
  const normalized =
    typeof value === "number" && Number.isFinite(value)
      ? Math.min(100, Math.max(0, value))
      : null;
  return (
    <div className="resource-scheduler-status__metric">
      <span>
        {icon}
        <small>{label}</small>
      </span>
      <strong>
        {normalized === null ? "측정 대기" : `${normalized.toFixed(1)}%`}
      </strong>
      <div
        className="resource-scheduler-status__track"
        role={normalized === null ? undefined : "meter"}
        aria-label={
          normalized === null ? undefined : `${label} ${normalized.toFixed(1)}%`
        }
        aria-valuemin={normalized === null ? undefined : 0}
        aria-valuemax={normalized === null ? undefined : 100}
        aria-valuenow={normalized ?? undefined}
      >
        <span
          style={normalized === null ? undefined : { width: `${normalized}%` }}
        />
        <i style={{ left: `${threshold}%` }} aria-hidden />
      </div>
    </div>
  );
}

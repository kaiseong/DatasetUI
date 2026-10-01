"use client";
import { useEffect, useState } from "react";
import { useProfile } from "./profile-context";
import { JobProgress } from "./job-progress";
import {
  isActiveJob,
  listActiveTeamJobs,
  listJobs,
  type Job,
} from "@/lib/workbench-api";
import { ResourceSchedulerStatus } from "./resource-scheduler-status";

export function WorkbenchJobProgress() {
  const { currentProfile } = useProfile();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [stale, setStale] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | null = null;
    async function poll() {
      controller = new AbortController();
      const timeout = setTimeout(() => controller?.abort(), 15000);
      try {
        const [running, recent] = await Promise.all([
          listActiveTeamJobs(controller.signal),
          listJobs(undefined, controller.signal),
        ]);
        if (active) {
          const combined = new Map(
            recent
              .filter((job) => !isActiveJob(job.status))
              .slice(0, 5)
              .map((job) => [job.id, job]),
          );
          for (const job of running) combined.set(job.id, job);
          setJobs([...combined.values()]);
          setStale(false);
          setLoaded(true);
        }
      } catch {
        if (active) setStale(true);
      } finally {
        clearTimeout(timeout);
        if (active) timer = setTimeout(poll, 2000);
      }
    }
    const accepted = (event: Event) => {
      const job = (event as CustomEvent<Job>).detail;
      if (!job?.id) return;
      active = false;
      controller?.abort();
      clearTimeout(timer);
      setJobs((current) => [
        job,
        ...current.filter((item) => item.id !== job.id),
      ]);
      setRefresh((value) => value + 1);
    };
    window.addEventListener("datasetui:job-accepted", accepted);
    void poll();
    return () => {
      active = false;
      controller?.abort();
      clearTimeout(timer);
      window.removeEventListener("datasetui:job-accepted", accepted);
    };
  }, [refresh]);
  const activeJobs = jobs.filter((job) => isActiveJob(job.status));
  const running = activeJobs.filter((job) => job.status === "running");
  const queued = activeJobs.filter((job) => job.status === "queued");
  const recent = jobs.filter((job) => !isActiveJob(job.status)).slice(0, 5);
  return (
    <section className="workbench-job-dock" aria-label="전체 작업 진행 상황">
      <button
        type="button"
        className="workbench-job-dock__summary"
        aria-expanded={expanded}
        aria-controls="team-job-details"
        onClick={() => setExpanded((value) => !value)}
      >
        <strong>팀 작업 진행 상황</strong>
        <span aria-live="polite">
          {stale
            ? "상태 확인 지연"
            : loaded
              ? `실행 ${running.length} · 대기 ${queued.length}`
              : "확인 중…"}
        </span>
        <span>{expanded ? "접기 ▾" : "펼치기 ▴"}</span>
      </button>
      {expanded && (
        <div id="team-job-details" className="workbench-job-dock__body">
          <p>
            팀 전체 작업을 2초마다 갱신합니다. 선택한 사용자의 작업만 취소할 수
            있습니다.
          </p>
          <ResourceSchedulerStatus />
          {stale && (
            <p role="alert">
              연결을 다시 시도하고 있습니다. 마지막 확인 상태를 표시합니다.
            </p>
          )}
          {activeJobs.length === 0 && loaded && (
            <p>진행·대기 중인 작업이 없습니다.</p>
          )}
          {activeJobs.map((job) => (
            <JobProgress
              key={job.id}
              job={job}
              stale={stale}
              currentProfileId={currentProfile?.id}
              onJobUpdate={(updated) =>
                setJobs((current) =>
                  current.map((item) =>
                    item.id === updated.id ? updated : item,
                  ),
                )
              }
            />
          ))}
          {recent.length > 0 && (
            <details className="job-recent">
              <summary>최근 완료·실패 작업 ({recent.length})</summary>
              {recent.map((job) => (
                <JobProgress
                  key={job.id}
                  job={job}
                  currentProfileId={currentProfile?.id}
                />
              ))}
            </details>
          )}
        </div>
      )}
    </section>
  );
}

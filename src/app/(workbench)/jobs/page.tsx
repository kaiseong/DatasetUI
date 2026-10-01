"use client";

import { useEffect, useMemo, useState } from "react";
import { LuActivity, LuListFilter, LuServerOff } from "react-icons/lu";
import EmptyState from "@/components/workbench/empty-state";
import JobRow from "@/components/workbench/job-row";
import { ResourceSchedulerStatus } from "@/components/workbench/resource-scheduler-status";
import { useProfile } from "@/components/workbench/profile-context";
import { isActiveJob, listJobs, type Job } from "@/lib/workbench-api";

export default function JobsPage() {
  const { currentProfile, profiles } = useProfile();
  const [scope, setScope] = useState<"mine" | "everyone">("mine");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadRequest, setReloadRequest] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    let timer: number | undefined;
    setLoading(true);
    const poll = async () => {
      if (scope === "mine" && !currentProfile) {
        setJobs([]);
        setLoading(false);
        return;
      }
      try {
        const loaded = await listJobs(
          scope === "mine" ? currentProfile?.id : undefined,
          controller.signal,
        );
        if (!active) return;
        setJobs(loaded);
        setError(null);
        timer = window.setTimeout(
          poll,
          loaded.some((job) => isActiveJob(job.status)) ? 1500 : 8000,
        );
      } catch (requestError) {
        if (!active || controller.signal.aborted) return;
        setError(
          requestError instanceof Error
            ? requestError.message
            : "작업 기록을 불러오지 못했습니다.",
        );
        timer = window.setTimeout(poll, 2000);
      } finally {
        if (active) setLoading(false);
      }
    };
    void poll();
    return () => {
      active = false;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [scope, currentProfile, reloadRequest]);

  const profileById = useMemo(
    () => new Map(profiles.map((profile) => [profile.id, profile])),
    [profiles],
  );
  const visibleJobs =
    scope === "everyone"
      ? jobs
      : jobs.filter((job) => job.profile_id === currentProfile?.id);
  const completed = visibleJobs.filter(
    (job) => job.status === "succeeded",
  ).length;
  const needsAttention = jobs.filter((job) =>
    ["failed", "interrupted"].includes(job.status),
  ).length;

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">ACTIVITY</p>
          <h1>작업 기록</h1>
          <p>누가 어떤 작업을 시작했고 어디까지 진행됐는지 확인합니다.</p>
        </div>
        <label className="jobs-scope">
          <LuListFilter aria-hidden />
          <span className="sr-only">작업자 범위</span>
          <select
            value={scope}
            onChange={(event) => setScope(event.target.value as typeof scope)}
          >
            <option value="mine">내 작업</option>
            <option value="everyone">모든 사용자</option>
          </select>
        </label>
      </section>

      <ResourceSchedulerStatus />

      <div className="activity-summary">
        <div>
          <small>표시 중</small>
          <strong>{jobs.length}</strong>
        </div>
        <div>
          <small>진행 중</small>
          <strong>
            {jobs.filter((job) => isActiveJob(job.status)).length}
          </strong>
        </div>
        <div>
          <small>완료</small>
          <strong>{completed}</strong>
        </div>
        <div className={needsAttention > 0 ? "has-alert" : ""}>
          <small>확인 필요</small>
          <strong>{needsAttention}</strong>
        </div>
      </div>

      {loading ? (
        <div className="library-skeleton">
          {[0, 1, 2].map((item) => (
            <div key={item} />
          ))}
        </div>
      ) : error && visibleJobs.length === 0 ? (
        <EmptyState
          icon={<LuServerOff />}
          title="작업 기록에 연결할 수 없습니다"
          description={error}
          action={
            <button
              type="button"
              className="workbench-button"
              onClick={() => setReloadRequest((value) => value + 1)}
            >
              다시 시도
            </button>
          }
        />
      ) : visibleJobs.length === 0 ? (
        <EmptyState
          icon={<LuActivity />}
          title="아직 작업 기록이 없습니다"
          description="라이브러리를 새로 확인하거나 데이터셋 작업을 시작하면 여기에 표시됩니다."
        />
      ) : (
        <section className="job-list" aria-label="작업 목록">
          {error && (
            <p role="alert">
              상태 갱신이 지연되고 있습니다. 마지막 확인 상태를 유지하며 다시
              연결합니다.
            </p>
          )}
          {visibleJobs.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              profile={profileById.get(job.profile_id)}
              currentProfileId={currentProfile?.id}
              onJobUpdate={(updated) =>
                setJobs((current) =>
                  current.map((item) =>
                    item.id === updated.id ? updated : item,
                  ),
                )
              }
              stale={Boolean(error)}
            />
          ))}
        </section>
      )}
    </div>
  );
}

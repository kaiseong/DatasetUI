"use client";

import { useEffect, useState } from "react";
import {
  LuCheck,
  LuChevronDown,
  LuCircleAlert,
  LuClock3,
  LuLoaderCircle,
  LuRefreshCw,
  LuX,
} from "react-icons/lu";
import {
  listJobEvents,
  publicJobError,
  type Job,
  type JobEvent,
  type JobStatus,
  type Profile,
} from "@/lib/workbench-api";

const STATUS: Record<
  JobStatus,
  { label: string; icon: typeof LuClock3; tone: string }
> = {
  queued: { label: "대기 중", icon: LuClock3, tone: "waiting" },
  running: { label: "진행 중", icon: LuLoaderCircle, tone: "running" },
  succeeded: { label: "완료", icon: LuCheck, tone: "success" },
  failed: { label: "확인 필요", icon: LuCircleAlert, tone: "error" },
  interrupted: { label: "중단됨", icon: LuCircleAlert, tone: "error" },
  cancelled: { label: "취소됨", icon: LuX, tone: "muted" },
};

export default function JobRow({
  job,
  profile,
}: {
  job: Job;
  profile?: Profile;
}) {
  const [expanded, setExpanded] = useState(false);
  const [events, setEvents] = useState<JobEvent[] | null>(null);
  const [eventsError, setEventsError] = useState<string | null>(null);
  const status = STATUS[job.status];
  const StatusIcon = status.icon;

  useEffect(() => {
    if (!expanded) return;
    let active = true;
    setEvents(null);
    setEventsError(null);
    void listJobEvents(job.id)
      .then((loaded) => active && setEvents(loaded))
      .catch((error: unknown) => {
        if (active) {
          setEventsError(
            error instanceof Error
              ? error.message
              : "상세 기록을 불러오지 못했습니다.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [expanded, job.id, job.status]);

  return (
    <article className="job-row">
      <button
        type="button"
        className="job-row__summary"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        <span className={`job-status job-status--${status.tone}`}>
          <StatusIcon
            className={job.status === "running" ? "animate-spin" : ""}
            aria-hidden
          />
        </span>
        <span className="job-row__identity">
          <strong>{jobLabel(job)}</strong>
          <small>{jobDescription(job)}</small>
        </span>
        <span className="job-row__person">
          {profile?.name ?? "알 수 없는 사용자"}
        </span>
        <span className={`job-row__state job-row__state--${status.tone}`}>
          {status.label}
        </span>
        <time className="job-row__time" dateTime={job.created_at}>
          {formatRelativeDate(job.created_at)}
        </time>
        <LuChevronDown
          className={`job-row__chevron ${expanded ? "is-expanded" : ""}`}
          aria-hidden
        />
      </button>

      {expanded && (
        <div className="job-row__detail">
          {job.error_code && (
            <p className="job-row__error" role="alert">
              {publicJobError(job.error_code)}
            </p>
          )}
          {eventsError ? (
            <p className="text-sm text-red-300">{eventsError}</p>
          ) : !events ? (
            <p className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
              <LuRefreshCw className="animate-spin" aria-hidden /> 기록 불러오는
              중…
            </p>
          ) : (
            <ol className="job-events">
              {events.map((event) => (
                <li key={event.sequence}>
                  <span />
                  <div>
                    <strong>{eventLabel(event.event_type)}</strong>
                    <time dateTime={event.created_at}>
                      {formatDate(event.created_at)}
                    </time>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </article>
  );
}

function jobLabel(job: Job) {
  return (
    {
      "datasets.scan": "라이브러리 새로 확인",
      "phase2.smoke": "시스템 확인",
      "hf.import": "Hugging Face 원본 가져오기",
    }[job.kind] ?? "지원되지 않는 작업"
  );
}

function jobDescription(job: Job) {
  if (job.kind === "datasets.scan") return "공유 저장소에서 데이터셋 찾기";
  if (job.kind === "phase2.smoke") return "DatasetUI 작동 상태 확인";
  if (job.kind === "hf.import") {
    const datasetName = job.payload.dataset_name;
    return typeof datasetName === "string"
      ? `${datasetName} revision을 원본으로 보존`
      : "선택한 revision을 원본으로 보존";
  }
  return "이 작업 유형은 현재 화면에서 지원되지 않습니다.";
}

function eventLabel(eventType: string) {
  return (
    {
      queued: "작업 접수",
      dispatched: "작업자에게 전달",
      running: "작업 시작",
      succeeded: "작업 완료",
      failed: "작업 실패",
      interrupted: "작업 중단",
      requeued: "자동 재시도 대기",
      dispatch_uncertain: "전달 상태 확인 중",
      dispatch_recovered: "작업 전달 복구",
    }[eventType] ?? "기타 작업 기록"
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatRelativeDate(value: string) {
  const elapsed = Date.now() - new Date(value).getTime();
  const minutes = Math.floor(elapsed / 60_000);
  if (minutes < 1) return "방금 전";
  if (minutes < 60) return `${minutes}분 전`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}시간 전`;
  return formatDate(value);
}

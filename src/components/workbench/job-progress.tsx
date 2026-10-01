"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { isActiveJob, publicJobError, type Job } from "@/lib/workbench-api";
import { JobCancelControl } from "./job-cancel-control";
import {
  formatDuration,
  jobLabel,
  jobDescription,
  jobResultLinks,
  wholeJobEstimate,
} from "@/lib/job-presentation";

const STAGES: Record<string, string> = {
  starting: "작업 시작",
  preparing: "원본·작업 조건 확인",
  read: "에피소드 읽기",
  write: "데이터 기록",
  video: "영상 처리",
  statistics: "통계 계산",
  relative_action_statistics: "Relative Action 통계 계산",
  validate: "데이터 검증",
  publish: "저장·무결성 확인",
  register: "라이브러리 등록",
  complete: "처리 완료",
  scan: "데이터셋 탐색",
  download: "Hugging Face 다운로드",
  upload: "Hugging Face 업로드",
  copy: "파일 복사",
  verify: "무결성 확인",
  connect: "대상 연결",
  segment: "SAM 마스크 생성",
  metadata: "메타데이터 확인",
  episodes: "에피소드 검사",
  export_gate: "내보내기 조건 확인",
  delete: "영구 삭제",
};
const UNITS: Record<string, string> = {
  chunks: "action chunks",
  episodes: "에피소드",
  frames: "프레임",
  frame_operations: "프레임 처리(디코딩+인코딩)",
  files: "파일",
  bytes: "바이트",
  items: "항목",
  datasets: "데이터셋",
  areas: "저장 영역",
};
const STATUS: Record<string, string> = {
  queued: "작업자 배정 대기",
  running: "진행 중",
  succeeded: "완료",
  failed: "실패",
  cancelled: "취소됨",
  interrupted: "중단됨",
};

export function JobProgress({
  job,
  stale = false,
  compact = false,
  currentProfileId,
  onJobUpdate,
}: {
  job: Job;
  stale?: boolean;
  compact?: boolean;
  currentProfileId?: string;
  onJobUpdate?: (job: Job) => void;
}) {
  const active = isActiveJob(job.status);
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    if (!active || stale) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active, stale, job.id]);
  const progress = job.status === "queued" ? null : job.progress;
  const completed =
    progress && Number.isFinite(progress.completed)
      ? Math.max(0, progress.completed)
      : 0;
  const total =
    progress && Number.isFinite(progress.total) ? progress.total : 0;
  const percent =
    total > 0 ? Math.min(100, Math.floor((completed / total) * 100)) : null;
  const recordedElapsed =
    progress && Number.isFinite(progress.elapsed_seconds)
      ? Math.max(0, Math.floor(progress.elapsed_seconds))
      : null;
  const started = job.started_at ? Date.parse(job.started_at) : NaN;
  const elapsed =
    active && !stale && now !== null && Number.isFinite(started)
      ? Math.max(recordedElapsed ?? 0, Math.floor((now - started) / 1000), 0)
      : recordedElapsed;
  const whole = active ? wholeJobEstimate(progress?.overall, elapsed) : null;
  const barPercent = whole ? whole.percent : percent;
  const links = jobResultLinks(job);
  return (
    <section
      className="generic-job-progress"
      aria-label={`${jobLabel(job)} 진행 상황`}
    >
      {!compact && (
        <header className="flex flex-wrap justify-between gap-2">
          <div>
            <h3>{jobLabel(job)}</h3>
            {job.profile_name && <small>작업자: {job.profile_name}</small>}
            <p className="break-all">{jobDescription(job)}</p>
          </div>
          <strong role="status">
            {stale && active
              ? "상태 확인 지연"
              : job.status === "succeeded" && job.result?.partial_failure
                ? "일부 실패"
                : (STATUS[job.status] ?? "상태 확인")}
          </strong>
        </header>
      )}
      <div
        className="validation-progress"
        data-active={active && !stale}
        data-determinate={barPercent !== null}
        data-status={job.status}
      >
        {active && (
          <div
            className="validation-progress__track"
            role="progressbar"
            aria-label={`${jobLabel(job)} ${whole ? "전체" : "현재 단계"} 진행률`}
            aria-valuemin={barPercent === null ? undefined : 0}
            aria-valuemax={barPercent === null ? undefined : 100}
            aria-valuenow={barPercent ?? undefined}
            aria-valuetext={
              stale
                ? `마지막 확인 상태${barPercent === null ? "" : ` ${barPercent}%`}`
                : undefined
            }
          >
            <span
              style={
                barPercent === null
                  ? undefined
                  : { width: `${barPercent}%`, minWidth: 0 }
              }
            />
          </div>
        )}
        {progress ? (
          <>
            {whole && (
              <p>
                전체 진행률 <strong>{whole.percent}%</strong>
                {whole.remainingSeconds !== null && (
                  <>
                    {" "}
                    · 남은 예상 약{" "}
                    <strong>{formatDuration(whole.remainingSeconds)}</strong>
                  </>
                )}
              </p>
            )}
            <p>
              현재 단계:{" "}
              <strong>{STAGES[progress.stage] ?? "작업 처리"}</strong>
              {percent !== null && (
                <>
                  {" "}
                  · 단계 진행률 <strong>{percent}%</strong>
                </>
              )}
            </p>
            {progress.current_item && (
              <p className="break-words">{progress.current_item}</p>
            )}
            {(total > 0 || completed > 0) && (
              <p>
                {active ? "처리량" : "마지막 처리량"}:{" "}
                {completed.toLocaleString()}
                {total > 0 ? ` / ${total.toLocaleString()}` : ""}{" "}
                {UNITS[progress.unit] ?? "항목"}
              </p>
            )}
            {elapsed !== null && (
              <p>
                경과: {Math.floor(elapsed / 60)}분 {elapsed % 60}초
              </p>
            )}
            {active && (
              <small>
                {whole
                  ? "전체 진행률과 남은 시간은 단계별 예상 작업량과 경과 시간으로 추정한 값입니다."
                  : percent === null
                    ? "전체 처리량을 알 수 없는 구간입니다. 현재 단계를 표시합니다."
                    : "현재 단계 기준 진행률입니다. 단계가 바뀌면 처리량이 새로 계산됩니다."}
              </small>
            )}
          </>
        ) : active ? (
          <p>
            {job.status === "queued"
              ? job.wait_reason === "validation"
                ? "전체 검사 결과를 기다리는 중입니다. 통과하면 자동으로 전달합니다."
                : job.wait_reason === "credentials"
                  ? "임시 비밀번호 준비를 기다리는 중입니다."
                  : job.queue_position != null
                    ? `${job.queue_name.toUpperCase()} 대기열 ${job.queue_position}번째 · 작업자 배정을 기다리고 있습니다.`
                    : "작업자 배정을 기다리고 있습니다. 아직 처리가 시작되지 않았습니다."
              : "작업이 실행 중입니다. 아직 세부 처리량이 보고되지 않아 진행률을 계산할 수 없습니다."}
          </p>
        ) : (
          <p>
            {job.status === "succeeded"
              ? "작업이 완료됐습니다."
              : "작업이 완료되지 않았습니다."}
          </p>
        )}
        {stale && active && (
          <p>마지막 확인 상태입니다. 연결 복구 후 자동 갱신합니다.</p>
        )}
        <JobCancelControl
          job={job}
          currentProfileId={currentProfileId}
          onJobUpdate={onJobUpdate}
        />
        {!compact && job.error_code && <p>{publicJobError(job.error_code)}</p>}
        {job.validation_job_id && (
          <Link
            href={`/validate?dataset=${encodeURIComponent(String(job.payload.dataset_id ?? ""))}&job=${encodeURIComponent(job.validation_job_id)}#validation-${encodeURIComponent(job.validation_job_id)}`}
            className="underline"
          >
            선행 검사 보고서 확인
          </Link>
        )}
        {job.kind === "datasets.empty_trash" && job.result && (
          <p role="status">
            삭제{" "}
            {Array.isArray(job.result.deleted) ? job.result.deleted.length : 0}
            개 · 실패{" "}
            {Array.isArray(job.result.failed) ? job.result.failed.length : 0}개
            · 제외{" "}
            {Array.isArray(job.result.skipped) ? job.result.skipped.length : 0}
            개
          </p>
        )}
        {links.length > 0 && (
          <nav aria-label="작업 결과" className="flex flex-wrap gap-3">
            {links.map((link) => (
              <Link key={link.href} className="underline" href={link.href}>
                {link.label}
              </Link>
            ))}
          </nav>
        )}
      </div>
      {!compact && (
        <footer className="flex flex-wrap gap-3">
          <span>작업 {job.id.slice(0, 8)}</span>
          <Link className="underline" href="/jobs">
            작업 기록 보기
          </Link>
        </footer>
      )}
    </section>
  );
}

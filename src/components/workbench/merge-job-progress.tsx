"use client";

import Link from "next/link";
import {
  isActiveJob,
  publicJobError,
  type Job,
  type DatasetSummary,
} from "@/lib/workbench-api";
import { JobCancelControl } from "./job-cancel-control";

const STAGES: Record<string, string> = {
  preparing: "원본 확인·호환성 확인",
  read: "에피소드 읽기",
  write: "데이터 기록",
  video: "영상 처리",
  statistics: "통계 계산",
  validate: "출력 구조 확인",
  publish: "NAS 저장·무결성 확인",
  register: "라이브러리 등록",
  complete: "처리 완료",
};
const UNITS: Record<string, string> = {
  episodes: "에피소드",
  frames: "프레임",
  frame_operations: "프레임 처리(디코딩+인코딩)",
  files: "파일",
  bytes: "바이트",
  items: "항목",
};

export function MergeJobProgress({
  job,
  datasets,
  currentProfileId,
  onJobUpdate,
  stale = false,
}: {
  job: Job;
  datasets: DatasetSummary[];
  currentProfileId?: string;
  onJobUpdate?: (job: Job) => void;
  stale?: boolean;
}) {
  const active = isActiveJob(job.status);
  const progress = job.status === "queued" ? null : job.progress;
  const total =
    progress && Number.isFinite(progress.total) ? progress.total : 0;
  const completed =
    progress && Number.isFinite(progress.completed)
      ? Math.max(0, progress.completed)
      : 0;
  const percent =
    total > 0 ? Math.min(100, Math.floor((completed / total) * 100)) : null;
  const title =
    stale && active
      ? "상태 확인 지연"
      : job.status === "queued"
        ? "작업자 배정 대기"
        : job.status === "running"
          ? "합치기 진행 중"
          : job.status === "succeeded"
            ? "합치기 완료"
            : job.status === "failed"
              ? "합치기 실패"
              : "합치기 중단";
  const outputName =
    typeof job.payload.output_name === "string"
      ? job.payload.output_name
      : "합치기 작업";
  const sources = Array.isArray(job.payload.sources) ? job.payload.sources : [];
  const names = sources.map((source) => {
    const id =
      typeof source === "object" && source !== null && "id" in source
        ? String(source.id)
        : String(source);
    return datasets.find((dataset) => dataset.id === id)?.name ?? id;
  });
  const output = job.result?.output;
  const episodes =
    output && typeof output === "object" && "episodes" in output
      ? output.episodes
      : null;
  const elapsed =
    progress && Number.isFinite(progress.elapsed_seconds)
      ? Math.max(0, Math.floor(progress.elapsed_seconds))
      : null;
  return (
    <article
      className="merge-job-progress"
      aria-label={`${outputName} 합치기 작업`}
    >
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="break-all font-semibold">{outputName}</h3>
        <strong role="status">{title}</strong>
      </header>
      {names.length > 0 && (
        <p className="break-words">입력: {names.join(" + ")}</p>
      )}
      <div
        className="validation-progress"
        data-active={active && !stale}
        data-determinate={percent !== null}
        data-status={job.status}
      >
        {active && (
          <div
            className="validation-progress__track"
            role="progressbar"
            aria-label="합치기 현재 단계 진행률"
            aria-valuemin={percent === null ? undefined : 0}
            aria-valuemax={percent === null ? undefined : 100}
            aria-valuenow={percent ?? undefined}
            aria-valuetext={
              stale
                ? `마지막 확인 시점의 처리량${percent === null ? "" : ` ${percent}%`}`
                : undefined
            }
          >
            <span
              style={
                percent === null
                  ? undefined
                  : { width: `${percent}%`, minWidth: 0 }
              }
            />
          </div>
        )}
        {progress ? (
          <>
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
            {total > 0 && (
              <p>
                {active ? "처리량" : "마지막 처리량"}:{" "}
                {completed.toLocaleString()} / {total.toLocaleString()}{" "}
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
                진행률은 현재 단계 기준입니다. 단계가 바뀌면 처리량이 새로
                계산됩니다.
              </small>
            )}
          </>
        ) : active ? (
          <p>
            {job.status === "queued"
              ? "작업자 배정을 기다리고 있습니다. 아직 합치기가 시작되지 않았습니다."
              : "현재 작업자는 세부 처리량을 보고하지 않습니다. 실행 상태만 표시하며, 완료 여부를 계속 확인합니다."}
          </p>
        ) : null}
        {stale && active && (
          <p>
            상태를 갱신하지 못했습니다. 마지막 확인 상태를 표시하며 자동으로
            재연결합니다.
          </p>
        )}
        <JobCancelControl
          job={job}
          currentProfileId={currentProfileId}
          onJobUpdate={onJobUpdate}
        />
        {job.status === "succeeded" && (
          <p>
            새 데이터셋 저장이 완료됐습니다.
            {typeof episodes === "number"
              ? ` 에피소드 ${episodes.toLocaleString()}개.`
              : ""}{" "}
            <Link className="underline" href="/library">
              라이브러리에서 확인
            </Link>
          </p>
        )}
        {job.error_code && <p role="alert">{publicJobError(job.error_code)}</p>}
        {!active && job.status !== "succeeded" && (
          <p>
            작업이 완료되지 않았습니다. 아래 작업 기록에서 상태를 확인하세요.
          </p>
        )}
      </div>
      <footer className="flex flex-wrap gap-3 text-xs">
        <time dateTime={job.created_at}>
          {new Date(job.created_at).toLocaleString("ko-KR")}
        </time>
        <span>작업 {job.id.slice(0, 8)}</span>
        <Link className="underline" href="/jobs">
          작업 기록 보기
        </Link>
      </footer>
    </article>
  );
}

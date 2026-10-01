"use client";

import { useState } from "react";
import { ValidationProgress } from "./validation-progress";
import { publicJobError, type ValidationRun } from "@/lib/workbench-api";

export function ValidationRunCard({
  run,
  stale,
  onDelete,
}: {
  run: ValidationRun;
  stale: boolean;
  onDelete: (run: ValidationRun) => Promise<void>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = run.status === "queued" || run.status === "running";
  const issues = run.result?.issues ?? [];
  const label = {
    quick: "빠른 검사",
    full: "전체 검사",
    export_gate: "내보내기 검사",
  }[run.mode];

  async function remove() {
    if (active || deleting) return;
    setDeleting(true);
    setError(null);
    try {
      await onDelete(run);
      setConfirming(false);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "검사 기록을 삭제하지 못했습니다.",
      );
    } finally {
      setDeleting(false);
    }
  }

  return (
    <article className="validation-run-card" aria-label={`${label} 기록`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <strong>{label}</strong>
        <time>{new Date(run.created_at).toLocaleString("ko-KR")}</time>
        <button
          type="button"
          className="workbench-button"
          disabled={active || deleting}
          onClick={() => setConfirming(true)}
          title={
            active
              ? "진행 중인 검사는 삭제할 수 없습니다."
              : "이 검사 기록 삭제"
          }
        >
          {deleting ? "삭제 중…" : "기록 삭제"}
        </button>
      </div>
      {active ? (
        <ValidationProgress run={run} stale={stale} />
      ) : (
        <details className="rounded border border-current/15 p-3">
          <summary className="cursor-pointer font-semibold">
            검사 결과 ·{" "}
            {run.result
              ? `실패 ${run.result.failures} · 경고 ${run.result.warnings}`
              : "결과 확인"}
          </summary>
          <ValidationProgress run={run} stale={stale} />
          {run.error_code && (
            <p role="alert">{publicJobError(run.error_code)}</p>
          )}
        </details>
      )}
      {issues.length > 0 && (
        <details className="rounded border border-current/15 p-3">
          <summary className="cursor-pointer font-semibold">
            문제 상세 · {issues.length}건
          </summary>
          <ul className="mt-3 space-y-3 text-sm" aria-label="검증 문제 상세">
            {issues.map((issue, index) => (
              <li key={`${issue.code}-${index}`}>
                <strong>
                  {issue.severity} · {issue.code}
                </strong>
                {issue.episode_index !== undefined && (
                  <span> · 에피소드 {issue.episode_index}</span>
                )}
                <p className="whitespace-pre-line break-words leading-relaxed">
                  {issue.message}
                </p>
              </li>
            ))}
          </ul>
        </details>
      )}
      {confirming && (
        <div
          className="rounded border border-current/20 p-3"
          role="group"
          aria-label="검사 기록 삭제 확인"
        >
          <p>
            이 기록을 검사 목록에서 삭제할까요? 모든 사용자 목록에 적용됩니다.
            데이터셋 원본은 삭제하지 않습니다.
          </p>
          <p>
            내보내기 검사 기록 삭제 시 재검사가 필요할 수 있습니다. 작업 로그는
            안전 확인용으로 보존됩니다.
          </p>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              className="workbench-button"
              disabled={deleting}
              onClick={() => void remove()}
            >
              삭제 확인
            </button>
            <button
              type="button"
              className="workbench-button"
              disabled={deleting}
              onClick={() => {
                setConfirming(false);
                setError(null);
              }}
            >
              취소
            </button>
          </div>
        </div>
      )}
      {error && <p role="alert">{error}</p>}
    </article>
  );
}

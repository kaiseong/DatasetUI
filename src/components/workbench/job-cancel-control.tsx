"use client";

import { useEffect, useState } from "react";
import { LuLoaderCircle, LuX } from "react-icons/lu";
import {
  cancelJob,
  getJob,
  WorkbenchApiError,
  type Job,
} from "@/lib/workbench-api";

export function JobCancelControl({
  job,
  currentProfileId,
  onJobUpdate,
}: {
  job: Job;
  currentProfileId?: string;
  onJobUpdate?: (job: Job) => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const cancellationRequested = Boolean(job.cancellation_requested);
  const cancellationGuarded =
    job.status === "running" && Boolean(job.cancellation_guarded_at);
  const canCancel =
    (job.status === "queued" || job.status === "running") &&
    !cancellationRequested &&
    !cancellationGuarded &&
    Boolean(currentProfileId) &&
    job.profile_id === currentProfileId;

  useEffect(() => {
    setConfirming(false);
    setPending(false);
    setMessage(null);
  }, [
    job.id,
    job.status,
    job.cancellation_requested,
    job.cancellation_guarded_at,
  ]);

  if (!canCancel && !message && !cancellationRequested && !cancellationGuarded)
    return null;

  async function submitCancellation() {
    if (!currentProfileId || pending) return;
    setPending(true);
    setMessage(null);
    try {
      const updated = await cancelJob(job.id, currentProfileId);
      setConfirming(false);
      setMessage(
        updated.status === "running"
          ? "취소 요청을 전달했습니다. 안전한 중단 지점까지 잠시 걸릴 수 있습니다."
          : "대기 중인 작업을 취소했습니다.",
      );
      onJobUpdate?.(updated);
    } catch (error) {
      if (error instanceof WorkbenchApiError && error.status === 409) {
        setMessage(
          "작업이 이미 완료됐거나 최종 게시 단계에 들어가 취소할 수 없습니다. 최신 상태를 다시 확인합니다.",
        );
        try {
          onJobUpdate?.(await getJob(job.id));
        } catch {
          // The normal job poll will retry if this immediate refresh fails.
        }
      } else {
        setMessage(
          "작업 취소 요청을 완료하지 못했습니다. 잠시 후 다시 시도하세요.",
        );
      }
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="job-cancel-control" aria-live="polite">
      {cancellationRequested && job.status === "running" && (
        <p className="job-cancel-control__requested" role="status">
          <LuLoaderCircle className="animate-spin" aria-hidden /> 취소 요청 중 ·
          안전하게 중단할 시점을 확인하고 있습니다.
        </p>
      )}
      {cancellationGuarded && !cancellationRequested && (
        <p className="job-cancel-control__guarded" role="status">
          마무리 중 · 결과 게시 단계에 들어가 취소할 수 없습니다.
        </p>
      )}
      {canCancel && !confirming && (
        <button
          type="button"
          className="job-cancel-control__trigger"
          onClick={() => {
            setConfirming(true);
            setMessage(null);
          }}
        >
          <LuX aria-hidden />
          {job.status === "running" ? "실행 작업 취소" : "대기 작업 취소"}
        </button>
      )}
      {canCancel && confirming && (
        <div className="job-cancel-control__confirm" role="group">
          <span>
            {job.status === "running"
              ? "실행 중인 작업에 안전한 중단을 요청할까요? 생성 중인 결과는 게시되지 않습니다."
              : "아직 시작되지 않은 이 작업을 취소할까요?"}
          </span>
          <button
            type="button"
            className="job-cancel-control__confirm-button"
            disabled={pending}
            onClick={() => void submitCancellation()}
          >
            {pending ? (
              <>
                <LuLoaderCircle className="animate-spin" aria-hidden /> 취소 중…
              </>
            ) : (
              "취소하기"
            )}
          </button>
          <button
            type="button"
            className="job-cancel-control__keep-button"
            disabled={pending}
            onClick={() => setConfirming(false)}
          >
            유지
          </button>
        </div>
      )}
      {message && (
        <p
          className="job-cancel-control__message"
          role={message.includes("취소했습니다") ? "status" : "alert"}
        >
          {message}
        </p>
      )}
    </div>
  );
}

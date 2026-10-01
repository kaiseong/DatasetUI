"use client";

import { LuBadgeCheck } from "react-icons/lu";
import type { SegmentationPreviewResult } from "@/lib/segmentation-api";

type Props = {
  preview: SegmentationPreviewResult;
  maxFrame: number;
  setFrameIndex: (frame: number) => void;
  approvalToken: string | null;
  approvePreview: () => Promise<void>;
};

export function PreviewReview({
  preview,
  maxFrame,
  setFrameIndex,
  approvalToken,
  approvePreview,
}: Props) {
  return (
    <div className="space-y-4">
      {preview.selection_required && (
        <p className="workbench-live-message" role="alert">
          후보 선택 전입니다. 객체를 선택해 다시 렌더링하기 전에는 승인할 수
          없습니다.
        </p>
      )}
      {preview.warnings?.map((warning) => (
        <p key={warning} className="workbench-live-message" role="alert">
          검토 필요: {warning}
        </p>
      ))}
      {!!preview.review_signals?.length && (
        <div className="segmentation-review-signals">
          <strong>
            직접 확인할 프레임 · 자동 품질 신호 (모델 신뢰도 아님)
          </strong>
          <p>
            빈 영역·화면 전체 선택·면적 급변을 표시합니다. 케이블·포트·접촉 부위
            누락과 깜빡임을 영상에서 확인하세요.
          </p>
          <div>
            {preview.review_signals.map((signal) => (
              <button
                key={`${signal.frame_index}:${signal.reason}`}
                type="button"
                className="workbench-button"
                onClick={() =>
                  setFrameIndex(Math.min(maxFrame, signal.frame_index))
                }
              >
                Frame {signal.frame_index} ·{" "}
                {signal.reason === "empty_mask"
                  ? "남길 영역 없음"
                  : signal.reason === "full_frame_mask"
                    ? "화면 전체 선택"
                    : signal.reason === "reentry"
                      ? `재등장 추정 · 후보 ${signal.candidate_id ?? ""} 확인`
                      : signal.reason === "auto_selected"
                        ? `자동 선택 · 후보 ${signal.candidate_id ?? ""} 확인`
                        : signal.reason === "object_gap"
                          ? `객체 ${signal.object_id ?? ""} 누락 시작 · ${signal.missing_frames ?? 0}프레임 미감지`
                          : "면적 급변"}
              </button>
            ))}
          </div>
        </div>
      )}
      {preview.review_blocked && (
        <p className="workbench-live-message" role="alert">
          모든 프레임에 남길 영역이 없습니다. 보존 대상을 수정한 뒤 다시
          생성해야 합니다.
        </p>
      )}
      <button
        type="button"
        className="workbench-button"
        disabled={
          !!approvalToken ||
          preview.selection_required ||
          preview.review_blocked
        }
        onClick={() => void approvePreview()}
      >
        <LuBadgeCheck aria-hidden />{" "}
        {approvalToken ? "승인됨" : "이 미리보기 승인"}
      </button>
    </div>
  );
}

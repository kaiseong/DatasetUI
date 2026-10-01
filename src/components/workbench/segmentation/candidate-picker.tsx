"use client";

import {
  segmentationArtifactUrl,
  type SegmentationPreviewResult,
} from "@/lib/segmentation-api";
import type { SegmentationDraft } from "@/lib/segmentation-draft";

type Props = {
  cacheSource: { preview: SegmentationPreviewResult; signature: string };
  draft: SegmentationDraft;
  editRender: (update: (draft: SegmentationDraft) => SegmentationDraft) => void;
};

export function CandidatePicker({ cacheSource, draft, editRender }: Props) {
  return (
    <section
      className="segmentation-candidates"
      aria-label="보존할 객체 후보 선택"
    >
      <h3>보존할 객체를 직접 선택하세요</h3>
      <p>
        텍스트가 찾은 후보를 자동으로 합치지 않습니다. 원하는 객체를 선택하고
        저장된 마스크로 다시 렌더링하세요.
      </p>
      <div>
        {(cacheSource.preview.candidates ?? []).map((candidate) => (
          <label
            key={candidate.candidate_id}
            className="segmentation-candidate"
          >
            {/* eslint-disable-next-line @next/next/no-img-element -- same-origin mask artifact */}
            <img
              src={segmentationArtifactUrl(
                cacheSource.preview.preview_id,
                candidate.artifact_name ??
                  `candidate-${candidate.candidate_id}.png`,
              )}
              alt={`객체 ${candidate.object_id} 후보 ${candidate.candidate_id} 마스크`}
            />
            <span>
              <input
                type="checkbox"
                checked={(draft.selected_candidate_ids ?? []).includes(
                  candidate.candidate_id,
                )}
                onChange={(event) =>
                  editRender((current) => ({
                    ...current,
                    selected_candidate_ids: event.target.checked
                      ? [
                          ...(current.selected_candidate_ids ?? []),
                          candidate.candidate_id,
                        ]
                      : (current.selected_candidate_ids ?? []).filter(
                          (id) => id !== candidate.candidate_id,
                        ),
                  }))
                }
              />{" "}
              객체 {candidate.object_id} · 후보 {candidate.candidate_id}
              {candidate.auto_selected ? " · 자동 선택됨" : ""}
              {candidate.reentry
                ? " · 재등장 추정"
                : candidate.late_track
                  ? ` · 프레임 ${candidate.first_visible_frame ?? candidate.frame_index}부터 등장`
                  : ""}
            </span>
          </label>
        ))}
      </div>
    </section>
  );
}

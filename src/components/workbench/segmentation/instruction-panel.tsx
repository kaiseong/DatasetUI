"use client";

import { JobProgress } from "../job-progress";
import type { Job } from "@/lib/workbench-api";
import {
  sampleArtifactUrl,
  type SegmentationPrompt,
  type SegmentationSampleResult,
} from "@/lib/segmentation-api";
import {
  candidateGuidanceSignature,
  type SegmentationDraft,
} from "@/lib/segmentation-draft";
import { TERMINAL, type EditorPanel } from "./support";

type Props = {
  panel: EditorPanel;
  instructionPrompt: SegmentationPrompt;
  editInstruction: (
    update: (prompt: SegmentationPrompt) => SegmentationPrompt,
  ) => void;
  editingObject: boolean;
  candidateBusy: boolean;
  frameSource: string;
  findCandidates: () => Promise<void>;
  frameIndex: number;
  setFrameIndex: (frame: number) => void;
  candidateJob: Job | null;
  candidateSample: SegmentationSampleResult | null;
  draft: SegmentationDraft;
  objectId: number;
  profileId: string;
};

export function InstructionPanel({
  panel,
  instructionPrompt,
  editInstruction,
  editingObject,
  candidateBusy,
  frameSource,
  findCandidates,
  frameIndex,
  setFrameIndex,
  candidateJob,
  candidateSample,
  draft,
  objectId,
  profileId,
}: Props) {
  return (
    <div hidden={panel !== "instruction"} role="tabpanel">
      <label className="block text-xs text-slate-300">
        <span className="mb-1 block font-medium">SAM3.1 분할 프롬프트</span>
        <input
          className="w-full rounded border border-white/15 bg-slate-950 px-3 py-2 text-sm"
          value={instructionPrompt.text}
          onChange={(event) =>
            editInstruction((current) => ({
              ...current,
              text: event.target.value,
              confidence_threshold: current.confidence_threshold ?? 0.5,
              selected_candidates: [],
            }))
          }
          placeholder="black plug"
        />
      </label>
      <label className="block my-3">
        최소 SAM 탐지 점수
        <input
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={instructionPrompt.confidence_threshold ?? 0.5}
          className="w-full bg-slate-950 p-2"
          onChange={(event) =>
            editInstruction((p) => ({
              ...p,
              confidence_threshold: Math.min(
                1,
                Math.max(0, Number(event.target.value)),
              ),
              selected_candidates: [],
            }))
          }
        />
      </label>
      <button
        className="workbench-button"
        disabled={
          !editingObject ||
          candidateBusy ||
          !instructionPrompt.text.trim() ||
          !frameSource
        }
        onClick={() => void findCandidates()}
      >
        {candidateBusy
          ? "후보 탐지 중…"
          : instructionPrompt.text.trim() &&
              instructionPrompt.frame_index !== frameIndex
            ? "현재 프레임에서 다시 탐지"
            : "후보 찾기"}
      </button>
      {!!instructionPrompt.text.trim() &&
        instructionPrompt.frame_index !== frameIndex && (
          <p className="my-2 text-xs text-slate-300">
            탐지 프레임 {instructionPrompt.frame_index}
            {instructionPrompt.selected_candidates?.length
              ? ` · 선택한 후보 ${instructionPrompt.selected_candidates.length}개`
              : " · 후보 미선택"}{" "}
            <button
              type="button"
              className="underline underline-offset-4"
              onClick={() => setFrameIndex(instructionPrompt.frame_index)}
            >
              그 프레임으로 이동
            </button>
            <span className="block text-slate-400">
              이 객체는 모든 프레임에 적용됩니다. 다른 프레임의 샘플은 탐지
              프레임부터 추적해 보여줍니다.
            </span>
          </p>
        )}
      {candidateBusy && candidateJob && !TERMINAL.has(candidateJob.status) && (
        <JobProgress job={candidateJob} compact />
      )}
      <p className="text-xs text-slate-400">
        SAM 탐지 점수는 정답 확률이 아닙니다. 여러 후보를 선택하면 하나의
        그룹으로 저장됩니다.
      </p>
      {candidateSample?.selection_signature ===
        candidateGuidanceSignature(draft, objectId, frameIndex) &&
        candidateSample.candidates.map((candidate) => (
          <label
            key={candidate.candidate_id}
            className="flex gap-2 items-center my-2"
          >
            <input
              type="checkbox"
              checked={
                !!instructionPrompt.selected_candidates?.some(
                  (r) => r.candidate_id === candidate.candidate_id,
                )
              }
              onChange={(event) =>
                editInstruction((p) => ({
                  ...p,
                  selected_candidates: event.target.checked
                    ? [
                        ...(p.selected_candidates ?? []),
                        {
                          sample_id: candidateSample.sample_id,
                          candidate_id: candidate.candidate_id,
                        },
                      ]
                    : (p.selected_candidates ?? []).filter(
                        (r) => r.candidate_id !== candidate.candidate_id,
                      ),
                }))
              }
            />
            {/* eslint-disable-next-line @next/next/no-img-element -- authenticated candidate artifact */}
            <img
              width={80}
              alt={`후보 ${candidate.candidate_id}`}
              src={sampleArtifactUrl(
                candidateSample.sample_id,
                candidate.artifact_name,
                profileId,
              )}
            />
            <span>
              후보 {candidate.candidate_id} · SAM{" "}
              {candidate.detection_score?.toFixed(2) ?? "점수 없음"}
              {candidate.manually_refined ? " · 수동 보정됨" : ""}
              {candidateSample.candidates.filter(
                (item) => item.object_id === objectId,
              ).length === 1
                ? " · 유일한 후보"
                : ""}
            </span>
          </label>
        ))}
      <p className="my-2 text-xs text-slate-400">
        객체를 찾을 단어·짧은 문장입니다. 학습 데이터셋의 instruction은 변경하지
        않습니다.
      </p>
      <button
        className="workbench-button"
        onClick={() =>
          editInstruction((current) => ({
            ...current,
            text: "",
            selected_candidates: [],
          }))
        }
      >
        현재 지시 삭제
      </button>
    </div>
  );
}

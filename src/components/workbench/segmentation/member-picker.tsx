"use client";

import { forwardRef } from "react";
import {
  sampleArtifactUrl,
  type SegmentationSampleResult,
} from "@/lib/segmentation-api";

type CandidateRef = { sample_id: string; candidate_id: string };

type Props = {
  candidates: CandidateRef[];
  value: string;
  onChange: (candidateId: string) => void;
  candidateSample: SegmentationSampleResult | null;
  profileId: string;
  /** Bumped whenever a blocked canvas click asks for a choice; replays the cue. */
  attention: number;
};

function thumbnailName(
  ref: CandidateRef,
  sample: SegmentationSampleResult | null,
): string {
  const match =
    sample?.sample_id === ref.sample_id
      ? sample.candidates.find((item) => item.candidate_id === ref.candidate_id)
      : undefined;
  return match?.artifact_name ?? `candidate-${ref.candidate_id}.png`;
}

/** Which Instruction candidate Labeling points, boxes and brush refine. */
export const MemberPicker = forwardRef<HTMLDivElement, Props>(
  function MemberPicker(
    { candidates, value, onChange, candidateSample, profileId, attention },
    ref,
  ) {
    const needsChoice = candidates.length > 1 && !value;
    return (
      <div
        ref={ref}
        // Remount on each blocked click so the cue animation plays again.
        key={needsChoice ? `cue-${attention}` : "chosen"}
        role="radiogroup"
        aria-label="보정할 그룹 내 후보"
        className={`rounded-lg border p-3 transition-colors ${
          needsChoice
            ? "animate-pulse border-cyan-300 bg-cyan-400/10 ring-2 ring-cyan-300/70"
            : "border-white/10 bg-black/15"
        }`}
      >
        <p className="mb-2 text-xs">
          {needsChoice ? (
            <strong className="text-cyan-200">
              Labeling 전에 보정할 후보를 고르세요
            </strong>
          ) : (
            <span className="text-slate-400">
              Labeling이 보정할 후보
              {candidates.length === 1 ? " · 유일한 후보라 자동 선택" : ""}
            </span>
          )}
        </p>
        <div className="flex flex-wrap gap-2">
          {candidates.map((candidate) => {
            const selected = candidate.candidate_id === value;
            return (
              <button
                key={candidate.candidate_id}
                type="button"
                role="radio"
                aria-checked={selected}
                // A single candidate is always the one refined.
                disabled={candidates.length === 1}
                onClick={() => onChange(candidate.candidate_id)}
                className={`flex items-center gap-2 rounded-md border px-2 py-1 text-xs ${
                  selected
                    ? "border-cyan-300 bg-cyan-400/20 text-cyan-100"
                    : "border-white/15 hover:border-cyan-300/60"
                }`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element -- authenticated candidate artifact */}
                <img
                  width={48}
                  alt=""
                  className="rounded"
                  src={sampleArtifactUrl(
                    candidate.sample_id,
                    thumbnailName(candidate, candidateSample),
                    profileId,
                  )}
                  onError={(event) => {
                    event.currentTarget.hidden = true;
                  }}
                />
                후보 {candidate.candidate_id}
              </button>
            );
          })}
        </div>
      </div>
    );
  },
);

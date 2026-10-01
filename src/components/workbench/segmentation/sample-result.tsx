"use client";

import {
  sampleArtifactUrl,
  type SegmentationSampleResult,
} from "@/lib/segmentation-api";

type Props = {
  sample: SegmentationSampleResult;
  profileId: string;
  sampleView: string;
  setSampleView: (view: string) => void;
};

export function SampleResult({
  sample,
  profileId,
  sampleView,
  setSampleView,
}: Props) {
  return (
    <section aria-label="샘플 이미지 결과">
      <p className="text-xs text-slate-400">
        프레임 {sample.source_frame_index} 샘플 · 영상 승인·내보내기에는
        사용되지 않습니다.
      </p>
      <div className="my-2 flex gap-2">
        {[
          ["original.png", "원본"],
          ["mask.png", "마스크"],
          ["composite.png", "합성"],
        ].map(([name, label]) => (
          <button
            key={name}
            className="workbench-button"
            aria-pressed={sampleView === name}
            onClick={() => setSampleView(name)}
          >
            {label}
          </button>
        ))}
      </div>
      {/* eslint-disable-next-line @next/next/no-img-element -- immutable sample artifact */}
      <img
        className="w-full"
        src={sampleArtifactUrl(sample.sample_id, sampleView, profileId)}
        alt="현재 프레임 분할 샘플"
      />
      <div className="flex gap-2">
        {sample.candidates.map((candidate) => (
          <button
            key={candidate.candidate_id}
            className="workbench-button"
            onClick={() => setSampleView(candidate.artifact_name)}
          >
            후보 {candidate.candidate_id} 확인
          </button>
        ))}
      </div>
    </section>
  );
}

"use client";

import {
  segmentationArtifactUrl,
  type SegmentationPreviewResult,
} from "@/lib/segmentation-api";
import type { SegmentationDraft } from "@/lib/segmentation-draft";

type Props = {
  preview: SegmentationPreviewResult;
  episodeIndex: number;
  videoKey: string;
  renderMode: SegmentationDraft["render_mode"];
};

export function EpisodeVideos({
  preview,
  episodeIndex,
  videoKey,
  renderMode,
}: Props) {
  return (
    <section aria-label="에피소드 영상 결과">
      {" "}
      <div className="grid gap-4 lg:grid-cols-2">
        <figure>
          <figcaption className="mb-2 text-xs font-medium text-slate-300">
            원본 · Episode {episodeIndex} / {videoKey}
          </figcaption>
          <video
            className="w-full rounded-lg border border-white/15 bg-black"
            controls
            muted
            preload="metadata"
            src={segmentationArtifactUrl(preview.preview_id, "original.mp4")}
          />
        </figure>
        <figure>
          <figcaption className="mb-2 text-xs font-medium text-cyan-200">
            작업 영역 · {renderMode === "image" ? "사진 배경" : "검은 배경"}
          </figcaption>
          <video
            className="w-full rounded-lg border border-cyan-400/30 bg-black"
            controls
            muted
            preload="metadata"
            src={segmentationArtifactUrl(preview.preview_id, "composite.mp4")}
          />
        </figure>
      </div>
      <details className="text-sm text-slate-300">
        <summary className="cursor-pointer">Mask video 보기</summary>
        <video
          className="mt-2 w-full max-w-xl rounded-lg border border-white/15 bg-black"
          controls
          muted
          preload="metadata"
          src={segmentationArtifactUrl(preview.preview_id, "mask.mp4")}
        />
      </details>
    </section>
  );
}

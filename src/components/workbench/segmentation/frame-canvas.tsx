"use client";

import type {
  Dispatch,
  PointerEvent as ReactPointerEvent,
  SetStateAction,
} from "react";
import type {
  SegmentationCorrection,
  SegmentationPrompt,
  SegmentationTarget,
} from "@/lib/segmentation-api";

type Props = {
  selectionLoading: boolean;
  frameLoading: boolean;
  frameSource: string;
  setSelectionAttempt: Dispatch<SetStateAction<number>>;
  frameUrl: string;
  datasetName: string;
  episodeIndex: number;
  videoKey: string;
  frameIndex: number;
  setFrameAspect: (aspect: number) => void;
  frameAspect: number;
  handlePointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  handlePointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void;
  handlePointerUp: (event: ReactPointerEvent<HTMLDivElement>) => void;
  shownBox: [number, number, number, number] | null | undefined;
  target: SegmentationTarget;
  corrections: SegmentationCorrection[];
  prompt: SegmentationPrompt;
  pointRadius: number;
};

export function FrameCanvas({
  selectionLoading,
  frameLoading,
  frameSource,
  setSelectionAttempt,
  frameUrl,
  datasetName,
  episodeIndex,
  videoKey,
  frameIndex,
  setFrameAspect,
  frameAspect,
  handlePointerDown,
  handlePointerMove,
  handlePointerUp,
  shownBox,
  target,
  corrections,
  prompt,
  pointRadius,
}: Props) {
  return (
    <>
      {(selectionLoading || frameLoading) && (
        <p role="status" className="p-4 text-sm">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-cyan-300 border-t-transparent" />{" "}
          {selectionLoading
            ? "선택한 에피소드·카메라 준비 중…"
            : "선택한 프레임 로딩 중…"}
        </p>
      )}
      {!selectionLoading && !frameSource && (
        <button
          className="workbench-button"
          onClick={() => setSelectionAttempt((value) => value + 1)}
        >
          로딩 재시도
        </button>
      )}
      <div className="relative overflow-hidden rounded-lg border border-white/15 bg-slate-950">
        {/* eslint-disable-next-line @next/next/no-img-element -- authenticated same-origin frame endpoint */}
        <img
          key={frameUrl}
          hidden={!frameSource}
          src={frameSource || undefined}
          alt={`${datasetName} episode ${episodeIndex}, ${videoKey}, frame ${frameIndex}`}
          className="block h-auto w-full select-none"
          draggable={false}
          onLoad={(event) => {
            const image = event.currentTarget;
            if (image.naturalHeight > 0) {
              setFrameAspect(image.naturalWidth / image.naturalHeight);
            }
          }}
        />
        <div
          className={`absolute inset-0 cursor-crosshair touch-none ${!frameSource || frameLoading ? "pointer-events-none hidden" : ""}`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        >
          <svg
            className="h-full w-full"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            aria-hidden
          >
            {shownBox && (
              <rect
                x={shownBox[0] * 100}
                y={shownBox[1] * 100}
                width={shownBox[2] * 100}
                height={shownBox[3] * 100}
                fill="rgba(56,189,248,.12)"
                stroke={target === "protect" ? "#c084fc" : "#fb7185"}
                strokeWidth="0.5"
              />
            )}
            {corrections.flatMap((correction) =>
              correction.points.map((point, index) => (
                <ellipse
                  key={`${correction.operation}-${point.x}-${point.y}-${index}`}
                  cx={point.x * 100}
                  cy={point.y * 100}
                  rx={correction.radius * 100}
                  ry={correction.radius * 100 * frameAspect}
                  fill={
                    correction.operation === "erase"
                      ? "rgba(251,113,133,.18)"
                      : target === "protect"
                        ? "rgba(192,132,252,.28)"
                        : "rgba(56,189,248,.25)"
                  }
                  stroke={
                    correction.operation === "erase"
                      ? "#fb7185"
                      : target === "protect"
                        ? "#c084fc"
                        : "#fb7185"
                  }
                  strokeDasharray={
                    correction.operation === "erase" ? "1 0.7" : undefined
                  }
                  strokeWidth="0.3"
                />
              )),
            )}
          </svg>
          {prompt.points.map((point, index) => (
            <span
              key={index}
              className="pointer-events-none absolute flex items-center justify-center rounded-full border border-slate-950 text-slate-950"
              style={{
                left: `${point.x * 100}%`,
                top: `${point.y * 100}%`,
                transform: "translate(-50%, -50%)",
                width: pointRadius * 2,
                height: pointRadius * 2,
                fontSize: pointRadius * 1.4,
                background: point.label ? "#c084fc" : "#fb7185",
              }}
            >
              {point.label ? "+" : "−"}
            </span>
          ))}
        </div>
      </div>
    </>
  );
}

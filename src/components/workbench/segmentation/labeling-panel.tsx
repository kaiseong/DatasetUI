"use client";

import {
  LuBrush,
  LuCircleMinus,
  LuCirclePlus,
  LuEraser,
  LuScan,
} from "react-icons/lu";
import type {
  SegmentationPrompt,
  SegmentationTarget,
} from "@/lib/segmentation-api";
import {
  clearFrameCorrections,
  type SegmentationDraft,
} from "@/lib/segmentation-draft";
import type { EditorPanel, Tool } from "./support";

type Props = {
  panel: EditorPanel;
  tool: Tool;
  setTool: (tool: Tool) => void;
  pointRadius: number;
  setPointRadius: (radius: number) => void;
  prompt: SegmentationPrompt;
  editCurrentPrompt: (
    update: (prompt: SegmentationPrompt) => SegmentationPrompt,
  ) => void;
  brushRadius: number;
  setBrushRadius: (radius: number) => void;
  brushOperation: "add" | "erase";
  setBrushOperation: (operation: "add" | "erase") => void;
  editDraft: (update: (draft: SegmentationDraft) => SegmentationDraft) => void;
  frameIndex: number;
  target: SegmentationTarget;
  objectId: number;
  memberCandidate: string;
};

export function LabelingPanel({
  panel,
  tool,
  setTool,
  pointRadius,
  setPointRadius,
  prompt,
  editCurrentPrompt,
  brushRadius,
  setBrushRadius,
  brushOperation,
  setBrushOperation,
  editDraft,
  frameIndex,
  target,
  objectId,
  memberCandidate,
}: Props) {
  return (
    <div hidden={panel !== "labeling"} role="tabpanel" className="space-y-4">
      <fieldset className="space-y-2 border-t border-white/10 pt-3">
        <legend>Point</legend>
        <div className="flex flex-wrap gap-2">
          {(
            [
              ["positive", "포함점 찍기 (+)"],
              ["negative", "제외점 찍기 (−)"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={`workbench-button ${tool === value ? "workbench-button--primary" : ""}`}
              aria-pressed={tool === value}
              onClick={() => setTool(value)}
            >
              {value === "positive" ? (
                <LuCirclePlus aria-hidden />
              ) : (
                <LuCircleMinus aria-hidden />
              )}
              {label}
            </button>
          ))}
        </div>
        <label className="block text-xs">
          표시 반경 · {pointRadius}px (추론에는 영향 없음)
          <input
            type="range"
            min={2}
            max={16}
            value={pointRadius}
            onChange={(event) => setPointRadius(Number(event.target.value))}
            className="w-full"
          />
        </label>
        <button
          className="workbench-button"
          disabled={!prompt.points.length}
          onClick={() =>
            editCurrentPrompt((current) => ({
              ...current,
              points: current.points.slice(0, -1),
            }))
          }
        >
          마지막 점 삭제
        </button>
        <button
          className="workbench-button"
          disabled={!prompt.points.length}
          onClick={() =>
            editCurrentPrompt((current) => ({ ...current, points: [] }))
          }
        >
          현재 객체·프레임 점 전체 삭제
        </button>
        <ol className="max-h-36 overflow-auto text-xs">
          {prompt.points.map((point, index) => (
            <li key={index} className="flex justify-between py-1">
              <span>
                {index + 1}. {point.label ? "포함" : "제외"} 점
              </span>
              <button
                aria-label={`점 ${index + 1} 삭제`}
                onClick={() =>
                  editCurrentPrompt((current) => ({
                    ...current,
                    points: current.points.filter((_, i) => i !== index),
                  }))
                }
              >
                삭제
              </button>
            </li>
          ))}
        </ol>
      </fieldset>
      <fieldset className="space-y-2 border-t border-white/10 pt-3">
        <legend>Box</legend>
        <button
          type="button"
          className={`workbench-button ${tool === "box" ? "workbench-button--primary" : ""}`}
          aria-pressed={tool === "box"}
          onClick={() => setTool("box")}
        >
          <LuScan aria-hidden /> 객체 선택 Box 그리기
        </button>
        <p className="text-xs text-slate-400">
          사각형 안의 객체를 찾습니다. 사각형 전체를 강제로 남기거나 지우지
          않습니다.
        </p>
        <button
          type="button"
          className="workbench-button"
          onClick={() =>
            editCurrentPrompt((current) => ({ ...current, box: null }))
          }
        >
          <LuEraser aria-hidden /> Box 지우기
        </button>
      </fieldset>
      <fieldset className="space-y-2 border-t border-white/10 pt-3">
        <legend>Brush</legend>
        <label className="block text-xs text-slate-300">
          Brush radius · {(brushRadius * 100).toFixed(1)}% width
          <input
            className="mt-1 w-full accent-cyan-400"
            type="range"
            min={0.005}
            max={0.1}
            step={0.005}
            value={brushRadius}
            onChange={(event) => setBrushRadius(Number(event.target.value))}
          />
        </label>
        <p className="text-xs text-slate-400">
          반경을 포함한 칠한 범위를 SAM 힌트로 사용합니다. 최종 경계는 SAM이
          판단하며 칠한 픽셀을 강제로 덧칠하지 않습니다.
        </p>
        <div className="grid grid-cols-2 gap-2" aria-label="Brush operation">
          <button
            type="button"
            className={`workbench-button ${tool === "brush" && brushOperation === "add" ? "workbench-button--primary" : ""}`}
            aria-pressed={tool === "brush" && brushOperation === "add"}
            onClick={() => {
              setTool("brush");
              setBrushOperation("add");
            }}
          >
            <LuBrush aria-hidden /> 포함 힌트 칠하기
          </button>
          <button
            type="button"
            className={`workbench-button ${tool === "brush" && brushOperation === "erase" ? "border-rose-400/60 text-rose-200" : ""}`}
            aria-pressed={tool === "brush" && brushOperation === "erase"}
            onClick={() => {
              setTool("brush");
              setBrushOperation("erase");
            }}
          >
            <LuEraser aria-hidden /> 제외 힌트 칠하기
          </button>
        </div>
        <button
          type="button"
          className="workbench-button"
          onClick={() =>
            editDraft((current) =>
              clearFrameCorrections(
                current,
                frameIndex,
                target,
                objectId,
                memberCandidate || undefined,
              ),
            )
          }
        >
          <LuEraser aria-hidden /> 현재 객체·프레임 브러시 삭제
        </button>
      </fieldset>
    </div>
  );
}

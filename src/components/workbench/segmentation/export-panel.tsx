"use client";

import Link from "next/link";
import { LuImagePlus } from "react-icons/lu";
import type { Job } from "@/lib/workbench-api";
import { OUTPUT_NAME_PATTERN, TERMINAL, jobMessage } from "./support";

type Props = {
  recomputeStatistics: boolean;
  setRecomputeStatistics: (value: boolean) => void;
  outputName: string;
  onOutputNameChange: (name: string) => void;
  exportJob: Job | null;
  approvalToken: string | null;
  exportDataset: () => Promise<void>;
};

export function ExportPanel({
  recomputeStatistics,
  setRecomputeStatistics,
  outputName,
  onOutputNameChange,
  exportJob,
  approvalToken,
  exportDataset,
}: Props) {
  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="text-sm text-slate-300">
        <input
          type="checkbox"
          checked={recomputeStatistics}
          onChange={(event) => setRecomputeStatistics(event.target.checked)}
        />{" "}
        분포 통계 재계산 (선택)
      </label>
      <label className="min-w-72 flex-1 text-xs text-slate-300">
        <span className="mb-1 block">새 데이터셋 이름</span>
        <input
          className="w-full rounded border border-white/15 bg-slate-950 px-3 py-2 text-sm"
          value={outputName}
          disabled={
            exportJob?.status === "queued" || exportJob?.status === "running"
          }
          onChange={(event) => onOutputNameChange(event.target.value)}
        />
      </label>
      <button
        type="button"
        className="workbench-button workbench-button--primary"
        disabled={
          !approvalToken ||
          !outputName.trim() ||
          !OUTPUT_NAME_PATTERN.test(outputName.trim()) ||
          outputName.includes("..") ||
          exportJob?.status === "queued" ||
          exportJob?.status === "running"
        }
        onClick={() => void exportDataset()}
      >
        <LuImagePlus aria-hidden />{" "}
        {exportJob && !TERMINAL.has(exportJob.status)
          ? "생성 중…"
          : "새 데이터셋 만들기"}
      </button>
      {exportJob?.status === "succeeded" && (
        <Link className="workbench-button" href="/library">
          라이브러리 열기
        </Link>
      )}
      <span className="text-xs text-slate-400" aria-live="polite">
        {jobMessage(exportJob)}
      </span>
    </div>
  );
}

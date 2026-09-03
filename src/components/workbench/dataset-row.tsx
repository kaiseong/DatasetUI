"use client";

import Link from "next/link";
import { useState } from "react";
import {
  LuBot,
  LuChevronDown,
  LuCircleAlert,
  LuCopy,
  LuDatabase,
  LuFileWarning,
  LuListChecks,
  LuPlay,
} from "react-icons/lu";
import type { DatasetReadiness, DatasetSummary } from "@/lib/workbench-api";
import { registeredDatasetViewerPath } from "@/utils/versionUtils";

const READINESS: Record<
  DatasetReadiness,
  { label: string; icon: typeof LuDatabase }
> = {
  ready: { label: "사용 가능", icon: LuDatabase },
  incomplete: { label: "파일 확인 필요", icon: LuFileWarning },
  unsupported: { label: "지원하지 않는 버전", icon: LuCircleAlert },
  invalid: { label: "정보를 읽을 수 없음", icon: LuCircleAlert },
};

export default function DatasetRow({ dataset }: { dataset: DatasetSummary }) {
  const [expanded, setExpanded] = useState(false);
  const readiness = READINESS[dataset.readiness];
  const StatusIcon = readiness.icon;

  return (
    <article className={`dataset-row dataset-row--${dataset.readiness}`}>
      <button
        type="button"
        className="dataset-row__summary"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        <span className="dataset-row__identity">
          <span className="dataset-row__name">{dataset.name}</span>
          <span className="dataset-row__path">{dataset.relative_path}</span>
        </span>

        <span className={`source-badge source-badge--${dataset.storage_area}`}>
          {dataset.storage_area === "raw" ? (
            <LuDatabase aria-hidden />
          ) : (
            <LuCopy aria-hidden />
          )}
          {dataset.storage_area === "raw" ? "원본" : "작업본"}
        </span>

        <span className="dataset-row__metric">
          <small>버전</small>
          <strong>{dataset.codebase_version ?? "—"}</strong>
        </span>
        <span className="dataset-row__metric">
          <small>에피소드</small>
          <strong>{formatCount(dataset.total_episodes)}</strong>
        </span>
        <span className="dataset-row__metric dataset-row__metric--frames">
          <small>프레임</small>
          <strong>{formatCount(dataset.total_frames)}</strong>
        </span>

        <span
          className={`readiness-badge readiness-badge--${dataset.readiness}`}
        >
          <StatusIcon aria-hidden />
          {readiness.label}
        </span>
        <LuChevronDown
          className={`dataset-row__chevron ${expanded ? "is-expanded" : ""}`}
          aria-hidden
        />
      </button>

      {expanded && (
        <div className="dataset-row__detail">
          <dl>
            <div>
              <dt>로봇</dt>
              <dd>
                <LuBot aria-hidden /> {dataset.robot_type ?? "정보 없음"}
              </dd>
            </div>
            <div>
              <dt>FPS</dt>
              <dd>{dataset.fps ?? "—"}</dd>
            </div>
            <div>
              <dt>태스크</dt>
              <dd>{formatCount(dataset.total_tasks)}</dd>
            </div>
            <div>
              <dt>마지막 확인</dt>
              <dd>{formatDate(dataset.last_seen_at)}</dd>
            </div>
          </dl>
          {dataset.scan_error && (
            <p className="dataset-row__notice">{dataset.scan_error}</p>
          )}
          {dataset.available && dataset.readiness === "ready" ? (
            <div className="dataset-row__actions">
              <p>원본 파일을 변경하지 않고 Viewer에서 엽니다.</p>
              <div className="flex flex-wrap justify-end gap-2">
                <Link
                  href={`/datasets/${encodeURIComponent(dataset.id)}/curate`}
                  className="workbench-button"
                >
                  <LuListChecks aria-hidden /> Curate
                </Link>
                <Link
                  href={registeredDatasetViewerPath(dataset.id)}
                  className="workbench-button workbench-button--primary"
                >
                  <LuPlay aria-hidden /> Viewer에서 열기
                </Link>
              </div>
            </div>
          ) : (
            <p className="dataset-row__phase-note">
              파일 상태를 확인한 뒤 Viewer에서 열 수 있습니다.
            </p>
          )}
        </div>
      )}
    </article>
  );
}

function formatCount(value: number | null) {
  return value === null ? "—" : value.toLocaleString("ko-KR");
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

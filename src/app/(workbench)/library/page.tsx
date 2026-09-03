"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  LuDatabase,
  LuFolderSearch,
  LuRefreshCw,
  LuSearch,
  LuServerOff,
} from "react-icons/lu";
import DatasetRow from "@/components/workbench/dataset-row";
import EmptyState from "@/components/workbench/empty-state";
import HuggingFaceLibrary from "@/components/workbench/hf-library";
import { useProfile } from "@/components/workbench/profile-context";
import {
  getJob,
  isActiveJob,
  listDatasets,
  publicJobError,
  refreshLibrary,
  type DatasetReadiness,
  type DatasetSummary,
} from "@/lib/workbench-api";

type AreaFilter = "all" | "raw" | "derived";
type ReadinessFilter = "all" | DatasetReadiness;

export default function LibraryPage() {
  const [source, setSource] = useState<"nas" | "hf">("nas");

  return (
    <div className="workbench-page">
      <nav className="library-source-tabs" aria-label="라이브러리 위치">
        <button
          type="button"
          className={source === "nas" ? "is-active" : ""}
          onClick={() => setSource("nas")}
        >
          NAS Library
        </button>
        <button
          type="button"
          className={source === "hf" ? "is-active" : ""}
          onClick={() => setSource("hf")}
        >
          Hugging Face
        </button>
      </nav>
      {source === "nas" ? <NasLibrary /> : <HuggingFaceLibrary />}
    </div>
  );
}

function NasLibrary() {
  const { currentProfile, openProfileDialog } = useProfile();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [area, setArea] = useState<AreaFilter>("all");
  const [readiness, setReadiness] = useState<ReadinessFilter>("all");
  const [refreshJobId, setRefreshJobId] = useState<string | null>(null);
  const [refreshMessage, setRefreshMessage] = useState("");
  const [submittingRefresh, setSubmittingRefresh] = useState(false);
  const refreshSubmissionInFlight = useRef(false);
  const refreshIntentKey = useRef<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setDatasets(await listDatasets());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "라이브러리를 불러오지 못했습니다.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!refreshJobId) return;
    let active = true;
    let timer: number | undefined;
    const check = async () => {
      try {
        const job = await getJob(refreshJobId);
        if (!active) return;
        if (isActiveJob(job.status)) {
          timer = window.setTimeout(check, 1500);
          return;
        }
        setRefreshJobId(null);
        if (job.status === "succeeded") {
          setRefreshMessage("공유 저장소 확인이 끝났습니다.");
          await load();
        } else {
          setRefreshMessage(publicJobError(job.error_code));
        }
      } catch (requestError) {
        if (!active) return;
        setRefreshJobId(null);
        setRefreshMessage(
          requestError instanceof Error
            ? requestError.message
            : "작업 상태를 확인하지 못했습니다.",
        );
      }
    };
    void check();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [refreshJobId, load]);

  async function handleRefresh() {
    if (refreshSubmissionInFlight.current) return;
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    refreshSubmissionInFlight.current = true;
    setSubmittingRefresh(true);
    setRefreshMessage("공유 저장소를 확인하고 있습니다…");
    setError(null);
    refreshIntentKey.current ??= `library-refresh-${crypto.randomUUID()}`;
    try {
      const job = await refreshLibrary(
        currentProfile.id,
        refreshIntentKey.current,
      );
      refreshIntentKey.current = null;
      setRefreshJobId(job.id);
    } catch (requestError) {
      setRefreshMessage("");
      setError(
        requestError instanceof Error
          ? requestError.message
          : "새로 확인하지 못했습니다.",
      );
    } finally {
      refreshSubmissionInFlight.current = false;
      setSubmittingRefresh(false);
    }
  }

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase("ko-KR");
    return datasets.filter((dataset) => {
      if (area !== "all" && dataset.storage_area !== area) return false;
      if (readiness !== "all" && dataset.readiness !== readiness) return false;
      if (
        normalizedQuery &&
        !`${dataset.name} ${dataset.relative_path} ${dataset.robot_type ?? ""}`
          .toLocaleLowerCase("ko-KR")
          .includes(normalizedQuery)
      ) {
        return false;
      }
      return true;
    });
  }, [datasets, query, area, readiness]);

  const readyCount = datasets.filter(
    (dataset) => dataset.readiness === "ready",
  ).length;
  const originalCount = datasets.filter(
    (dataset) => dataset.storage_area === "raw",
  ).length;

  return (
    <>
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">NAS LIBRARY</p>
          <h1>공유 데이터셋</h1>
          <p>원본은 그대로 보존하고, 팀이 함께 쓰는 데이터셋을 확인합니다.</p>
        </div>
        <button
          type="button"
          className="workbench-button workbench-button--primary"
          onClick={handleRefresh}
          disabled={submittingRefresh || Boolean(refreshJobId)}
        >
          <LuRefreshCw
            className={refreshJobId || submittingRefresh ? "animate-spin" : ""}
            aria-hidden
          />
          {refreshJobId || submittingRefresh
            ? "확인 중…"
            : "라이브러리 새로 확인"}
        </button>
      </section>

      <div className="library-summary" aria-label="라이브러리 요약">
        <div>
          <small>전체</small>
          <strong>{datasets.length.toLocaleString("ko-KR")}</strong>
          <span>datasets</span>
        </div>
        <div>
          <small>바로 사용 가능</small>
          <strong>{readyCount.toLocaleString("ko-KR")}</strong>
          <span>ready</span>
        </div>
        <div>
          <small>보존된 원본</small>
          <strong>{originalCount.toLocaleString("ko-KR")}</strong>
          <span>originals</span>
        </div>
        <div className="library-summary__note">
          <LuDatabase aria-hidden />
          <span>원본 영역은 읽기 전용으로 연결됩니다.</span>
        </div>
      </div>

      <section className="library-toolbar" aria-label="데이터셋 필터">
        <label className="library-search">
          <LuSearch aria-hidden />
          <span className="sr-only">데이터셋 검색</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="이름, 경로, 로봇으로 검색"
          />
        </label>
        <FilterGroup
          label="위치"
          value={area}
          onChange={(value) => setArea(value as AreaFilter)}
          options={[
            ["all", "전체"],
            ["raw", "원본"],
            ["derived", "작업본"],
          ]}
        />
        <FilterGroup
          label="상태"
          value={readiness}
          onChange={(value) => setReadiness(value as ReadinessFilter)}
          options={[
            ["all", "모든 상태"],
            ["ready", "사용 가능"],
            ["incomplete", "확인 필요"],
            ["unsupported", "미지원"],
            ["invalid", "읽기 실패"],
          ]}
        />
      </section>

      <p className="workbench-live-message" aria-live="polite">
        {refreshMessage}
      </p>

      {loading ? (
        <div className="library-skeleton" aria-label="라이브러리 불러오는 중">
          {[0, 1, 2].map((item) => (
            <div key={item} />
          ))}
        </div>
      ) : error ? (
        <EmptyState
          icon={<LuServerOff />}
          title="라이브러리에 연결할 수 없습니다"
          description={error}
          action={
            <button type="button" className="workbench-button" onClick={load}>
              다시 시도
            </button>
          }
        />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<LuFolderSearch />}
          title={
            datasets.length === 0
              ? "아직 데이터셋이 없습니다"
              : "검색 결과가 없습니다"
          }
          description={
            datasets.length === 0
              ? "라이브러리를 새로 확인하면 공유 저장소의 데이터셋을 찾아옵니다."
              : "검색어나 필터를 바꿔보세요."
          }
          action={
            datasets.length === 0 ? (
              <button
                type="button"
                className="workbench-button workbench-button--primary"
                onClick={handleRefresh}
                disabled={Boolean(refreshJobId)}
              >
                <LuRefreshCw aria-hidden /> 새로 확인
              </button>
            ) : undefined
          }
        />
      ) : (
        <section className="dataset-list" aria-label="데이터셋 목록">
          <div className="dataset-list__head" aria-hidden>
            <span>데이터셋</span>
            <span>위치</span>
            <span>버전</span>
            <span>에피소드</span>
            <span>프레임</span>
            <span>상태</span>
            <span />
          </div>
          {filtered.map((dataset) => (
            <DatasetRow key={dataset.id} dataset={dataset} />
          ))}
        </section>
      )}
    </>
  );
}

function FilterGroup({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: [string, string][];
}) {
  return (
    <label className="library-select">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>
            {optionLabel}
          </option>
        ))}
      </select>
    </label>
  );
}

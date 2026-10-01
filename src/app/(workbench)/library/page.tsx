"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  LuDatabase,
  LuFolderSearch,
  LuRefreshCw,
  LuSearch,
  LuServerOff,
} from "react-icons/lu";
import DatasetRow from "@/components/workbench/dataset-row";
import { DatasetTrashPanel } from "@/components/workbench/dataset-trash-panel";
import EmptyState from "@/components/workbench/empty-state";
import HuggingFaceLibrary from "@/components/workbench/hf-library";
import { useProfile } from "@/components/workbench/profile-context";
import {
  getJob,
  isActiveJob,
  listDatasetTrash,
  listDatasets,
  publicJobError,
  refreshLibrary,
  type DatasetReadiness,
  type DatasetSummary,
  type DatasetSort,
  type DatasetTrashEntry,
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
  const [trashEntries, setTrashEntries] = useState<DatasetTrashEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [trashError, setTrashError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<DatasetSort>("newest");
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [area, setArea] = useState<AreaFilter>("all");
  const [readiness, setReadiness] = useState<ReadinessFilter>("all");
  const [refreshJobId, setRefreshJobId] = useState<string | null>(null);
  const [refreshMessage, setRefreshMessage] = useState("");
  const [submittingRefresh, setSubmittingRefresh] = useState(false);
  const refreshSubmissionInFlight = useRef(false);
  const refreshIntentKey = useRef<string | null>(null);
  const loadGeneration = useRef(0);
  const loadController = useRef<AbortController | null>(null);
  const moreInFlight = useRef(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("datasetui.library.sort.v1");
      if (["newest", "oldest", "name_asc", "name_desc"].includes(saved ?? ""))
        setSort(saved as DatasetSort);
    } catch {
      /* Storage can be disabled by browser policy. */
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => setSearch(query.trim()), 200);
    return () => window.clearTimeout(timer);
  }, [query]);

  const load = useCallback(async () => {
    const generation = ++loadGeneration.current;
    loadController.current?.abort();
    const controller = new AbortController();
    loadController.current = controller;
    setError(null);
    setTrashError(null);
    const profileId = currentProfile?.id;
    const [datasetResult, trashResult] = await Promise.allSettled([
      listDatasets({
        sort,
        query: search,
        storageArea: area === "all" ? undefined : area,
        readiness: readiness === "all" ? undefined : readiness,
        limit: 100,
        signal: controller.signal,
      }),
      profileId ? listDatasetTrash(profileId) : Promise.resolve([]),
    ]);
    if (controller.signal.aborted || generation !== loadGeneration.current)
      return;
    if (datasetResult.status === "fulfilled") {
      setDatasets(datasetResult.value);
      setHasMore(datasetResult.value.length === 100);
    } else {
      const requestError = datasetResult.reason as unknown;
      setError(
        requestError instanceof Error
          ? requestError.message
          : "라이브러리를 불러오지 못했습니다.",
      );
    }
    if (trashResult.status === "fulfilled") {
      setTrashEntries(trashResult.value);
    } else {
      setTrashEntries([]);
      setTrashError("휴지통을 불러오지 못했습니다. 잠시 후 다시 확인하세요.");
    }
    setLoading(false);
  }, [currentProfile?.id, sort, search, area, readiness]);

  useEffect(() => {
    void load();
    return () => loadController.current?.abort();
  }, [load]);

  async function loadMore() {
    if (moreInFlight.current || !hasMore) return;
    moreInFlight.current = true;
    setLoadingMore(true);
    const generation = loadGeneration.current;
    try {
      const items = await listDatasets({
        sort,
        query: search,
        storageArea: area === "all" ? undefined : area,
        readiness: readiness === "all" ? undefined : readiness,
        limit: 100,
        offset: datasets.length,
      });
      if (generation !== loadGeneration.current) return;
      setDatasets((current) => [
        ...new Map(
          [...current, ...items].map((item) => [item.id, item]),
        ).values(),
      ]);
      setHasMore(items.length === 100);
    } catch (requestError) {
      if (generation === loadGeneration.current)
        setError(
          requestError instanceof Error
            ? requestError.message
            : "목록을 더 불러오지 못했습니다.",
        );
    } finally {
      moreInFlight.current = false;
      setLoadingMore(false);
    }
  }

  function changeSort(value: string) {
    setSort(value as DatasetSort);
    try {
      localStorage.setItem("datasetui.library.sort.v1", value);
    } catch {
      /* Optional preference. */
    }
  }

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
          disabled={submittingRefresh}
        >
          <LuRefreshCw
            className={refreshJobId || submittingRefresh ? "animate-spin" : ""}
            aria-hidden
          />
          {submittingRefresh ? "접수 중…" : "라이브러리 새로 확인"}
        </button>
      </section>

      <div className="library-summary" aria-label="라이브러리 요약">
        <div>
          <small>현재 불러온 목록</small>
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
          <span>원본 내용은 보존하며, 삭제 시 휴지통으로 이동합니다.</span>
        </div>
      </div>

      <section className="library-toolbar" aria-label="데이터셋 필터">
        <label className="library-search">
          <LuSearch aria-hidden />
          <span className="sr-only">데이터셋 검색</span>
          <input
            value={query}
            maxLength={160}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="이름, 경로, 로봇으로 검색"
          />
        </label>
        <FilterGroup
          label="정렬"
          value={sort}
          onChange={changeSort}
          options={[
            ["newest", "최근 등록순"],
            ["oldest", "오래된 등록순"],
            ["name_asc", "이름 오름차순"],
            ["name_desc", "이름 내림차순"],
          ]}
        />
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

      <DatasetTrashPanel
        entries={trashEntries}
        currentProfileId={currentProfile?.id}
        error={trashError}
        onRefresh={load}
        onRestored={(_restored, trashId) => {
          setTrashEntries((current) =>
            current.filter((entry) => entry.dataset.id !== trashId),
          );
          void load();
        }}
      />

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
      ) : datasets.length === 0 ? (
        <EmptyState
          icon={<LuFolderSearch />}
          title={
            search || area !== "all" || readiness !== "all"
              ? "검색 결과가 없습니다"
              : "아직 데이터셋이 없습니다"
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
                disabled={submittingRefresh}
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
          {datasets.map((dataset) => (
            <DatasetRow
              key={dataset.id}
              dataset={dataset}
              currentProfileId={currentProfile?.id}
              onRequireProfile={openProfileDialog}
              onRenamed={() => void load()}
              onDeleted={(entry) => {
                setDatasets((current) =>
                  current.filter((item) => item.id !== dataset.id),
                );
                setTrashEntries((current) => [
                  entry,
                  ...current.filter(
                    (item) => item.dataset.id !== entry.dataset.id,
                  ),
                ]);
              }}
              onRefresh={load}
            />
          ))}
        </section>
      )}
      {hasMore && !error && (
        <button
          type="button"
          className="workbench-button"
          disabled={loadingMore}
          onClick={() => void loadMore()}
        >
          {loadingMore ? "불러오는 중…" : "데이터셋 더 보기"}
        </button>
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
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>
            {optionLabel}
          </option>
        ))}
      </select>
    </label>
  );
}

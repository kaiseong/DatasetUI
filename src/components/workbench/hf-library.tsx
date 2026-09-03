"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  LuCheck,
  LuCloudDownload,
  LuGitBranch,
  LuLoaderCircle,
  LuRefreshCw,
  LuSearch,
  LuServerOff,
  LuShieldCheck,
  LuX,
} from "react-icons/lu";
import EmptyState from "@/components/workbench/empty-state";
import { useProfile } from "@/components/workbench/profile-context";
import {
  getJob,
  importHuggingFaceDataset,
  isActiveJob,
  listHuggingFaceDatasets,
  listHuggingFaceRevisions,
  publicJobError,
  type HuggingFaceDataset,
  type HuggingFaceDatasetStatus,
  type HuggingFaceRevision,
} from "@/lib/workbench-api";

const STATUS: Record<
  HuggingFaceDatasetStatus,
  { label: string; action: string }
> = {
  not_downloaded: { label: "가져오지 않음", action: "가져오기" },
  queued: { label: "대기 중", action: "대기 중" },
  downloading: { label: "가져오는 중", action: "가져오는 중" },
  ready: { label: "사용 가능", action: "revision 보기" },
  update_available: { label: "업데이트 있음", action: "새 revision" },
  incomplete: { label: "확인 필요", action: "다시 가져오기" },
  validation_failed: { label: "검증 실패", action: "다시 시도" },
};

export default function HuggingFaceLibrary() {
  const { currentProfile, openProfileDialog } = useProfile();
  const [datasets, setDatasets] = useState<HuggingFaceDataset[]>([]);
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<HuggingFaceDataset | null>(null);
  const [revisions, setRevisions] = useState<HuggingFaceRevision[]>([]);
  const [revision, setRevision] = useState<HuggingFaceRevision | null>(null);
  const [loadingRevisions, setLoadingRevisions] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState("");
  const dialogRef = useRef<HTMLDialogElement>(null);
  const importIntent = useRef<string | null>(null);

  const load = useCallback(
    async (search = submittedQuery) => {
      setError(null);
      try {
        setDatasets(await listHuggingFaceDatasets(search));
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Hugging Face 목록을 불러오지 못했습니다.",
        );
      } finally {
        setLoading(false);
      }
    },
    [submittedQuery],
  );

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (
      !datasets.some((dataset) =>
        ["queued", "downloading"].includes(dataset.status),
      )
    ) {
      return;
    }
    const timer = window.setInterval(() => void load(), 2500);
    return () => window.clearInterval(timer);
  }, [datasets, load]);

  async function openImport(dataset: HuggingFaceDataset) {
    setSelected(dataset);
    setRevision(null);
    setRevisions([]);
    setMessage("");
    setLoadingRevisions(true);
    dialogRef.current?.showModal();
    try {
      const loaded = await listHuggingFaceRevisions(dataset.name);
      setRevisions(loaded);
      setRevision(loaded[0] ?? null);
    } catch (requestError) {
      setMessage(
        requestError instanceof Error
          ? requestError.message
          : "revision 목록을 불러오지 못했습니다.",
      );
    } finally {
      setLoadingRevisions(false);
    }
  }

  async function handleImport() {
    if (!selected || !revision || submitting) return;
    if (!currentProfile) {
      dialogRef.current?.close();
      openProfileDialog();
      return;
    }
    setSubmitting(true);
    setMessage("원본 revision을 안전하게 가져오고 있습니다…");
    importIntent.current ??= `hf-import-${selected.name}-${revision.commit_sha}-${crypto.randomUUID()}`;
    try {
      const job = await importHuggingFaceDataset(
        currentProfile.id,
        selected.name,
        revision,
        importIntent.current,
      );
      importIntent.current = null;
      dialogRef.current?.close();
      setMessage("");
      setDatasets((current) =>
        current.map((item) =>
          item.name === selected.name ? { ...item, status: "queued" } : item,
        ),
      );
      await waitForImport(job.id);
    } catch (requestError) {
      const safeMessage =
        requestError instanceof Error
          ? requestError.message
          : "가져오기 작업을 시작하지 못했습니다.";
      if (dialogRef.current?.open) setMessage(safeMessage);
      else setError(safeMessage);
    } finally {
      setSubmitting(false);
    }
  }

  async function waitForImport(jobId: string) {
    for (;;) {
      const job = await getJob(jobId);
      if (!isActiveJob(job.status)) {
        if (job.status !== "succeeded")
          setError(publicJobError(job.error_code));
        await load();
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
  }

  function handleSearch(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setSubmittedQuery(query.trim());
  }

  return (
    <>
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">HUGGING FACE · RAINBOWROBOTICS</p>
          <h1>팀 데이터 허브</h1>
          <p>
            선택한 revision을 고정해 공유 저장소의 보존 원본으로 가져옵니다.
          </p>
        </div>
        <button
          type="button"
          className="workbench-button"
          onClick={() => void load()}
        >
          <LuRefreshCw aria-hidden /> 목록 새로 확인
        </button>
      </section>

      <div className="hf-trust-note">
        <LuShieldCheck aria-hidden />
        <span>
          rainbowrobotics 조직만 표시됩니다. 접근 토큰은 서버 안에서만
          사용합니다.
        </span>
      </div>

      <form className="library-toolbar" onSubmit={handleSearch} role="search">
        <label className="library-search">
          <LuSearch aria-hidden />
          <span className="sr-only">Hugging Face 데이터셋 검색</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="데이터셋 이름으로 검색"
            maxLength={100}
          />
        </label>
        <button type="submit" className="workbench-button">
          검색
        </button>
      </form>

      {loading ? (
        <div
          className="library-skeleton"
          aria-label="Hugging Face 목록 불러오는 중"
        >
          {[0, 1, 2].map((item) => (
            <div key={item} />
          ))}
        </div>
      ) : error ? (
        <EmptyState
          icon={<LuServerOff />}
          title="Hugging Face에 연결할 수 없습니다"
          description={error}
          action={
            <button
              type="button"
              className="workbench-button"
              onClick={() => void load()}
            >
              다시 시도
            </button>
          }
        />
      ) : datasets.length === 0 ? (
        <EmptyState
          icon={<LuSearch />}
          title="데이터셋을 찾지 못했습니다"
          description="검색어를 바꾸거나 조직에 공개된 데이터셋을 확인해 주세요."
        />
      ) : (
        <section
          className="hf-dataset-grid"
          aria-label="Hugging Face 데이터셋 목록"
        >
          {datasets.map((dataset) => {
            const status = STATUS[dataset.status];
            const active =
              dataset.status === "queued" || dataset.status === "downloading";
            return (
              <article className="hf-dataset-card" key={dataset.repo_id}>
                <div className="hf-dataset-card__top">
                  <span className={`hf-status hf-status--${dataset.status}`}>
                    {status.label}
                  </span>
                  <code>{dataset.latest_commit_sha.slice(0, 8)}</code>
                </div>
                <h2>{dataset.name}</h2>
                <p>
                  {dataset.private ? "비공개 팀 데이터셋" : "공개 팀 데이터셋"}
                </p>
                <div className="hf-dataset-card__meta">
                  <span>
                    {dataset.downloads.toLocaleString("ko-KR")} downloads
                  </span>
                  <span>{formatDate(dataset.last_modified)}</span>
                </div>
                <button
                  type="button"
                  className="workbench-button"
                  disabled={active}
                  onClick={() => void openImport(dataset)}
                >
                  {active ? (
                    <LuLoaderCircle className="animate-spin" aria-hidden />
                  ) : dataset.status === "ready" ? (
                    <LuCheck aria-hidden />
                  ) : (
                    <LuCloudDownload aria-hidden />
                  )}
                  {status.action}
                </button>
              </article>
            );
          })}
        </section>
      )}

      <dialog
        ref={dialogRef}
        className="profile-dialog hf-import-dialog"
        onClose={() => {
          setSelected(null);
          setMessage("");
        }}
        aria-labelledby="hf-import-title"
      >
        <div className="profile-dialog__topline" />
        <div className="flex items-start justify-between gap-6">
          <div>
            <p className="workbench-eyebrow">IMMUTABLE IMPORT</p>
            <h2 id="hf-import-title" className="mt-2 text-2xl font-semibold">
              revision 선택
            </h2>
            <p className="mt-2 text-sm text-[var(--text-muted)]">
              {selected?.name}의 선택된 commit을 원본으로 보존합니다.
            </p>
          </div>
          <button
            type="button"
            className="icon-button"
            onClick={() => dialogRef.current?.close()}
            aria-label="닫기"
          >
            <LuX aria-hidden />
          </button>
        </div>

        <label className="hf-revision-field">
          <span>
            <LuGitBranch aria-hidden /> revision
          </span>
          <select
            value={revision ? `${revision.kind}:${revision.name}` : ""}
            disabled={loadingRevisions || revisions.length === 0}
            onChange={(event) => {
              const next = revisions.find(
                (item) => `${item.kind}:${item.name}` === event.target.value,
              );
              setRevision(next ?? null);
              importIntent.current = null;
            }}
          >
            {revisions.map((item) => (
              <option
                key={`${item.kind}:${item.name}`}
                value={`${item.kind}:${item.name}`}
              >
                {item.name} · {item.commit_sha.slice(0, 12)}
              </option>
            ))}
          </select>
        </label>

        <div className="hf-import-dialog__footer">
          <p aria-live="polite">
            {loadingRevisions ? "revision 확인 중…" : message}
          </p>
          <button
            type="button"
            className="workbench-button workbench-button--primary"
            disabled={!revision || loadingRevisions || submitting}
            onClick={() => void handleImport()}
          >
            {submitting ? (
              <LuLoaderCircle className="animate-spin" aria-hidden />
            ) : (
              <LuCloudDownload aria-hidden />
            )}
            {submitting ? "작업 접수 중…" : "선택한 revision 가져오기"}
          </button>
        </div>
      </dialog>
    </>
  );
}

function formatDate(value: string | null) {
  if (!value) return "업데이트 시각 없음";
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(
    new Date(value),
  );
}

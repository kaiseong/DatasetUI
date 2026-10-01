"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { LuSparkles } from "react-icons/lu";
import SegmentationWorkflow from "@/components/workbench/segmentation-workflow";
import { useProfile } from "@/components/workbench/profile-context";
import { listDatasets, type DatasetSummary } from "@/lib/workbench-api";
import {
  getSegmentationCapabilities,
  getSegmentationCatalog,
  type SegmentationCapabilities,
  type SegmentationScope,
} from "@/lib/segmentation-api";

export default function AugmentPage() {
  return (
    <Suspense
      fallback={
        <div className="workbench-page">
          <p className="workbench-live-message">증강 작업대를 준비하는 중…</p>
        </div>
      }
    >
      <AugmentContent />
    </Suspense>
  );
}

function AugmentContent() {
  const searchParams = useSearchParams();
  const requestedDataset = searchParams.get("dataset") ?? "";
  const { currentProfile } = useProfile();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [confirmedId, setConfirmedId] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [loading, setLoading] = useState(false);
  const [capabilities, setCapabilities] =
    useState<SegmentationCapabilities | null>(null);
  const [scope, setScope] = useState<SegmentationScope | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const requestGenerationRef = useRef(0);

  useEffect(() => {
    let active = true;
    void Promise.all([listDatasets(), getSegmentationCapabilities()])
      .then(([items, nextCapabilities]) => {
        if (!active) return;
        const ready = items.filter(
          (item) => item.available && item.readiness === "ready",
        );
        setDatasets(ready);
        setCapabilities(nextCapabilities);
        setDatasetId(
          ready.some((item) => item.id === requestedDataset)
            ? requestedDataset
            : (ready[0]?.id ?? ""),
        );
      })
      .catch((error: unknown) => {
        if (active) {
          setMessage(
            error instanceof Error
              ? error.message
              : "증강 기능 정보를 불러오지 못했습니다.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [requestedDataset]);

  useEffect(() => {
    const generation = ++requestGenerationRef.current;
    setScope(null);
    setMessage(null);
    if (!confirmedId) return;
    const controller = new AbortController();
    setLoading(true);
    void getSegmentationCatalog(confirmedId, controller.signal)
      .then((nextScope) => {
        if (requestGenerationRef.current === generation)
          setScope({
            fingerprint: "",
            frame_token: "",
            metadata_revision: nextScope.metadata_revision,
            video_keys: nextScope.video_keys,
            episodes: Array.from(
              { length: nextScope.total_episodes },
              (_, episode_index) => ({ episode_index, length: 0 }),
            ),
          });
      })
      .catch((error: unknown) => {
        if (requestGenerationRef.current === generation) {
          setMessage(
            error instanceof Error
              ? error.message
              : "데이터셋 범위를 불러오지 못했습니다.",
          );
        }
      })
      .finally(() => {
        if (requestGenerationRef.current === generation) setLoading(false);
      });
    return () => {
      controller.abort();
      requestGenerationRef.current += 1;
    };
  }, [confirmedId, attempt]);

  const selected = datasets.find((item) => item.id === confirmedId);

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">AUGMENT</p>
          <h1>작업 영역 Segmentation</h1>
          <p>
            SAM3.1로 보드·그리퍼·플러그 등 필요한 픽셀만 남깁니다. 검은 배경이
            기본이며, 사진 배경은 선택 사항입니다.
          </p>
        </div>
        <LuSparkles className="h-8 w-8 text-cyan-300" aria-hidden />
      </section>

      <section className="mb-5 rounded-lg border border-cyan-400/20 bg-cyan-400/[0.06] p-4 text-sm text-slate-200">
        <strong className="text-cyan-200">원본 보존</strong>
        <p className="mt-1 text-slate-300">
          자르기·확대·리사이즈 없이 원래 위치와 해상도를 유지합니다. 선택한 모든
          영상의 검토·승인 후 하나의 전체 데이터셋으로 생성하며, 원본은 덮어쓰지
          않습니다.
        </p>
      </section>

      <label className="mb-5 block max-w-xl text-xs text-slate-300">
        <span className="mb-1 block font-medium">원본 데이터셋</span>
        <select
          className="w-full rounded border border-white/15 bg-slate-950 px-3 py-2 text-sm"
          value={datasetId}
          onChange={(event) => setDatasetId(event.target.value)}
        >
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name} · {dataset.total_episodes ?? 0} episodes
            </option>
          ))}
        </select>
      </label>
      <button
        className="workbench-button workbench-button--primary mb-4"
        disabled={!datasetId}
        onClick={() => {
          setConfirmedId(datasetId);
          setAttempt((value) => value + 1);
        }}
      >
        확인
      </button>
      {loading && (
        <p role="status">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-cyan-300 border-t-transparent" />{" "}
          데이터셋 메타정보를 준비하는 중…
        </p>
      )}

      {!currentProfile && (
        <p className="workbench-live-message">
          미리보기와 Export는 현재 브라우저에서 선택한 사용자에게 귀속됩니다.
        </p>
      )}
      {message && (
        <p className="workbench-live-message" role="alert">
          {message}
        </p>
      )}
      {selected &&
      scope &&
      capabilities &&
      scope.episodes.length > 0 &&
      scope.video_keys.length > 0 ? (
        <SegmentationWorkflow
          key={`${currentProfile?.id ?? "no-profile"}:${selected.id}:${scope.fingerprint}`}
          datasetId={selected.id}
          datasetName={selected.name}
          scope={scope}
          capabilities={capabilities}
        />
      ) : scope &&
        (scope.episodes.length === 0 || scope.video_keys.length === 0) ? (
        <p className="workbench-live-message" role="alert">
          처리할 video episode 또는 camera key가 없습니다.
        </p>
      ) : confirmedId && !message && loading ? (
        <p className="workbench-live-message" aria-live="polite">
          원본 frame 범위를 불러오는 중…
        </p>
      ) : null}
    </div>
  );
}

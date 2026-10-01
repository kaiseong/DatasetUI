"use client";

import { Suspense, useEffect, useRef, useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { LuRefreshCw, LuShieldCheck } from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import { ValidationRunCard } from "@/components/workbench/validation-run-card";
import {
  convertDatasetToV21,
  deleteDatasetValidation,
  listDatasets,
  listDatasetValidations,
  validateDataset,
  type DatasetSummary,
  type ValidationMode,
  type ValidationRun,
} from "@/lib/workbench-api";

export default function ValidatePage() {
  return (
    <Suspense
      fallback={<div className="workbench-page">검증 기록을 준비하는 중…</div>}
    >
      <ValidateContent />
    </Suspense>
  );
}

function ValidateContent() {
  const searchParams = useSearchParams();
  const requestedDataset = searchParams.get("dataset") ?? "";
  const requestedJob = searchParams.get("job") ?? "";
  const focusedJob = useRef("");
  const { currentProfile, openProfileDialog } = useProfile();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [mode, setMode] = useState<ValidationMode>("quick");
  const [runs, setRuns] = useState<ValidationRun[]>([]);
  const [outputName, setOutputName] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [pollError, setPollError] = useState(false);
  const deletedIds = useRef(new Set<string>());
  const selected = datasets.find((item) => item.id === datasetId);
  const visibleRuns = runs.filter((run) => run.dataset_id === datasetId);

  useEffect(() => {
    let active = true;
    void listDatasets()
      .then((items) => {
        if (!active) return;
        const ready = items.filter(
          (item) => item.available && item.readiness === "ready",
        );
        setDatasets(ready);
        setDatasetId((current) =>
          ready.some((item) => item.id === requestedDataset)
            ? requestedDataset
            : current || ready[0]?.id || "",
        );
      })
      .catch((error: unknown) => {
        if (active)
          setMessage(
            error instanceof Error
              ? error.message
              : "목록을 불러오지 못했습니다.",
          );
      });
    return () => {
      active = false;
    };
  }, [requestedDataset]);

  useEffect(() => {
    if (
      !requestedJob ||
      focusedJob.current === requestedJob ||
      !runs.some((run) => run.job_id === requestedJob)
    )
      return;
    const element = document.getElementById(`validation-${requestedJob}`);
    if (element) {
      element.scrollIntoView({ block: "center" });
      focusedJob.current = requestedJob;
    }
  }, [requestedJob, runs]);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    setPollError(false);
    if (!datasetId) return;
    async function poll() {
      try {
        const loaded = await listDatasetValidations(datasetId);
        if (active) {
          setRuns(loaded.filter((run) => !deletedIds.current.has(run.job_id)));
          setPollError(false);
        }
      } catch (error) {
        if (active) {
          setPollError(true);
          setMessage(
            error instanceof Error
              ? error.message
              : "검증 기록을 불러오지 못했습니다.",
          );
        }
      } finally {
        if (active) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [datasetId, refresh]);

  async function runValidation(event: FormEvent) {
    event.preventDefault();
    if (submitting) return;
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    setSubmitting(true);
    try {
      const job = await validateDataset(
        datasetId,
        currentProfile.id,
        mode,
        crypto.randomUUID(),
      );
      setMessage(
        `검사 요청이 접수됐습니다. 아래에서 진행 상태를 확인하세요. (${job.id.slice(0, 8)})`,
      );
      setRuns((current) => [
        {
          job_id: job.id,
          dataset_id: datasetId,
          dataset_fingerprint: selected?.fingerprint ?? "",
          mode,
          created_at: job.created_at,
          status: job.status,
          result: null,
          error_code: job.error_code,
          error_message: null,
          started_at: job.started_at,
          progress: null,
          finished_at: null,
        },
        ...current.filter((run) => run.job_id !== job.id),
      ]);
      setRefresh((value) => value + 1);
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "검증을 시작하지 못했습니다.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  async function deleteRun(run: ValidationRun) {
    await deleteDatasetValidation(run.dataset_id, run.job_id);
    deletedIds.current.add(run.job_id);
    setRuns((current) => current.filter((item) => item.job_id !== run.job_id));
    setMessage("검사 기록을 삭제했습니다. 데이터셋 원본은 그대로입니다.");
  }

  async function convert() {
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    setSubmitting(true);
    try {
      const job = await convertDatasetToV21(
        datasetId,
        currentProfile.id,
        outputName.trim(),
        crypto.randomUUID(),
      );
      setMessage(`v2.1 변환을 시작했습니다. (${job.id.slice(0, 8)})`);
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "변환을 시작하지 못했습니다.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">VALIDATE</p>
          <h1>데이터셋 검증</h1>
          <p>문제를 보고 내보내기 전에 안전 기준을 확인합니다.</p>
        </div>
      </section>
      <form className="curation-builder" onSubmit={runValidation}>
        <div className="curation-trim-fields">
          <label>
            <span>데이터셋</span>
            <select
              value={datasetId}
              disabled={submitting}
              onChange={(event) => setDatasetId(event.target.value)}
            >
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} · {dataset.codebase_version}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>검사 수준</span>
            <select
              value={mode}
              onChange={(event) =>
                setMode(event.target.value as ValidationMode)
              }
            >
              <option value="quick">빠른 검사</option>
              <option value="full">전체 검사</option>
              <option value="export_gate">
                내보내기 검사 · 전체 검사 + 파일 무결성
              </option>
            </select>
          </label>
        </div>
        <p className="workbench-live-message">
          빠른 검사는 처음·마지막 에피소드의 구조를 확인합니다. 전체 검사는 모든
          에피소드·영상과 통계를 비교합니다. 내보내기 검사는 파일 내용까지
          고정해 확인하며, 외부 학습 프로그램 실행을 보증하는 검사는 아닙니다.
        </p>
        <button
          className="workbench-button workbench-button--primary"
          disabled={!datasetId || submitting}
        >
          <LuShieldCheck aria-hidden />{" "}
          {submitting ? "요청 접수 중…" : "검사 시작"}
        </button>
        {submitting && (
          <div className="validation-progress" data-active="true">
            <div
              className="validation-progress__track"
              role="progressbar"
              aria-label="요청 접수 중"
            >
              <span />
            </div>
          </div>
        )}
        {selected?.codebase_version === "v3.0" && (
          <div className="curation-trim-panel">
            <label className="curation-trim-wide">
              <span>v2.1 출력 이름</span>
              <input
                value={outputName}
                onChange={(event) => setOutputName(event.target.value)}
                placeholder={`${selected.name}-v21`}
              />
            </label>
            <button
              type="button"
              className="workbench-button"
              onClick={() => void convert()}
              disabled={submitting || !outputName.trim()}
            >
              <LuRefreshCw aria-hidden /> v2.1 작업본 만들기
            </button>
          </div>
        )}
        <p className="workbench-live-message" aria-live="polite">
          {message}
        </p>
      </form>
      <section className="curation-recipe-list" aria-label="검증 기록">
        {visibleRuns.map((run) => (
          <div key={run.job_id} id={`validation-${run.job_id}`}>
            <ValidationRunCard
              run={run}
              stale={pollError}
              onDelete={deleteRun}
            />
          </div>
        ))}
      </section>
    </div>
  );
}

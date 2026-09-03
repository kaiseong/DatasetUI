"use client";

import { useEffect, useState, type FormEvent } from "react";
import { LuBadgeCheck, LuRefreshCw, LuShieldCheck } from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import {
  convertDatasetToV21,
  listDatasets,
  listDatasetValidations,
  validateDataset,
  type DatasetSummary,
  type ValidationMode,
  type ValidationRun,
} from "@/lib/workbench-api";

export default function ValidatePage() {
  const { currentProfile, openProfileDialog } = useProfile();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [mode, setMode] = useState<ValidationMode>("quick");
  const [runs, setRuns] = useState<ValidationRun[]>([]);
  const [outputName, setOutputName] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const selected = datasets.find((item) => item.id === datasetId);

  useEffect(() => {
    void listDatasets().then((items) => {
      const ready = items.filter((item) => item.available && item.readiness === "ready");
      setDatasets(ready);
      setDatasetId((current) => current || ready[0]?.id || "");
    });
  }, []);

  useEffect(() => {
    if (!datasetId) return;
    void listDatasetValidations(datasetId).then(setRuns);
  }, [datasetId]);

  async function runValidation(event: FormEvent) {
    event.preventDefault();
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    setSubmitting(true);
    try {
      const job = await validateDataset(datasetId, currentProfile.id, mode, crypto.randomUUID());
      setMessage(`검증을 시작했습니다. (${job.id.slice(0, 8)})`);
      setRuns(await listDatasetValidations(datasetId));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "검증을 시작하지 못했습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  async function convert() {
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    setSubmitting(true);
    try {
      const job = await convertDatasetToV21(datasetId, currentProfile.id, outputName.trim(), crypto.randomUUID());
      setMessage(`v2.1 변환을 시작했습니다. (${job.id.slice(0, 8)})`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "변환을 시작하지 못했습니다.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading"><div><p className="workbench-eyebrow">VALIDATE</p><h1>데이터셋 검증</h1><p>문제를 보고 내보내기 전에 안전 기준을 확인합니다.</p></div></section>
      <form className="curation-builder" onSubmit={runValidation}>
        <div className="curation-trim-fields">
          <label><span>데이터셋</span><select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.codebase_version}</option>)}</select></label>
          <label><span>검사 수준</span><select value={mode} onChange={(event) => setMode(event.target.value as ValidationMode)}><option value="quick">빠른 검사</option><option value="full">전체 검사</option><option value="export_gate">내보내기 검사</option></select></label>
        </div>
        <button className="workbench-button workbench-button--primary" disabled={!datasetId || submitting}><LuShieldCheck aria-hidden /> {submitting ? "시작 중…" : "검사 시작"}</button>
        {selected?.codebase_version === "v3.0" && <div className="curation-trim-panel"><label className="curation-trim-wide"><span>v2.1 출력 이름</span><input value={outputName} onChange={(event) => setOutputName(event.target.value)} placeholder={`${selected.name}-v21`} /></label><button type="button" className="workbench-button" onClick={() => void convert()} disabled={submitting || !outputName.trim()}><LuRefreshCw aria-hidden /> v2.1 작업본 만들기</button></div>}
        <p className="workbench-live-message" aria-live="polite">{message}</p>
      </form>
      <section className="curation-recipe-list" aria-label="검증 기록">
        {runs.map((run) => <article key={run.job_id}><div><span>{run.mode}</span><time>{new Date(run.created_at).toLocaleString("ko-KR")}</time></div><h3>{run.status === "succeeded" ? (run.result?.passed ? "통과" : "문제 발견") : run.status}</h3><p>실패 {run.result?.failures ?? 0} · 경고 {run.result?.warnings ?? 0}</p>{run.result?.passed && <LuBadgeCheck aria-label="통과" />}</article>)}
      </section>
    </div>
  );
}

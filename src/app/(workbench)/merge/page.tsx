"use client";

import { useEffect, useMemo, useState, type FormEvent } from "react";
import { LuMerge, LuShieldAlert } from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import {
  listDatasets,
  mergeDatasets,
  type DatasetSummary,
} from "@/lib/workbench-api";

export default function MergePage() {
  const { currentProfile, openProfileDialog } = useProfile();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [outputName, setOutputName] = useState("");
  const [robotType, setRobotType] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    void listDatasets().then((items) =>
      setDatasets(
        items.filter((item) => item.available && item.readiness === "ready"),
      ),
    );
  }, []);

  const selectedDatasets = useMemo(
    () => datasets.filter((item) => selected.includes(item.id)),
    [datasets, selected],
  );
  const compatible = useMemo(() => {
    if (selectedDatasets.length < 2) return true;
    const first = selectedDatasets[0];
    return selectedDatasets.every(
      (item) =>
        item.codebase_version === first.codebase_version &&
        item.fps === first.fps,
    );
  }, [selectedDatasets]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    setSubmitting(true);
    setMessage(null);
    try {
      const job = await mergeDatasets(
        currentProfile.id,
        selected,
        outputName.trim(),
        robotType.trim(),
        crypto.randomUUID(),
      );
      setMessage(`합치기 작업을 시작했습니다. (${job.id.slice(0, 8)})`);
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "합치기를 시작하지 못했습니다.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">MERGE</p>
          <h1>데이터셋 합치기</h1>
          <p>호환되는 데이터셋을 순서대로 이어 새 작업본을 만듭니다.</p>
        </div>
      </section>
      <form className="curation-builder" onSubmit={submit}>
        <div className="curation-section-title">
          <span>01</span>
          <div>
            <h2>원본 선택</h2>
            <p>두 개 이상 선택하세요. 원본은 변경되지 않습니다.</p>
          </div>
        </div>
        <div className="curation-recipe-list">
          {datasets.map((dataset) => (
            <label key={dataset.id} className="curation-trim-toggle">
              <input
                type="checkbox"
                checked={selected.includes(dataset.id)}
                onChange={(event) =>
                  setSelected((current) =>
                    event.target.checked
                      ? [...current, dataset.id]
                      : current.filter((id) => id !== dataset.id),
                  )
                }
              />
              <LuMerge aria-hidden />
              <span>
                <strong>{dataset.name}</strong>
                <small>
                  {dataset.codebase_version} · {dataset.fps} FPS ·{" "}
                  {dataset.total_episodes ?? 0} episodes
                </small>
              </span>
            </label>
          ))}
        </div>
        {!compatible && (
          <p className="workbench-live-message">
            <LuShieldAlert aria-hidden /> 버전과 FPS가 같은 데이터셋만 합칠 수
            있습니다.
          </p>
        )}
        <div className="curation-section-title">
          <span>02</span>
          <div>
            <h2>새 데이터셋</h2>
            <p>결과 이름과 robot_type을 지정하세요.</p>
          </div>
        </div>
        <div className="curation-trim-fields">
          <label>
            <span>출력 이름</span>
            <input
              value={outputName}
              onChange={(event) => setOutputName(event.target.value)}
              required
            />
          </label>
          <label>
            <span>robot_type</span>
            <input
              value={robotType}
              onChange={(event) => setRobotType(event.target.value)}
              placeholder={selectedDatasets[0]?.robot_type ?? "rby1"}
              required
            />
          </label>
        </div>
        <button
          className="workbench-button workbench-button--primary"
          disabled={
            submitting ||
            selected.length < 2 ||
            !compatible ||
            !outputName.trim() ||
            !robotType.trim()
          }
        >
          <LuMerge aria-hidden />{" "}
          {submitting ? "시작 중…" : "새 데이터셋 만들기"}
        </button>
        <p className="workbench-live-message" aria-live="polite">
          {message}
        </p>
      </form>
    </div>
  );
}

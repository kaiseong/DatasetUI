"use client";

import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { LuMerge, LuShieldAlert } from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import { MergeJobProgress } from "@/components/workbench/merge-job-progress";
import {
  listDatasets,
  listMergeJobs,
  isActiveJob,
  mergeDatasets,
  type DatasetSummary,
  type Job,
} from "@/lib/workbench-api";

export default function MergePage() {
  const { currentProfile, openProfileDialog } = useProfile();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [outputName, setOutputName] = useState("");
  const [robotType, setRobotType] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [pollError, setPollError] = useState(false);
  const [loadedProfileId, setLoadedProfileId] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const submittingRef = useRef(false);
  const pendingIntent = useRef<{ signature: string; key: string } | null>(null);
  const profileId = currentProfile?.id;
  const latestProfileId = useRef(profileId);
  useEffect(() => {
    latestProfileId.current = profileId;
    setMessage(null);
  }, [profileId]);
  const visibleJobs = jobs.filter(
    (job) => job.profile_id === profileId && job.kind === "datasets.merge",
  );
  const hasActiveOutput = visibleJobs.some(
    (job) =>
      isActiveJob(job.status) && job.payload.output_name === outputName.trim(),
  );

  useEffect(() => {
    let active = true;
    void listDatasets()
      .then(
        (items) =>
          active &&
          setDatasets(
            items.filter(
              (item) => item.available && item.readiness === "ready",
            ),
          ),
      )
      .catch(
        () => active && setMessage("데이터셋 목록을 불러오지 못했습니다."),
      );
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!profileId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setPollError(false);
    async function poll() {
      try {
        const loaded = await listMergeJobs(profileId!, controller.signal);
        if (active) {
          setJobs(loaded);
          setLoadedProfileId(profileId!);
          setPollError(false);
        }
      } catch {
        if (active) setPollError(true);
      } finally {
        if (active) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [profileId, refresh]);

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
    if (submittingRef.current || hasActiveOutput) return;
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    submittingRef.current = true;
    setSubmitting(true);
    setMessage(null);
    try {
      const signature = JSON.stringify([
        currentProfile.id,
        selected,
        outputName.trim(),
        robotType.trim(),
      ]);
      if (pendingIntent.current?.signature !== signature) {
        pendingIntent.current = { signature, key: crypto.randomUUID() };
      }
      const job = await mergeDatasets(
        currentProfile.id,
        selected,
        outputName.trim(),
        robotType.trim(),
        pendingIntent.current.key,
      );
      pendingIntent.current = null;
      if (latestProfileId.current !== currentProfile.id) return;
      setJobs((current) => [
        job,
        ...current.filter((item) => item.id !== job.id),
      ]);
      setRefresh((value) => value + 1);
      setMessage(
        `합치기 요청이 접수됐습니다. 아래에서 진행 상태를 확인하세요. (${job.id.slice(0, 8)})`,
      );
    } catch (error) {
      if (latestProfileId.current !== currentProfile.id) return;
      setMessage(
        error instanceof Error
          ? error.message
          : "합치기를 시작하지 못했습니다.",
      );
    } finally {
      submittingRef.current = false;
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
            hasActiveOutput ||
            selected.length < 2 ||
            !compatible ||
            !outputName.trim() ||
            !robotType.trim()
          }
        >
          <LuMerge aria-hidden />{" "}
          {submitting
            ? "시작 중…"
            : hasActiveOutput
              ? "같은 출력 이름의 합치기 진행 중"
              : "새 데이터셋 만들기"}
        </button>
        <p className="workbench-live-message" aria-live="polite">
          {message}
        </p>
      </form>
      <section className="merge-job-list" aria-label="합치기 진행 상황">
        <h2 className="text-lg font-semibold">합치기 진행 상황</h2>
        <p>
          현재 사용자의 최근 합치기 작업을 2초마다 확인합니다. 새로고침해도 다시
          불러옵니다.
        </p>
        {!profileId && (
          <p>사용자를 선택하면 해당 사용자의 작업이 표시됩니다.</p>
        )}
        {pollError && (
          <p role="alert">
            {loadedProfileId === profileId
              ? "작업 상태 연결이 지연되고 있습니다. 마지막 확인 상태를 유지하며 다시 시도합니다."
              : "작업 목록을 불러오지 못했습니다. 자동으로 다시 시도합니다."}
          </p>
        )}
        {profileId && !pollError && visibleJobs.length === 0 && (
          <p>
            {loadedProfileId === profileId
              ? "아직 표시할 합치기 작업이 없습니다."
              : "합치기 작업을 불러오는 중입니다."}
          </p>
        )}
        {visibleJobs.map((job) => (
          <MergeJobProgress
            key={job.id}
            job={job}
            datasets={datasets}
            currentProfileId={profileId}
            onJobUpdate={(updated) =>
              setJobs((current) =>
                current.map((item) =>
                  item.id === updated.id ? updated : item,
                ),
              )
            }
            stale={pollError}
          />
        ))}
      </section>
    </div>
  );
}

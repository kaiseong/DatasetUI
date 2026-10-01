"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { LuCloudUpload, LuHardDrive, LuLaptop, LuSend } from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import {
  copyDatasetToPcWithKey,
  copyDatasetToPcWithPassword,
  exportDatasetToNas,
  listDatasets,
  getDeliveryCapabilities,
  uploadDatasetToHuggingFace,
  type DatasetSummary,
} from "@/lib/workbench-api";

export default function DeliverPage() {
  const { currentProfile, openProfileDialog } = useProfile();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [nasName, setNasName] = useState("");
  const [repoName, setRepoName] = useState("");
  const [visibility, setVisibility] = useState<"private" | "public">("private");
  const [auth, setAuth] = useState<"key" | "password">("key");
  const [host, setHost] = useState("");
  const [port, setPort] = useState(22);
  const [username, setUsername] = useState("");
  const [destination, setDestination] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const currentProfileId = currentProfile?.id;
  const latestProfileId = useRef(currentProfileId);
  useEffect(() => {
    latestProfileId.current = currentProfileId;
    setMessage(null);
    setPassword("");
  }, [currentProfileId]);
  const [pendingLabel, setPendingLabel] = useState("");
  const [hfConfigured, setHfConfigured] = useState<boolean | null>(null);
  const [capabilitiesError, setCapabilitiesError] = useState(false);
  useEffect(() => {
    let active = true;
    void getDeliveryCapabilities()
      .then((value) => {
        if (active) setHfConfigured(value.hf_upload_configured);
      })
      .catch(() => {
        if (active) setCapabilitiesError(true);
      });
    return () => {
      active = false;
    };
  }, []);
  const selected = datasets.find((item) => item.id === datasetId);
  const selectedName = selected?.name;
  useEffect(() => {
    setPassword("");
  }, [datasetId]);

  useEffect(() => {
    void listDatasets().then((items) => {
      const ready = items.filter(
        (item) => item.available && item.readiness === "ready",
      );
      setDatasets(ready);
      setDatasetId(ready[0]?.id ?? "");
    });
  }, []);
  useEffect(() => {
    if (!selectedName) return;
    setNasName(selectedName);
    setRepoName(selectedName);
    setDestination(`~/${selectedName}`);
  }, [selectedName]);

  function profileId(): string | null {
    if (!currentProfile) {
      openProfileDialog();
      return null;
    }
    return currentProfile.id;
  }
  async function run(
    action: () => Promise<object>,
    success: string,
    pending = "전달 요청 접수 중",
  ) {
    if (busyRef.current) return;
    busyRef.current = true;
    const submittedProfile = currentProfileId;
    setPendingLabel(pending);
    setBusy(true);
    setMessage(null);
    try {
      const result = await action();
      const jobId =
        "id" in result && typeof result.id === "string" ? result.id : null;
      if (latestProfileId.current === submittedProfile)
        setMessage(`${success}${jobId ? ` (${jobId.slice(0, 8)})` : ""}`);
    } catch (error) {
      if (latestProfileId.current === submittedProfile)
        setMessage(
          error instanceof Error
            ? error.message
            : "전달을 시작하지 못했습니다.",
        );
    } finally {
      busyRef.current = false;
      setPassword("");
      setBusy(false);
    }
  }
  async function nas(event: FormEvent) {
    event.preventDefault();
    const id = profileId();
    if (!id) return;
    await run(
      () =>
        exportDatasetToNas(datasetId, id, nasName.trim(), crypto.randomUUID()),
      "NAS 전달 요청을 접수했습니다. 필요한 검사 후 자동으로 전달합니다.",
    );
  }
  async function hf(event: FormEvent) {
    event.preventDefault();
    if (hfConfigured !== true) return;
    const id = profileId();
    if (!id) return;
    await run(
      () =>
        uploadDatasetToHuggingFace(
          datasetId,
          id,
          repoName.trim(),
          visibility,
          crypto.randomUUID(),
        ),
      "Hugging Face 업로드 요청을 접수했습니다. 필요한 검사 후 자동으로 업로드합니다.",
    );
  }
  async function pc(event: FormEvent) {
    event.preventDefault();
    const id = profileId();
    if (!id) return;
    const target = { host, port, username, destination };
    if (auth === "key") {
      await run(
        () =>
          copyDatasetToPcWithKey(datasetId, id, target, crypto.randomUUID()),
        "PC 전달 요청을 접수했습니다. 필요한 검사 후 자동으로 전송합니다.",
      );
    } else {
      await run(
        () =>
          copyDatasetToPcWithPassword(
            datasetId,
            id,
            target,
            password,
            crypto.randomUUID(),
          ),
        "PC 전달 요청을 접수했습니다. 화면을 이동해도 검사와 전송이 이어집니다.",
      );
    }
  }

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">DELIVER</p>
          <h1>데이터셋 전달</h1>
          <p>필요한 전체 검사를 자동으로 실행하고, 통과한 결과를 전달합니다.</p>
        </div>
      </section>
      <div className="delivery-dataset-picker">
        <label>
          <span>데이터셋</span>
          <select
            value={datasetId}
            onChange={(event) => setDatasetId(event.target.value)}
          >
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name} · {dataset.codebase_version}
              </option>
            ))}
          </select>
        </label>
      </div>
      <section className="delivery-grid">
        <form className="delivery-card" onSubmit={nas}>
          <span className="delivery-card__icon">
            <LuHardDrive aria-hidden />
          </span>
          <h2>NAS Export</h2>
          <p>공유 exports 영역에 검증된 복사본을 만듭니다.</p>
          <label className="delivery-field">
            <span>Export 이름</span>
            <input
              value={nasName}
              onChange={(event) => setNasName(event.target.value)}
              required
            />
          </label>
          <button className="workbench-button" disabled={busy || !datasetId}>
            <LuSend aria-hidden /> 저장
          </button>
        </form>
        <form className="delivery-card" onSubmit={hf}>
          <span className="delivery-card__icon">
            <LuCloudUpload aria-hidden />
          </span>
          <h2>Hugging Face</h2>
          <p>rainbowrobotics 아래 새 저장소만 만들며 기본값은 비공개입니다.</p>
          {hfConfigured === false && (
            <p role="status">
              서버에 Hugging Face 쓰기 토큰이 설정되지 않았습니다. 관리자 설정
              후 업로드할 수 있습니다.
            </p>
          )}
          {capabilitiesError && (
            <p role="status">
              업로드 설정을 확인하지 못했습니다. 새로고침 후 다시 확인하세요.
            </p>
          )}
          {hfConfigured === null && !capabilitiesError && (
            <p>업로드 설정 확인 중…</p>
          )}
          <label className="delivery-field">
            <span>저장소 이름</span>
            <input
              value={repoName}
              onChange={(event) => setRepoName(event.target.value)}
              required
            />
          </label>
          <label className="delivery-field">
            <span>공개 범위</span>
            <select
              value={visibility}
              onChange={(event) =>
                setVisibility(event.target.value as "private" | "public")
              }
            >
              <option value="private">비공개</option>
              <option value="public">공개</option>
            </select>
          </label>
          <button
            className="workbench-button"
            disabled={busy || !datasetId || hfConfigured !== true}
          >
            <LuSend aria-hidden /> 업로드
          </button>
        </form>
        <form className="delivery-card delivery-card--wide" onSubmit={pc}>
          <span className="delivery-card__icon">
            <LuLaptop aria-hidden />
          </span>
          <h2>Ubuntu PC</h2>
          <p>
            SSH로 복사합니다. 비밀번호는 별도 메모리 저장소에 일시 보관하며,
            처리·취소·만료 후 삭제합니다. 서버가 재시작되면 다시 요청해야
            합니다.
          </p>
          <div className="delivery-card__fields">
            <label className="delivery-field">
              <span>IP</span>
              <input
                value={host}
                onChange={(event) => setHost(event.target.value)}
                placeholder="192.168.0.51"
                required
              />
            </label>
            <label className="delivery-field">
              <span>포트</span>
              <input
                type="number"
                value={port}
                onChange={(event) => setPort(Number(event.target.value))}
                required
              />
            </label>
            <label className="delivery-field">
              <span>사용자 이름</span>
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                required
              />
            </label>
            <label className="delivery-field">
              <span>저장 위치</span>
              <input
                value={destination}
                onChange={(event) => setDestination(event.target.value)}
                required
              />
            </label>
          </div>
          <div className="delivery-auth">
            <label>
              <input
                type="radio"
                checked={auth === "key"}
                onChange={() => {
                  setPassword("");
                  setAuth("key");
                }}
              />{" "}
              등록된 SSH key
            </label>
            <label>
              <input
                type="radio"
                checked={auth === "password"}
                onChange={() => setAuth("password")}
              />{" "}
              일회용 비밀번호
            </label>
          </div>
          {auth === "password" && (
            <label className="delivery-field delivery-password">
              <span>일회용 SSH 비밀번호</span>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="off"
                required
                aria-label="일회용 SSH 비밀번호"
                disabled={busy}
              />
            </label>
          )}
          <button className="workbench-button" disabled={busy || !datasetId}>
            <LuSend aria-hidden /> PC로 복사
          </button>
        </form>
      </section>
      {busy && (
        <section
          className="generic-job-progress"
          aria-label="전달 요청 진행 상황"
        >
          <p role="status">{pendingLabel}</p>
          <div
            className="validation-progress"
            data-active="true"
            data-determinate="false"
          >
            <div
              className="validation-progress__track"
              role="progressbar"
              aria-label="전달 요청 처리 중"
            >
              <span />
            </div>
          </div>
        </section>
      )}
      <p className="workbench-live-message" aria-live="polite">
        {message}
      </p>
    </div>
  );
}

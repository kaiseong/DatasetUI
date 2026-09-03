"use client";

import { useEffect, useState, type FormEvent } from "react";
import { LuCloudUpload, LuHardDrive, LuLaptop, LuSend } from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";
import {
  copyDatasetToPcWithKey,
  copyDatasetToPcWithPassword,
  exportDatasetToNas,
  listDatasets,
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
  const selected = datasets.find((item) => item.id === datasetId);
  const selectedName = selected?.name;

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
  async function run(action: () => Promise<object>, success: string) {
    setBusy(true);
    setMessage(null);
    try {
      const result = await action();
      const jobId =
        "id" in result && typeof result.id === "string" ? result.id : null;
      setMessage(`${success}${jobId ? ` (${jobId.slice(0, 8)})` : ""}`);
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "전달을 시작하지 못했습니다.",
      );
    } finally {
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
      "NAS Export를 시작했습니다.",
    );
  }
  async function hf(event: FormEvent) {
    event.preventDefault();
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
      "Hugging Face 업로드를 시작했습니다.",
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
        "PC 복사를 시작했습니다.",
      );
    } else {
      await run(
        () => copyDatasetToPcWithPassword(datasetId, id, target, password),
        "PC 복사를 마쳤습니다.",
      );
    }
  }

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">DELIVER</p>
          <h1>데이터셋 전달</h1>
          <p>내보내기 검사를 통과한 현재 리비전만 전달할 수 있습니다.</p>
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
          <button className="workbench-button" disabled={busy || !datasetId}>
            <LuSend aria-hidden /> 업로드
          </button>
        </form>
        <form className="delivery-card delivery-card--wide" onSubmit={pc}>
          <span className="delivery-card__icon">
            <LuLaptop aria-hidden />
          </span>
          <h2>Ubuntu PC</h2>
          <p>SSH로 복사합니다. 비밀번호는 이 요청 중에만 메모리에 있습니다.</p>
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
                onChange={() => setAuth("key")}
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
                autoComplete="new-password"
                required
                aria-label="일회용 SSH 비밀번호"
              />
            </label>
          )}
          <button className="workbench-button" disabled={busy || !datasetId}>
            <LuSend aria-hidden /> PC로 복사
          </button>
        </form>
      </section>
      <p className="workbench-live-message" aria-live="polite">
        {message}
      </p>
    </div>
  );
}

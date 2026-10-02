"use client";

import { useEffect, useState } from "react";
import { useProfile } from "./profile-context";
import {
  armDimensionStatus,
  augmentationSummary,
  DEFAULT_JOINT_RANGES_DEG,
  MAX_COPIES,
  MAX_JOINT_RANGE_DEG,
  sampleEpisodes,
  validRanges,
} from "@/lib/joint-offset";
import {
  createJointOffsetAugmentation,
  getDatasetInfo,
  type Job,
} from "@/lib/workbench-api";
import "./segmentation-workflow.css";

type Props = { datasetId: string; datasetName: string };

const OUTPUT_NAME = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;

/** Per-episode arm joint offsets added to both observation.state and action. */
export default function JointOffsetAugmentation({
  datasetId,
  datasetName,
}: Props) {
  const { currentProfile, openProfileDialog } = useProfile();
  const [info, setInfo] = useState<unknown>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [ranges, setRanges] = useState<number[]>(DEFAULT_JOINT_RANGES_DEG);
  const [copies, setCopies] = useState(1);
  const [seed, setSeed] = useState(0);
  const [scope, setScope] = useState<"all" | "selected">("all");
  const [selected, setSelected] = useState<number[]>([]);
  const [percent, setPercent] = useState(30);
  const [sampleSeed, setSampleSeed] = useState(0);
  const [outputName, setOutputName] = useState(`${datasetName}-joint-offset`);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setInfo(null);
    setLoadError(null);
    void getDatasetInfo(datasetId)
      .then((value) => {
        if (active) setInfo(value);
      })
      .catch((error: unknown) => {
        if (active)
          setLoadError(
            error instanceof Error
              ? error.message
              : "데이터셋 정보를 불러오지 못했습니다.",
          );
      });
    return () => {
      active = false;
    };
  }, [datasetId]);

  const total = Number(
    (info as { total_episodes?: number } | null)?.total_episodes ?? 0,
  );
  const episodes = Array.from({ length: total }, (_, index) => index);
  const status = armDimensionStatus(info);
  const augmented = scope === "all" ? total : selected.length;
  const summary = augmentationSummary(total, augmented, copies);
  const nameValid = OUTPUT_NAME.test(outputName) && !outputName.includes("..");
  const canRun =
    !!info &&
    status.ready &&
    validRanges(ranges) &&
    augmented > 0 &&
    nameValid &&
    !busy;

  async function run() {
    if (!currentProfile) {
      openProfileDialog();
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const created = await createJointOffsetAugmentation(
        currentProfile.id,
        datasetId,
        {
          ranges_deg: ranges,
          copies,
          seed,
          episode_indices: scope === "all" ? null : selected,
          output_name: outputName.trim(),
        },
        crypto.randomUUID(),
      );
      setJob(created);
      setMessage(
        "관절 오프셋 증강 작업을 등록했습니다. 하단 작업 진행 상황에서 확인하세요.",
      );
    } catch (error) {
      setJob(null);
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section
      className="segmentation-panel"
      aria-label="관절 오프셋 증강"
      data-joint-offset
    >
      <div className="segmentation-panel-heading">
        <span>JOINT OFFSET</span>
        <h2>관절 오프셋 증강</h2>
        <p>
          에피소드마다 팔 관절별로 균등분포 ±범위에서 오프셋을 한 번 뽑아, 그
          에피소드의 모든 프레임 observation.state와 action에 같은 값을
          더합니다(도 단위를 라디안으로 변환, 오른팔·왼팔 따로, 그리퍼 제외).
          원본 에피소드는 그대로 두고 증강 사본을 추가한 새 데이터셋을 만듭니다.
        </p>
      </div>

      {loadError && (
        <p className="workbench-live-message" role="alert">
          {loadError}
        </p>
      )}
      {!info && !loadError && (
        <p className="workbench-live-message" aria-live="polite">
          데이터셋 정보를 불러오는 중…
        </p>
      )}
      {!!info && !status.ready && (
        <p className="workbench-live-message" role="alert">
          이 데이터셋에는 RBY1 팔 관절 이름(right_arm_0..6, left_arm_0..6)이
          없어 적용할 수 없습니다. 없는 이름:{" "}
          {status.missing.slice(0, 6).join(", ")}
          {status.missing.length > 6 ? " …" : ""}
        </p>
      )}

      <fieldset className="segmentation-joint-ranges">
        <legend>관절별 범위 (±도)</legend>
        {ranges.map((value, joint) => (
          <label key={joint}>
            j{joint}
            <input
              type="number"
              min={0}
              max={MAX_JOINT_RANGE_DEG}
              step={0.05}
              value={value}
              onChange={(event) => {
                const next = event.target.valueAsNumber;
                setRanges((current) =>
                  current.map((item, index) => (index === joint ? next : item)),
                );
              }}
            />
          </label>
        ))}
        <button
          type="button"
          className="workbench-button"
          onClick={() => setRanges(DEFAULT_JOINT_RANGES_DEG)}
        >
          기본값
        </button>
      </fieldset>
      {!validRanges(ranges) && (
        <p className="workbench-live-message" role="alert">
          범위는 0~{MAX_JOINT_RANGE_DEG}도 사이로 입력하세요.
        </p>
      )}

      <div className="segmentation-fields">
        <label>
          에피소드당 증강 사본 수
          <input
            type="number"
            min={1}
            max={MAX_COPIES}
            value={copies}
            onChange={(event) =>
              setCopies(
                Math.min(
                  MAX_COPIES,
                  Math.max(1, Math.round(event.target.valueAsNumber) || 1),
                ),
              )
            }
          />
        </label>
        <label>
          오프셋 시드
          <input
            type="number"
            min={0}
            value={seed}
            onChange={(event) =>
              setSeed(Math.max(0, Math.round(event.target.valueAsNumber) || 0))
            }
          />
        </label>
      </div>

      <fieldset className="segmentation-joint-scope">
        <legend>증강할 에피소드</legend>
        <label>
          <input
            type="radio"
            name={`joint-offset-scope-${datasetId}`}
            checked={scope === "all"}
            onChange={() => setScope("all")}
          />{" "}
          전체 {total}개
        </label>
        <label>
          <input
            type="radio"
            name={`joint-offset-scope-${datasetId}`}
            checked={scope === "selected"}
            onChange={() => setScope("selected")}
          />{" "}
          선택
        </label>
      </fieldset>

      {scope === "selected" && (
        <div className="segmentation-selection">
          <fieldset>
            <legend>선택한 에피소드 · {selected.length}개</legend>
            <div className="segmentation-fields">
              <label>
                비율 (%)
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={percent}
                  onChange={(event) =>
                    setPercent(
                      Math.min(
                        100,
                        Math.max(
                          1,
                          Math.round(event.target.valueAsNumber) || 1,
                        ),
                      ),
                    )
                  }
                />
              </label>
              <label>
                샘플링 시드
                <input
                  type="number"
                  min={0}
                  value={sampleSeed}
                  onChange={(event) =>
                    setSampleSeed(
                      Math.max(0, Math.round(event.target.valueAsNumber) || 0),
                    )
                  }
                />
              </label>
            </div>
            <div className="segmentation-selection-actions">
              <button
                type="button"
                className="workbench-button"
                onClick={() =>
                  setSelected(sampleEpisodes(episodes, percent, sampleSeed))
                }
              >
                랜덤 샘플링
              </button>
              <button
                type="button"
                className="workbench-button"
                onClick={() => setSelected(episodes)}
              >
                전체 선택
              </button>
              <button
                type="button"
                className="workbench-button"
                onClick={() => setSelected([])}
              >
                선택 해제
              </button>
            </div>
            <p className="segmentation-note">
              랜덤 샘플링은 아래 체크박스만 채웁니다. 실행 전에 직접 더하거나 뺄
              수 있습니다.
            </p>
            <div className="segmentation-episode-grid">
              {episodes.map((episode) => (
                <label key={episode}>
                  <input
                    type="checkbox"
                    checked={selected.includes(episode)}
                    onChange={(event) =>
                      setSelected((current) =>
                        event.target.checked
                          ? [...current, episode].sort((a, b) => a - b)
                          : current.filter((value) => value !== episode),
                      )
                    }
                  />{" "}
                  {episode}
                </label>
              ))}
            </div>
          </fieldset>
        </div>
      )}

      <p className="segmentation-note" aria-live="polite">
        결과: 원본 {total} + 증강 {augmented}×{copies} = 총 {summary.output}
        에피소드 · 영상 용량 약 {summary.videoFactor.toFixed(2)}배
      </p>

      <div className="segmentation-fields">
        <label>
          출력 이름
          <input
            value={outputName}
            onChange={(event) => setOutputName(event.target.value)}
          />
        </label>
        <button
          type="button"
          className="workbench-button workbench-button--primary"
          disabled={!canRun}
          onClick={() => void run()}
        >
          {busy ? "등록 중…" : "관절 오프셋 증강 실행"}
        </button>
      </div>
      {!nameValid && (
        <p className="workbench-live-message" role="alert">
          출력 이름은 영문·숫자로 시작/끝나야 하며 . _ - 만 사용할 수 있습니다.
        </p>
      )}
      {message && (
        <p className="workbench-live-message" role={job ? "status" : "alert"}>
          {message}
        </p>
      )}
    </section>
  );
}

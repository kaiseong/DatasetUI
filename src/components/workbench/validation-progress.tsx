import type {
  ValidationCheck,
  ValidationCheckStatus,
  ValidationProgressSnapshot,
  ValidationRun,
} from "@/lib/workbench-api";

const STAGE_LABELS: Record<ValidationProgressSnapshot["stage"], string> = {
  metadata: "메타데이터",
  data: "에피소드 데이터",
  video: "영상",
  statistics: "통계",
  complete: "검사 마무리",
};

const CHECK_STATUS_LABELS: Record<ValidationCheckStatus, string> = {
  pending: "대기",
  running: "검사 중",
  passed: "통과",
  warning: "경고",
  failed: "실패",
  skipped: "미검사",
};

const CHECK_LABELS: Record<string, string> = {
  metadata: "메타데이터·데이터셋 구조",
  indices: "에피소드·프레임 인덱스",
  timestamps: "타임스탬프·FPS 일관성",
  features: "전체 수치 feature·Action·State",
  videos: "영상 디코딩·프레임 범위",
  statistics: "데이터셋 통계",
};

function clampPercent(completed: number, total: number): number | null {
  if (!Number.isFinite(completed) || !Number.isFinite(total) || total <= 0) {
    return null;
  }
  return Math.min(100, Math.max(0, Math.round((completed / total) * 100)));
}

function formatElapsed(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainder = safeSeconds % 60;
  if (hours > 0) return `${hours}시간 ${minutes}분 ${remainder}초`;
  if (minutes > 0) return `${minutes}분 ${remainder}초`;
  return `${remainder}초`;
}

function CheckList({
  checks,
  terminal = false,
}: {
  checks: ValidationCheck[];
  terminal?: boolean;
}) {
  return (
    <ul className="validation-checks" aria-label="검사 항목별 결과">
      {checks.map((check) => {
        const interrupted =
          terminal &&
          (check.status === "pending" || check.status === "running");
        return (
          <li
            key={check.id}
            data-status={interrupted ? "interrupted" : check.status}
          >
            <span className="validation-checks__marker" aria-hidden />
            <div>
              <strong>{CHECK_LABELS[check.id] ?? check.label}</strong>
              {check.detail && <p>{check.detail}</p>}
            </div>
            <span className="validation-checks__status">
              {interrupted ? "완료 전 중단" : CHECK_STATUS_LABELS[check.status]}
              {(check.failures > 0 || check.warnings > 0) && (
                <small>
                  실패 {check.failures} · 경고 {check.warnings}
                </small>
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export function ValidationProgress({
  run,
  stale = false,
}: {
  run: ValidationRun;
  stale?: boolean;
}) {
  const active = run.status === "queued" || run.status === "running";
  const completed = run.status === "succeeded" && run.result !== null;
  const legacyPolicy =
    completed && run.result?.validator_policy !== "datasetui-semantic-v2";
  const terminalWithoutResult = !active && !completed;
  const progress = run.progress ?? null;
  const progressPercent = progress
    ? clampPercent(progress.completed, progress.total)
    : null;
  const checks = run.result?.checks ?? progress?.checks ?? null;
  const title = legacyPolicy
    ? "이전 검사 기준 · 재검사 필요"
    : stale && active
      ? "상태 확인 지연"
      : run.status === "queued"
        ? "대기 중"
        : run.status === "running"
          ? "검사 중"
          : completed
            ? run.result?.passed
              ? "검사 완료 · 통과"
              : "검사 완료 · 문제 발견"
            : run.status === "failed"
              ? "검사 실패"
              : "검사 중단 또는 결과 확인 필요";

  return (
    <div
      className="validation-progress"
      data-active={active && !stale}
      data-determinate={active && progressPercent !== null}
      data-status={run.status}
    >
      <strong role="status">{title}</strong>
      {active && (
        <>
          <div
            className="validation-progress__track"
            role="progressbar"
            aria-label={
              progressPercent !== null ? "에피소드 검사 진행률" : title
            }
            aria-valuemin={progressPercent !== null ? 0 : undefined}
            aria-valuemax={progressPercent !== null ? 100 : undefined}
            aria-valuenow={progressPercent ?? undefined}
            aria-valuetext={stale ? "마지막으로 확인된 진행 상태" : undefined}
          >
            <span
              style={
                progressPercent !== null
                  ? { width: `${progressPercent}%` }
                  : undefined
              }
            />
          </div>
          {!progress && (
            <p>
              {stale
                ? "서버 상태를 갱신하지 못했습니다. 마지막으로 확인된 상태를 표시합니다."
                : run.status === "queued"
                  ? "작업자 배정을 기다리고 있습니다. 검사 결과가 아닙니다."
                  : "이 작업은 아직 처리량을 보고하지 않아 진행률을 계산할 수 없습니다."}
            </p>
          )}
          {!stale && (
            <small>
              2초마다 상태를 확인하며, 완료되면 결과로 자동 전환됩니다.
            </small>
          )}
        </>
      )}
      {progress && !completed && (
        <div className="validation-progress__metrics">
          <span>
            <strong>
              {progressPercent === null
                ? "에피소드 진행률 계산 중"
                : `에피소드 진행률 ${progressPercent}%`}
            </strong>
            에피소드 {progress.completed.toLocaleString()} /{" "}
            {progress.total > 0 ? progress.total.toLocaleString() : "확인 중"}
          </span>
          <span>
            현재 단계 <strong>{STAGE_LABELS[progress.stage]}</strong>
          </span>
          <span>
            디코딩 프레임{" "}
            <strong>{progress.decoded_frames.toLocaleString()}</strong>
          </span>
          <span>
            경과 <strong>{formatElapsed(progress.elapsed_seconds)}</strong>
          </span>
        </div>
      )}
      {stale && active && progress && (
        <p className="validation-progress__stale">
          서버 상태를 갱신하지 못했습니다. 수치는 마지막 확인 시점 기준입니다.
        </p>
      )}
      {terminalWithoutResult && progress && (
        <p className="validation-progress__stale">
          검사가 끝나기 전 중단되었습니다. 수치는 마지막 처리 시점 기준입니다.
        </p>
      )}
      {completed && run.result && (
        <p>
          검사한 에피소드 {run.result.checked_episodes} /{" "}
          {run.result.total_episodes}개 · 프레임{" "}
          {run.result.checked_frames.toLocaleString()}개
        </p>
      )}
      {run.finished_at && (
        <small>종료: {new Date(run.finished_at).toLocaleString("ko-KR")}</small>
      )}
      {legacyPolicy ? (
        <p className="validation-progress__legacy">
          검사 기준 강화 전 기록입니다. 현재 기준의 통과 여부는 새 검사로
          확인하세요. 상세 검사 이력을 제공하지 않는 이전 검사 기록도 여기에
          포함됩니다.
        </p>
      ) : checks && checks.length > 0 ? (
        <details>
          <summary className="cursor-pointer font-semibold">
            검사 항목별 결과
          </summary>
          <CheckList checks={checks} terminal={terminalWithoutResult} />
        </details>
      ) : !active && run.result ? (
        <p className="validation-progress__legacy">
          이 결과는 상세 검사 이력을 제공하지 않는 이전 검사 기록입니다.
        </p>
      ) : null}
    </div>
  );
}

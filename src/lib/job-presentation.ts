import type { Job } from "./workbench-api";

const LABELS: Record<string, string> = {
  "datasets.scan": "라이브러리 새로 확인",
  "phase2.smoke": "시스템 확인",
  "hf.import": "Hugging Face 원본 가져오기",
  "curation.materialize": "큐레이션 작업본 생성",
  "datasets.merge": "데이터셋 병합",
  "augment.joint_offset": "관절 오프셋 증강",
  "datasets.validate": "데이터셋 검증",
  "datasets.convert_v21": "v2.1 작업본 변환",
  "datasets.export_nas": "NAS로 내보내기",
  "datasets.upload_hf": "Hugging Face 업로드",
  "datasets.copy_pc_key": "사용자 PC로 복사",
  "datasets.copy_pc_password": "사용자 PC로 복사",
  "datasets.delivery_preflight": "전달 전 전체 검사",
  "hf.delete": "Hugging Face 영구 삭제",
  "datasets.empty_trash": "NAS 휴지통 비우기",
  "segmentation.sample": "현재 프레임 분할 샘플",
  "segmentation.batch_prepare": "Segmentation 원본 확인 · 일괄 준비",
  "segmentation.preview": "작업 영역 마스크 · 미리보기",
  "segmentation.export": "Segmentation 데이터셋 생성",
  "segmentation.batch_export": "Segmentation 일괄 데이터셋 생성",
};

export function jobLabel(job: Pick<Job, "kind">): string {
  return LABELS[job.kind] ?? `작업 · ${job.kind}`;
}

export function jobDescription(job: Pick<Job, "kind" | "payload">): string {
  for (const key of [
    "output_name",
    "dataset_name",
    "repo_name",
    "destination",
  ]) {
    const value = job.payload[key];
    if (typeof value === "string") return value;
  }
  return job.kind === "datasets.scan"
    ? "공유 저장소에서 데이터셋 찾기"
    : "펼쳐서 결과와 진행 기록을 확인하세요.";
}

export function jobResultLinks(job: Pick<Job, "kind" | "status" | "result">) {
  if (job.status !== "succeeded" || !job.result) return [];
  const links: Array<{ label: string; href: string }> = [];
  const id = job.result.dataset_id;
  if (typeof id === "string" && /^[a-f0-9-]{36}$/.test(id)) {
    links.push({ label: "데이터셋 열기", href: `/datasets/${id}/curate` });
  }
  if (["datasets.validate", "datasets.delivery_preflight"].includes(job.kind)) {
    links.push({ label: "상세 검증 결과", href: "/validate" });
  }
  if (
    Array.isArray(job.result.outputs) ||
    ["datasets.merge", "datasets.convert_v21", "augment.joint_offset"].includes(
      job.kind,
    )
  ) {
    links.push({ label: "생성된 데이터셋 목록", href: "/library" });
  }
  const repo = job.result.repo_id;
  if (
    job.kind !== "hf.delete" &&
    typeof repo === "string" &&
    /^[\w.-]+\/[\w.-]+$/.test(repo)
  ) {
    links.push({
      label: "Hugging Face 열기",
      href: `https://huggingface.co/datasets/${repo}`,
    });
  }
  return links;
}

export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  if (hours) return `${hours}시간 ${minutes}분`;
  if (minutes) return `${minutes}분 ${rest}초`;
  return `${rest}초`;
}

/** Whole-job percent and remaining time, extrapolated from elapsed time. */
export function wholeJobEstimate(
  overall: number | undefined,
  elapsedSeconds: number | null,
): { percent: number; remainingSeconds: number | null } | null {
  if (overall === undefined || !Number.isFinite(overall)) return null;
  const fraction = Math.min(1, Math.max(0, overall));
  const remainingSeconds =
    elapsedSeconds !== null &&
    elapsedSeconds >= 3 &&
    fraction >= 0.03 &&
    fraction < 1
      ? (elapsedSeconds * (1 - fraction)) / fraction
      : null;
  return { percent: Math.floor(fraction * 100), remainingSeconds };
}

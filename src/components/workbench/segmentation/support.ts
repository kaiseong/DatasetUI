import type { Job } from "@/lib/workbench-api";

/** Active drawing tool on the frame canvas. */
export type Tool = "positive" | "negative" | "box" | "brush";

/** Which object-guidance tab is open. */
export type EditorPanel = "instruction" | "labeling";

export const MAX_BACKGROUND_BYTES = 10 * 1024 * 1024;
export const BACKGROUND_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
]);
export const OUTPUT_NAME_PATTERN =
  /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;
export const TERMINAL = new Set([
  "succeeded",
  "failed",
  "cancelled",
  "interrupted",
]);

export function jobMessage(job: Job | null): string {
  if (!job) return "";
  if (job.status === "failed")
    return `실패: ${job.error_message ?? job.error_code ?? "UNKNOWN"}`;
  if (job.status === "succeeded") return "완료";
  return job.status === "queued" ? "대기 중" : "처리 중";
}

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function readRawBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("배경 이미지를 읽지 못했습니다."));
    reader.onload = () => {
      const value = String(reader.result ?? "");
      resolve(value.slice(value.indexOf(",") + 1));
    };
    reader.readAsDataURL(file);
  });
}

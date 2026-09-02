import type { WebContents, WebPreferences } from "electron";

export const CONTENT_SECURITY_POLICY = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data: datasetui-media:",
  "media-src 'self' data: blob: datasetui-media:",
  "font-src 'self'",
  "connect-src 'none'",
  "object-src 'none'",
  "base-uri 'none'",
  "frame-ancestors 'none'",
  "form-action 'none'",
].join("; ");

const DOCTOR_HOST = "jashshah999-lerobot-doctor.hf.space";
const PROJECT_ID = /^[A-Za-z0-9_-]{1,128}$/;

export function createSecureWebPreferences(preload: string): WebPreferences {
  return {
    preload,
    contextIsolation: true,
    nodeIntegration: false,
    nodeIntegrationInWorker: false,
    nodeIntegrationInSubFrames: false,
    sandbox: true,
    webSecurity: true,
    allowRunningInsecureContent: false,
    webviewTag: false,
    spellcheck: false,
  };
}

export function isAllowedNavigation(target: string, applicationUrl: string): boolean {
  try {
    const candidate = new URL(target);
    const trusted = new URL(applicationUrl);
    candidate.hash = "";
    trusted.hash = "";
    return candidate.href === trusted.href;
  } catch {
    return false;
  }
}

export function isAllowedExternalUrl(target: string): boolean {
  try {
    const candidate = new URL(target);
    if (
      candidate.protocol !== "https:" ||
      candidate.hostname !== DOCTOR_HOST ||
      candidate.port ||
      candidate.username ||
      candidate.password ||
      candidate.hash ||
      candidate.pathname !== "/"
    ) {
      return false;
    }
    const keys = [...candidate.searchParams.keys()];
    if (keys.length !== 1 || keys[0] !== "dataset") return false;
    const dataset = candidate.searchParams.get("dataset") ?? "";
    return /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(dataset);
  } catch {
    return false;
  }
}

export interface DatasetMediaRequest {
  projectId: string;
  relativePath: string;
}

export function parseDatasetMediaUrl(target: string): DatasetMediaRequest | undefined {
  try {
    const rawPath = target.slice(target.indexOf(":") + 1);
    if (/(^|\/)(?:\.{1,2}|%2e(?:%2e)?)(?:\/|%2f|$)/i.test(rawPath)) return undefined;
    const candidate = new URL(target);
    if (
      candidate.protocol !== "datasetui-media:" ||
      !PROJECT_ID.test(candidate.hostname) ||
      candidate.username ||
      candidate.password ||
      candidate.port ||
      candidate.search ||
      candidate.hash
    ) {
      return undefined;
    }
    const encodedParts = candidate.pathname.split("/").slice(1);
    if (!encodedParts.length || encodedParts.some((part) => !part)) return undefined;
    const parts = encodedParts.map((part) => decodeURIComponent(part));
    if (
      parts.some(
        (part) =>
          !part ||
          part === "." ||
          part === ".." ||
          part.includes("/") ||
          part.includes("\\") ||
          part.includes("\0"),
      )
    ) {
      return undefined;
    }
    return { projectId: candidate.hostname, relativePath: parts.join("/") };
  } catch {
    return undefined;
  }
}

export function installWebContentsGuards(
  webContents: WebContents,
  getApplicationUrl: () => string,
): void {
  webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  webContents.on("will-navigate", (event, target) => {
    if (!isAllowedNavigation(target, getApplicationUrl())) {
      event.preventDefault();
    }
  });
  webContents.on("will-attach-webview", (event) => {
    event.preventDefault();
  });
}

import type { WebContents, WebPreferences } from "electron";

export const CONTENT_SECURITY_POLICY = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'none'",
  "object-src 'none'",
  "base-uri 'none'",
  "frame-ancestors 'none'",
  "form-action 'none'",
].join("; ");

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

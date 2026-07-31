import { resolve } from "node:path";

import { describe, expect, it, vi } from "vitest";

import { buildPythonSpawnSpec } from "../../src/main/python-backend.js";
import {
  CONTENT_SECURITY_POLICY,
  createSecureWebPreferences,
  isAllowedNavigation,
} from "../../src/main/security.js";
import { createDatasetEditorApi } from "../../src/preload/api.js";

describe("Electron security invariants", () => {
  it("enforces isolation, sandboxing, and no renderer Node integration", () => {
    const preferences = createSecureWebPreferences("/trusted/preload.cjs");
    expect(preferences).toMatchObject({
      preload: "/trusted/preload.cjs",
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      webviewTag: false,
    });
  });

  it("allows only the exact application URL and same-document fragments", () => {
    const appUrl = "file:///opt/DatasetUI/out/renderer/index.html";
    expect(isAllowedNavigation(appUrl, appUrl)).toBe(true);
    expect(isAllowedNavigation(`${appUrl}#status`, appUrl)).toBe(true);
    expect(isAllowedNavigation("https://example.com", appUrl)).toBe(false);
    expect(isAllowedNavigation("file:///etc/passwd", appUrl)).toBe(false);
  });

  it("uses a strict CSP without unsafe-inline or unsafe-eval", () => {
    expect(CONTENT_SECURITY_POLICY).toContain("default-src 'none'");
    expect(CONTENT_SECURITY_POLICY).toContain("script-src 'self'");
    expect(CONTENT_SECURITY_POLICY).not.toContain("unsafe-inline");
    expect(CONTENT_SECURITY_POLICY).not.toContain("unsafe-eval");
  });

  it("exposes only frozen ping and report methods over fixed IPC channels", async () => {
    const invoke = vi.fn(async (channel: string) => ({ channel }));
    const api = createDatasetEditorApi(invoke);
    expect(Object.keys(api).sort()).toEqual(["ping", "report"]);
    expect(Object.isFrozen(api)).toBe(true);
    await expect(api.ping()).resolves.toEqual({ channel: "rpc:ping" });
    await expect(api.report()).resolves.toEqual({ channel: "rpc:report" });
    expect(invoke.mock.calls.map(([channel]) => channel)).toEqual(["rpc:ping", "rpc:report"]);
  });

  it("spawns only the fixed module entry without a shell or inherited secret environment", () => {
    const root = resolve("/tmp/DatasetUI");
    const spec = buildPythonSpawnSpec({
      appRoot: root,
      pythonExecutable: "/usr/bin/python3.12",
      parentEnv: {
        PATH: "/usr/bin:/bin",
        HOME: "/home/test",
        LANG: "C.UTF-8",
        AWS_SECRET_ACCESS_KEY: "must-not-cross-boundary",
        HF_TOKEN: "must-not-cross-boundary",
      },
    });
    expect(spec.command).toBe("/usr/bin/python3.12");
    expect(spec.args).toEqual(["-m", "lerobot_dataset_editor.rpc"]);
    expect(spec.options.shell).toBe(false);
    expect(spec.options.cwd).toBe(root);
    expect(spec.options.env).toEqual({
      HOME: "/home/test",
      LANG: "C.UTF-8",
      PATH: "/usr/bin:/bin",
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONPATH: resolve(root, "python/src"),
      PYTHONUNBUFFERED: "1",
    });
  });
});

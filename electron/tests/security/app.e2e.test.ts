/// <reference lib="dom" />

import { resolve } from "node:path";

import electronBinary from "electron";
import { _electron as electron, type ElectronApplication, type Page } from "playwright-core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const ROOT = resolve(import.meta.dirname, "../../..");
let application: ElectronApplication;
let page: Page;

beforeAll(async () => {
  const env = Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
  );
  env.LEROBOT_DATASET_EDITOR_PYTHON = resolve(ROOT, ".venv/bin/python");
  application = await electron.launch({
    executablePath: electronBinary as unknown as string,
    args: [resolve(ROOT, "out/main/index.js"), "--e2e"],
    cwd: ROOT,
    env,
    timeout: 30_000,
  });
  page = await application.firstWindow();
  await page.waitForLoadState("domcontentloaded");
});

afterAll(async () => {
  await application?.close();
});

describe("secure Electron vertical slice", () => {
  it("renders a schema-valid backend report through the fixed preload API", async () => {
    const status = page.getByTestId("backend-status");
    await expect.poll(() => status.textContent(), { timeout: 10_000 }).toContain("ok");
    await expect.poll(() => page.getByTestId("dataset-editor-version").textContent()).toBe("0.1.0");
    await expect.poll(() => page.getByTestId("space-count").textContent()).toBe("37");
  });

  it("keeps Node globals out and exposes only fixed contract methods", async () => {
    const probe = await page.evaluate(() => {
      const api = (window as unknown as { datasetEditor: object }).datasetEditor;
      return {
        requireType: typeof (globalThis as Record<string, unknown>).require,
        processType: typeof (globalThis as Record<string, unknown>).process,
        apiKeys: Object.keys(api).sort(),
        apiFrozen: Object.isFrozen(api),
      };
    });
    expect(probe).toEqual({
      requireType: "undefined",
      processType: "undefined",
      apiKeys: [
        "annotationsExport",
        "annotationsList",
        "annotationsSave",
        "datasetAnalytics",
        "datasetBrowse",
        "datasetEpisode",
        "datasetOpen",
        "datasetSummary",
        "datasetValidate",
        "hubImport",
        "hubInfo",
        "hubSearch",
        "mediaUrl",
        "openExternal",
        "ping",
        "progressRead",
        "projectGet",
        "projectList",
        "projectRegister",
        "projectRemove",
        "projectUpdate",
        "replayMap",
        "report",
        "runtimeDoctor",
        "runtimeSelect",
      ],
      apiFrozen: true,
    });
  });

  it("applies the glassmorphism surface system without inline styles", async () => {
    const glass = await page.locator(".status-card").evaluate((element) => {
      const style = getComputedStyle(element);
      return {
        backdropFilter: style.backdropFilter,
        backgroundColor: style.backgroundColor,
        boxShadow: style.boxShadow,
        borderColor: style.borderColor,
      };
    });
    expect(glass.backdropFilter).toContain("blur(");
    expect(glass.backgroundColor).toMatch(/^rgba\(/);
    expect(glass.boxShadow).not.toBe("none");
    expect(glass.borderColor).toMatch(/^rgba\(/);
    expect(await page.locator('[style]').count()).toBe(0);
  });

  it("enforces CSP and blocks external navigation and popup creation", async () => {
    const originalUrl = page.url();
    const csp = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute("content");
    expect(csp).toContain("default-src 'none'");
    expect(csp).not.toContain("unsafe-inline");
    expect(csp).not.toContain("unsafe-eval");

    const evalBlocked = await page.getByTestId("csp-eval-blocked").textContent();
    const scriptSecurity = await page.evaluate(() => {
      const script = document.createElement("script");
      script.textContent = "globalThis.__datasetUiInlineExecuted = true";
      document.body.append(script);
      return {
        inlineExecuted: Boolean((globalThis as Record<string, unknown>).__datasetUiInlineExecuted),
        popupWasNull: window.open("https://example.com") === null,
      };
    });
    expect(evalBlocked).toBe("true");
    expect(scriptSecurity).toEqual({
      inlineExecuted: false,
      popupWasNull: true,
    });

    await page.evaluate(() => {
      window.location.href = "https://example.com/blocked";
    });
    await page.waitForTimeout(250);
    expect(page.url()).toBe(originalUrl);
    expect(application.windows()).toHaveLength(1);
  });
});

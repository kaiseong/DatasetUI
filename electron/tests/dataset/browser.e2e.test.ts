/// <reference lib="dom" />

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import electronBinary from "electron";
import { _electron as electron, type ElectronApplication, type Page } from "playwright-core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { DatasetEditorApi } from "../../src/preload/api.js";

const ROOT = resolve(import.meta.dirname, "../../..");
let application: ElectronApplication;
let page: Page;
let home: string;

beforeAll(async () => {
  home = await mkdtemp(join(tmpdir(), "datasetui-browser-e2e-"));
  const env = Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
  );
  env.HOME = home;
  env.XDG_DATA_HOME = join(home, "data");
  env.XDG_CACHE_HOME = join(home, "cache");
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
  await rm(home, { recursive: true, force: true });
});

describe("version-neutral side-by-side browser", () => {
  it("opens v2.1 and v3 through the same document UI", async () => {
    await page.evaluate(async ({ v21, v30 }) => {
      const api = (window as unknown as { datasetEditor: DatasetEditorApi }).datasetEditor;
      await api.projectRegister({
        name: "Fixture v2.1",
        source_path: v21,
        target_format: "v2.1",
        runtime_mode: "embedded",
      });
      await api.projectRegister({
        name: "Fixture v3",
        source_path: v30,
        target_format: "v3",
        runtime_mode: "embedded",
      });
    }, {
      v21: resolve(ROOT, "fixtures/v21_valid"),
      v30: resolve(ROOT, "fixtures/v30_valid"),
    });

    await page.getByTestId("dataset-refresh").click();
    await page.getByTestId("dataset-left-select").selectOption({ label: "Fixture v2.1" });
    await page.getByTestId("dataset-right-select").selectOption({ label: "Fixture v3" });

    await expect.poll(
      () => page.getByTestId("dataset-version-left").textContent(),
      { timeout: 10_000 },
    ).toBe("v2.1");
    await expect.poll(
      () => page.getByTestId("dataset-version-right").textContent(),
      { timeout: 10_000 },
    ).toBe("v3.0");
    await expect.poll(() => page.getByTestId("dataset-episodes-left").textContent()).toBe("2");
    await expect.poll(() => page.getByTestId("dataset-episodes-right").textContent()).toBe("3");
    await expect.poll(
      () => page.getByTestId("dataset-schema-left").textContent(),
    ).toContain("action");
    await expect.poll(
      () => page.getByTestId("dataset-episode-left-0").textContent(),
    ).toContain("0–10");
    await expect.poll(
      () => page.getByTestId("dataset-episode-right-2").textContent(),
    ).toContain("20–30");
  });
});

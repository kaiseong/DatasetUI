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
  home = await mkdtemp(join(tmpdir(), "datasetui-parity-e2e-"));
  const env = Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
  );
  env.HOME = home;
  env.XDG_CONFIG_HOME = join(home, "config");
  env.XDG_DATA_HOME = join(home, "data");
  env.XDG_STATE_HOME = join(home, "state");
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
  await page.evaluate(async (source) => {
    const api = (window as unknown as { datasetEditor: DatasetEditorApi }).datasetEditor;
    await api.projectRegister({
      name: "org/parity-fixture",
      source_path: source,
      target_format: "v3",
      runtime_mode: "embedded",
    });
  }, resolve(ROOT, "fixtures/v30_valid"));
  await page.getByTestId("refresh-projects").click();
  await page.getByTestId("parity-project").locator("option", { hasText: "org/parity-fixture" }).waitFor({ state: "attached", timeout: 15_000 });
  await page.getByTestId("parity-project").selectOption({ label: "org/parity-fixture" });
  await page.getByLabel("Episode", { exact: true }).waitFor({ timeout: 15_000 });
});

afterAll(async () => {
  await application?.close();
  await rm(home, { recursive: true, force: true });
});

describe("37-capability workspace vertical slice", () => {
  it("drives image-sequence playback, charts, and camera-backed annotation persistence", async () => {
    await expect.poll(() => page.getByLabel("Action chart").count()).toBe(1);
    await expect.poll(() => page.getByLabel("observation.images.front image").count()).toBe(1);
    const initial = await page.getByText(/\/ 1\.00s$/).first().textContent();
    await page.getByRole("button", { name: "Play", exact: true }).click();
    await page.waitForTimeout(180);
    expect(await page.getByText(/\/ 1\.00s$/).first().textContent()).not.toBe(initial);
    await page.getByRole("button", { name: "Pause", exact: true }).click();

    await page.locator("summary", { hasText: "Language annotations" }).click();
    await expect.poll(() => page.getByTestId("annotations-panel").count()).toBe(1);
    await expect.poll(() => page.locator(".annotation-overlay-source").first().getAttribute("src")).toMatch(/^data:image\/png;base64,/);
    await page.getByTestId("annotation-content").fill("Reach for the object");
    await page.getByTestId("annotation-add").click();
    await page.getByTestId("annotations-save").click();
    await expect.poll(() => page.getByText("Draft saved", { exact: true }).count()).toBe(1);
    await expect.poll(() => page.getByTestId("annotations-timeline").locator(".timeline-marker").count()).toBe(1);
  });

  it("opens quality, insights, replay-map, and local Doctor surfaces", async () => {
    await page.getByRole("button", { name: "quality", exact: true }).click();
    await expect.poll(() => page.getByRole("heading", { name: "First / last frames" }).count()).toBe(1);
    await page.getByRole("button", { name: /Episode 0/ }).click();
    await expect.poll(() => page.getByTestId("delete-preview").textContent()).toContain("episode_indices");

    await page.getByRole("button", { name: "insights", exact: true }).click();
    await expect.poll(() => page.getByLabel("Action variance heatmap").count()).toBe(1);
    await expect.poll(() => page.getByRole("heading", { name: "Action ↔ state temporal alignment" }).count()).toBe(1);

    await page.getByRole("button", { name: "replay", exact: true }).click();
    await expect.poll(() => page.getByLabel("so100 URDF pose").count()).toBe(1);
    await expect.poll(() => page.getByText(/End-effector trail:/).textContent()).toContain("1 second");

    await page.getByRole("button", { name: "doctor", exact: true }).click();
    await expect.poll(() => page.getByText(/Dataset structure: valid/).count()).toBe(1);
    await expect.poll(() => page.getByRole("button", { name: "Open LeRobot Doctor" }).isEnabled()).toBe(true);
    expect(await page.locator("[style]").count()).toBe(0);
  });
});

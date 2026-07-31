/// <reference lib="dom" />

import { chmod, mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import electronBinary from "electron";
import { _electron as electron, type ElectronApplication, type Page } from "playwright-core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const ROOT = resolve(import.meta.dirname, "../../..");
let application: ElectronApplication;
let page: Page;
let tempDir: string;
let sourceOne: string;
let sourceTwo: string;
let fakePython: string;
let fakeFfmpeg: string;

beforeAll(async () => {
  tempDir = await mkdtemp(join(tmpdir(), "datasetui-projects-e2e-"));
  sourceOne = join(tempDir, "dataset-one");
  sourceTwo = join(tempDir, "dataset-two");
  await mkdir(sourceOne);
  await mkdir(sourceTwo);

  fakePython = join(tempDir, "python3.12");
  const probe = JSON.stringify({
    python: { path: fakePython, version: "3.12.8", available: true },
    lerobot: { version: "0.6.0", available: true, import_error: null },
    torch: { version: "2.7.1+cu128", cuda_runtime: "12.8", cuda_available: false },
    secret_seen: false,
  });
  await writeFile(fakePython, `#!/bin/sh\nprintf '%s\\n' '${probe}'\n`, "utf8");
  await chmod(fakePython, 0o755);

  fakeFfmpeg = join(tempDir, "ffmpeg");
  await writeFile(
    fakeFfmpeg,
    "#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then echo 'ffmpeg version 7.1.1 static'; exit 0; fi\nif [ \"$1\" = \"-encoders\" ]; then echo ' V....D libx264 H.264'; exit 0; fi\nexit 2\n",
    "utf8",
  );
  await chmod(fakeFfmpeg, 0o755);

  const env = Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
  );
  env.HOME = tempDir;
  env.XDG_CONFIG_HOME = join(tempDir, "config");
  env.XDG_DATA_HOME = join(tempDir, "data");
  env.XDG_STATE_HOME = join(tempDir, "state");
  env.XDG_CACHE_HOME = join(tempDir, "cache");
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
  await rm(tempDir, { recursive: true, force: true });
});

async function registerProject(name: string, source: string, target: "v2.1" | "v3"): Promise<void> {
  await page.getByTestId("project-name").fill(name);
  await page.getByTestId("project-source").fill(source);
  await page.getByTestId("project-target").selectOption(target);
  await page.getByTestId("project-register").click();
}

describe("project registry and runtime management", () => {
  it("registers two explicit-target projects and switches one to a compatible external runtime", async () => {
    await registerProject("Assembly v3", sourceOne, "v3");
    await expect.poll(
      () => page.getByTestId("project-count").textContent(),
      { timeout: 10_000 },
    ).toBe("1");
    await registerProject("Assembly v21", sourceTwo, "v2.1");

    await expect.poll(
      () => page.getByTestId("project-count").textContent(),
      { timeout: 10_000 },
    ).toBe("2");
    await expect.poll(() => page.getByText("Assembly v3", { exact: true }).count()).toBe(1);
    await expect.poll(() => page.getByText("Assembly v21", { exact: true }).count()).toBe(1);

    await page.getByText("Assembly v3", { exact: true }).click();
    await page.getByTestId("runtime-mode").selectOption("external");
    await page.getByTestId("runtime-python").fill(fakePython);
    await page.getByTestId("runtime-ffmpeg").fill(fakeFfmpeg);
    await page.getByTestId("runtime-apply").click();

    await expect.poll(
      () => page.getByTestId("runtime-compatible").textContent(),
      { timeout: 10_000 },
    ).toBe("compatible");
    await expect.poll(
      () => page.getByTestId("selected-runtime").textContent(),
      { timeout: 10_000 },
    ).toBe("external");
  });
});

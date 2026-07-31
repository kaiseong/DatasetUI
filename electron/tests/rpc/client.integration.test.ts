import { resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { PythonBackend } from "../../src/main/python-backend.js";

const ROOT = resolve(import.meta.dirname, "../../..");
let backend: PythonBackend | undefined;

afterEach(async () => {
  await backend?.stop();
  backend = undefined;
});

describe("PythonBackend integration", () => {
  it("spawns without a shell and completes initialize, ping, report, shutdown", async () => {
    backend = new PythonBackend({
      appRoot: ROOT,
      pythonExecutable: resolve(ROOT, ".venv/bin/python"),
      requestTimeoutMs: 10_000,
    });
    const initialized = await backend.start();
    expect(initialized.protocolVersion).toBe(1);

    const ping = await backend.request<Record<string, unknown>>("system.ping");
    expect(ping).toMatchObject({ service: "lerobot-dataset-editor", protocolVersion: 1 });

    const report = await backend.request<Record<string, unknown>>("report.get");
    expect(report).toMatchObject({ operation: "report", status: "ok" });

    await backend.stop();
    expect(backend.exitCode).toBe(0);
  });
});

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { PythonBackend } from "../../src/main/python-backend.js";

const ROOT = resolve(import.meta.dirname, "../../..");
let backend: PythonBackend | undefined;
let tempDir: string;

beforeEach(async () => {
  tempDir = await mkdtemp(join(tmpdir(), "registry-test-"));
});

afterEach(async () => {
  await backend?.stop();
  backend = undefined;
  await rm(tempDir, { recursive: true, force: true });
});

describe("Project registry via PythonBackend RPC", () => {
  it("registers a project, lists it, and runs runtime.doctor", async () => {
    backend = new PythonBackend({
      appRoot: ROOT,
      pythonExecutable: resolve(ROOT, ".venv/bin/python"),
      requestTimeoutMs: 10_000,
      parentEnv: {
        PATH: "/usr/bin:/bin",
        HOME: tempDir,
        XDG_DATA_HOME: join(tempDir, "xdg_data"),
      },
    });
    const initialized = await backend.start();
    expect(initialized.protocolVersion).toBe(1);
    expect(initialized.methods).toContain("project.register");
    expect(initialized.methods).toContain("project.list");
    expect(initialized.methods).toContain("runtime.doctor");
    expect(initialized.methods).toContain("runtime.select");

    // Register a project
    const reg = await backend.request<{ id: string; created_at: string }>(
      "project.register",
      {
        name: "TestProject",
        source_path: tempDir,
        target_format: "v3",
        runtime_mode: "embedded",
      },
    );
    expect(reg.id).toHaveLength(36);
    expect(reg.created_at).toBeTruthy();

    // List projects
    const list = await backend.request<{ projects: Array<Record<string, unknown>>; total: number }>(
      "project.list",
      { limit: 10 },
    );
    expect(list.total).toBe(1);
    expect(list.projects[0]).toMatchObject({
      name: "TestProject",
      target_format: "v3",
      runtime_mode: "embedded",
    });

    // Runtime doctor
    const doctor = await backend.request<Record<string, unknown>>("runtime.doctor", {
      project_id: reg.id,
    });
    expect(doctor).toHaveProperty("python");
    expect(doctor).toHaveProperty("compute");
    expect((doctor.python as Record<string, unknown>).available).toBe(true);

    await backend.stop();
    expect(backend.exitCode).toBe(0);
  });
});

import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..");
const PACKAGE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
const ARTIFACT_NAME = `DatasetUI-${PACKAGE.version}-linux-x86_64.AppImage`;
const ARTIFACT = resolve(process.env.DATASETUI_APPIMAGE ?? join(ROOT, "release", ARTIFACT_NAME));
const EVIDENCE = join(ROOT, "appimage-spike", "evidence", "spike-result.json");
const REQUIRED_RESOURCES = [
  "resources/app.asar",
  "resources/python/src/lerobot_dataset_editor/rpc/__main__.py",
  "resources/contracts/rpc.schema.json",
  "resources/contracts/rpc-transport.schema.json",
  "resources/fixtures/v21_valid/meta/info.json",
  "resources/tests/fixtures/official/v21_v033/meta/info.json",
  "resources/third_party/visualize_dataset.REVISION",
  "resources/tools/lerobot_dataset_tools/cli.py",
  "resources/dataset_tools.sh",
];

function fail(message, result) {
  const details = result
    ? `\nstdout:\n${result.stdout ?? ""}\nstderr:\n${result.stderr ?? ""}`
    : "";
  throw new Error(`${message}${details}`);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    timeout: 120_000,
    ...options,
  });
  if (result.error) fail(`${basename(command)} failed: ${result.error.message}`, result);
  if (result.status !== 0) {
    fail(`${basename(command)} exited with ${String(result.status)}`, result);
  }
  return result;
}

function parseOsRelease() {
  const values = {};
  for (const line of readFileSync("/etc/os-release", "utf8").split("\n")) {
    const match = /^([A-Z_]+)=(.*)$/.exec(line);
    if (!match) continue;
    let value = match[2];
    if (value.startsWith('"') && value.endsWith('"')) value = value.slice(1, -1);
    values[match[1]] = value;
  }
  return values;
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function displayPythonPath(path) {
  const relativePath = relative(ROOT, path);
  return relativePath && !relativePath.startsWith("..") ? relativePath : path;
}

function main() {
  if (process.platform !== "linux" || process.arch !== "x64") {
    fail(`AppImage spike requires linux x64, got ${process.platform} ${process.arch}`);
  }
  const osRelease = parseOsRelease();
  if (osRelease.ID !== "ubuntu") {
    fail(`AppImage spike requires Ubuntu, got ${osRelease.ID ?? "unknown"}`);
  }
  if (!existsSync(ARTIFACT)) fail(`AppImage artifact not found: ${ARTIFACT}`);

  const python = resolve(
    process.env.LEROBOT_DATASET_EDITOR_PYTHON ?? join(ROOT, ".venv", "bin", "python"),
  );
  if (!existsSync(python)) {
    fail(`Host Python not found: ${python}; run uv sync or set LEROBOT_DATASET_EDITOR_PYTHON`);
  }
  const pythonResult = run(python, ["--version"]);
  const pythonMatch = /Python (3\.12(?:\.\d+)?)/.exec(
    `${pythonResult.stdout}\n${pythonResult.stderr}`,
  );
  if (!pythonMatch) fail("The spike requires host Python 3.12", pythonResult);

  const work = mkdtempSync(join(tmpdir(), "datasetui-appimage-spike-"));
  try {
    const extraction = run(ARTIFACT, ["--appimage-extract"], { cwd: work });
    const appDir = join(work, "squashfs-root");
    const resources = REQUIRED_RESOURCES.map((path) => ({
      path,
      present: existsSync(join(appDir, path)),
    }));
    const missing = resources.filter((item) => !item.present).map((item) => item.path);
    if (missing.length) fail(`Extracted AppImage is missing resources: ${missing.join(", ")}`);

    const smoke = run(join(appDir, "AppRun"), ["--rpc-smoke"], {
      cwd: appDir,
      env: {
        ...process.env,
        LEROBOT_DATASET_EDITOR_PYTHON: python,
      },
    });
    const markerMatch = /^DATASETUI_RPC_SMOKE=(\{.*\})$/m.exec(smoke.stdout);
    if (!markerMatch) fail("Extracted AppRun emitted no RPC smoke marker", smoke);
    const ping = JSON.parse(markerMatch[1]);
    if (ping.service !== "lerobot-dataset-editor" || ping.protocolVersion !== 1) {
      fail(`Unexpected RPC smoke response: ${JSON.stringify(ping)}`, smoke);
    }

    const evidence = {
      schema_version: 1,
      generated_at: new Date().toISOString(),
      status: "passed",
      host: {
        os: "linux",
        distribution: osRelease.ID,
        version: osRelease.VERSION_ID,
        architecture: "x86_64",
      },
      artifact: {
        name: basename(ARTIFACT),
        target: "AppImage",
        architecture: "x86_64",
        electron_version: PACKAGE.devDependencies.electron,
        size_bytes: statSync(ARTIFACT).size,
        sha256: sha256(ARTIFACT),
      },
      extraction: {
        method: "--appimage-extract",
        fuse_required: false,
        exit_code: extraction.status,
        resources,
      },
      rpc_smoke: {
        command: "AppRun --rpc-smoke",
        exit_code: smoke.status,
        marker: "DATASETUI_RPC_SMOKE",
        service: ping.service,
        protocol_version: ping.protocolVersion,
        timestamp: ping.timestamp,
      },
      python_runtime: {
        python_mode: "host-python-spike",
        executable: displayPythonPath(python),
        version: pythonMatch[1],
        dependency_source: "host-environment",
        self_contained: false,
        production_packaging_task: 22,
        limitation:
          "This feasibility artifact is not self-contained: it requires a compatible host Python 3.12 environment with DatasetUI runtime dependencies; production Python bundling is deferred to Task 22.",
      },
    };

    mkdirSync(dirname(EVIDENCE), { recursive: true });
    writeFileSync(EVIDENCE, `${JSON.stringify(evidence, null, 2)}\n`, "utf8");
    process.stdout.write(`AppImage spike passed: ${ARTIFACT_NAME}\n`);
    process.stdout.write(`Evidence: ${EVIDENCE}\n`);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
}

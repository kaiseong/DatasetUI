import { spawn, type ChildProcessWithoutNullStreams, type SpawnOptionsWithoutStdio } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { RpcClient, type InitializeResult } from "../rpc/client.js";

export interface PythonSpawnInput {
  appRoot: string;
  pythonExecutable: string;
  parentEnv?: NodeJS.ProcessEnv;
}

export interface PythonSpawnSpec {
  command: string;
  args: string[];
  options: SpawnOptionsWithoutStdio & {
    cwd: string;
    env: Record<string, string>;
    shell: false;
  };
}

export function buildPythonSpawnSpec(input: PythonSpawnInput): PythonSpawnSpec {
  const parent = input.parentEnv ?? process.env;
  const env: Record<string, string> = {};
  for (const name of [
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
  ] as const) {
    const value = parent[name];
    if (value !== undefined) env[name] = value;
  }
  env.PYTHONDONTWRITEBYTECODE = "1";
  env.PYTHONPATH = resolve(input.appRoot, "python/src");
  env.PYTHONUNBUFFERED = "1";
  return {
    command: input.pythonExecutable,
    args: ["-m", "lerobot_dataset_editor.rpc"],
    options: {
      cwd: input.appRoot,
      env,
      shell: false,
    },
  };
}

export interface PythonBackendOptions {
  appRoot: string;
  pythonExecutable?: string;
  requestTimeoutMs?: number;
  parentEnv?: NodeJS.ProcessEnv;
}

function defaultPython(appRoot: string): string {
  const venvPython = resolve(appRoot, ".venv/bin/python");
  return existsSync(venvPython) ? venvPython : "python3.12";
}

export class PythonBackend {
  private child: ChildProcessWithoutNullStreams | undefined;
  private client: RpcClient | undefined;
  private _exitCode: number | null = null;
  private stopping: Promise<void> | undefined;

  constructor(private readonly options: PythonBackendOptions) {}

  get exitCode(): number | null {
    return this._exitCode;
  }

  async start(): Promise<InitializeResult> {
    if (this.child) throw new Error("Python backend already started");
    const spec = buildPythonSpawnSpec({
      appRoot: this.options.appRoot,
      pythonExecutable: this.options.pythonExecutable ?? defaultPython(this.options.appRoot),
      ...(this.options.parentEnv ? { parentEnv: this.options.parentEnv } : {}),
    });
    const child = spawn(spec.command, spec.args, {
      ...spec.options,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    child.stderr.on("data", (chunk: Buffer) => {
      process.stderr.write(`[python-backend] ${chunk.toString("utf8")}`);
    });
    this.client = new RpcClient(child.stdout, child.stdin, {
      requestTimeoutMs: this.options.requestTimeoutMs ?? 30_000,
    });
    child.once("exit", (code, signal) => {
      this._exitCode = code;
      if (code !== 0) {
        this.client?.dispose(
          new Error(`Python backend exited with code ${String(code)} signal ${String(signal)}`),
        );
      }
    });
    child.once("error", (error) => this.client?.dispose(error));
    try {
      return await this.client.start({ name: "datasetui-electron", version: "0.2.0" });
    } catch (error) {
      if (child.exitCode === null) child.kill("SIGKILL");
      throw error;
    }
  }

  request<T>(method: string, params?: Record<string, unknown> | unknown[]): Promise<T> {
    if (!this.client) return Promise.reject(new Error("Python backend is not started"));
    return this.client.request<T>(method, params);
  }

  async stop(): Promise<void> {
    if (this.stopping) return this.stopping;
    this.stopping = this.stopOnce();
    return this.stopping;
  }

  private async stopOnce(): Promise<void> {
    const child = this.child;
    const client = this.client;
    if (!child) return;
    if (child.exitCode === null && client) {
      try {
        await client.request("shutdown");
      } catch {
        // A crashed backend is handled by the exit path below.
      }
    }
    if (child.exitCode === null) {
      await new Promise<void>((resolveExit) => {
        const timeout = setTimeout(() => {
          if (child.exitCode === null) child.kill("SIGKILL");
        }, 5_000);
        child.once("exit", () => {
          clearTimeout(timeout);
          resolveExit();
        });
      });
    }
    this._exitCode = child.exitCode;
    client?.dispose(new Error("Python backend stopped"));
  }
}

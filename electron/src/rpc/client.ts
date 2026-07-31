import type { Readable, Writable } from "node:stream";

import { FrameDecoder, encodeFrame, type RpcMessage } from "./framing.js";

export interface RpcClientOptions {
  requestTimeoutMs: number;
}

export interface RpcClientIdentity {
  name: string;
  version: string;
}

export interface InitializeResult {
  protocolVersion: number;
  methods: string[];
  server?: { name: string; version: string };
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timeout: NodeJS.Timeout;
}

export class RpcRemoteError extends Error {
  readonly code: number;
  readonly data: unknown;

  constructor(code: number, message: string, data?: unknown) {
    super(message);
    this.name = "RpcRemoteError";
    this.code = code;
    this.data = data;
  }
}

export class RpcClient {
  private readonly decoder = new FrameDecoder();
  private readonly pending = new Map<number, PendingRequest>();
  private readonly readyPromise: Promise<void>;
  private resolveReady!: () => void;
  private rejectReady!: (reason: Error) => void;
  private nextId = 1;
  private initialized = false;
  private closedError: Error | undefined;

  constructor(
    private readonly input: Readable,
    private readonly output: Writable,
    private readonly options: RpcClientOptions,
  ) {
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    input.on("data", (chunk: Buffer | string) => {
      this.onData(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
    });
    input.on("error", (error) => this.fail(error));
    input.on("end", () => this.fail(new Error("RPC transport ended")));
    input.on("close", () => this.fail(new Error("RPC transport closed")));
    output.on("error", (error) => this.fail(error));
  }

  async start(client: RpcClientIdentity): Promise<InitializeResult> {
    await this.readyPromise;
    const result = await this.sendRequest<InitializeResult>("initialize", {
      client,
      protocolVersion: 1,
    });
    if (result.protocolVersion !== 1) {
      throw new Error(`unsupported RPC protocol version: ${result.protocolVersion}`);
    }
    this.initialized = true;
    return result;
  }

  request<T>(method: string, params?: Record<string, unknown> | unknown[]): Promise<T> {
    if (!this.initialized) {
      return Promise.reject(new Error("RPC client is not initialized"));
    }
    return this.sendRequest<T>(method, params);
  }

  dispose(reason = new Error("RPC client disposed")): void {
    this.fail(reason);
  }

  private sendRequest<T>(
    method: string,
    params?: Record<string, unknown> | unknown[],
  ): Promise<T> {
    if (this.closedError) {
      return Promise.reject(this.closedError);
    }
    const id = this.nextId++;
    const message: RpcMessage = { jsonrpc: "2.0", id, method };
    if (params !== undefined) {
      message.params = params;
    }
    return new Promise<T>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`RPC request timed out after ${this.options.requestTimeoutMs}ms: ${method}`));
      }, this.options.requestTimeoutMs);
      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timeout,
      });
      this.output.write(encodeFrame(message), (error) => {
        if (error) {
          const pending = this.pending.get(id);
          if (pending) {
            clearTimeout(pending.timeout);
            this.pending.delete(id);
            pending.reject(error);
          }
        }
      });
    });
  }

  private onData(chunk: Buffer): void {
    if (this.closedError) return;
    let messages: RpcMessage[];
    try {
      messages = this.decoder.push(chunk);
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    for (const message of messages) {
      if (message.method === "server.ready" && message.id === undefined) {
        const params = message.params as { protocolVersion?: unknown } | undefined;
        if (params?.protocolVersion !== 1) {
          this.fail(new Error("backend announced an unsupported protocol version"));
        } else {
          this.resolveReady();
        }
        continue;
      }
      const id = message.id;
      if (typeof id !== "number") continue;
      const pending = this.pending.get(id);
      if (!pending) continue;
      clearTimeout(pending.timeout);
      this.pending.delete(id);
      if (message.error && typeof message.error === "object") {
        const remote = message.error as { code?: unknown; message?: unknown; data?: unknown };
        pending.reject(
          new RpcRemoteError(
            typeof remote.code === "number" ? remote.code : -32603,
            typeof remote.message === "string" ? remote.message : "Unknown RPC error",
            remote.data,
          ),
        );
      } else if (Object.hasOwn(message, "result")) {
        pending.resolve(message.result);
      } else {
        pending.reject(new Error("malformed JSON-RPC response"));
      }
    }
  }

  private fail(error: Error): void {
    if (this.closedError) return;
    this.closedError = error;
    this.rejectReady(error);
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pending.clear();
  }
}

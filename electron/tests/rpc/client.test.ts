import { PassThrough } from "node:stream";

import { describe, expect, it } from "vitest";

import { RpcClient, RpcRemoteError } from "../../src/rpc/client.js";
import { FrameDecoder, encodeFrame } from "../../src/rpc/framing.js";

function nextMessage(stream: PassThrough): Promise<Record<string, unknown>> {
  const decoder = new FrameDecoder();
  return new Promise((resolve, reject) => {
    const onData = (chunk: Buffer): void => {
      try {
        const messages = decoder.push(chunk);
        if (messages[0]) {
          stream.off("data", onData);
          resolve(messages[0]);
        }
      } catch (error) {
        reject(error);
      }
    };
    stream.on("data", onData);
  });
}

describe("RpcClient", () => {
  it("waits for server.ready, initializes, and correlates monotonic request ids", async () => {
    const backendOutput = new PassThrough();
    const backendInput = new PassThrough();
    const client = new RpcClient(backendOutput, backendInput, { requestTimeoutMs: 2_000 });

    const startPromise = client.start({ name: "vitest", version: "1" });
    let settled = false;
    void startPromise.then(() => {
      settled = true;
    });
    await new Promise((resolve) => setImmediate(resolve));
    expect(settled).toBe(false);

    const initializeRequestPromise = nextMessage(backendInput);
    backendOutput.write(
      encodeFrame({ jsonrpc: "2.0", method: "server.ready", params: { protocolVersion: 1 } }),
    );
    const initializeRequest = await initializeRequestPromise;
    expect(initializeRequest).toMatchObject({ jsonrpc: "2.0", id: 1, method: "initialize" });
    backendOutput.write(
      encodeFrame({
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: 1, methods: ["initialize", "system.ping"] },
      }),
    );
    await expect(startPromise).resolves.toMatchObject({ protocolVersion: 1 });

    const pingRequestPromise = nextMessage(backendInput);
    const pingPromise = client.request("system.ping");
    const pingRequest = await pingRequestPromise;
    expect(pingRequest).toMatchObject({ id: 2, method: "system.ping" });
    backendOutput.write(encodeFrame({ jsonrpc: "2.0", id: 2, result: { ok: true } }));
    await expect(pingPromise).resolves.toEqual({ ok: true });
  });

  it("rejects calls before initialization and surfaces remote JSON-RPC errors", async () => {
    const backendOutput = new PassThrough();
    const backendInput = new PassThrough();
    const client = new RpcClient(backendOutput, backendInput, { requestTimeoutMs: 2_000 });
    await expect(client.request("system.ping")).rejects.toThrow("not initialized");

    const start = client.start({ name: "vitest", version: "1" });
    const initializeRequest = nextMessage(backendInput);
    backendOutput.write(
      encodeFrame({ jsonrpc: "2.0", method: "server.ready", params: { protocolVersion: 1 } }),
    );
    await initializeRequest;
    backendOutput.write(
      encodeFrame({ jsonrpc: "2.0", id: 1, result: { protocolVersion: 1, methods: [] } }),
    );
    await start;

    const requestMessage = nextMessage(backendInput);
    const request = client.request("missing.method");
    await requestMessage;
    backendOutput.write(
      encodeFrame({
        jsonrpc: "2.0",
        id: 2,
        error: { code: -32601, message: "Method not found" },
      }),
    );
    await expect(request).rejects.toBeInstanceOf(RpcRemoteError);
  });

  it("rejects pending work when the transport closes", async () => {
    const backendOutput = new PassThrough();
    const backendInput = new PassThrough();
    const client = new RpcClient(backendOutput, backendInput, { requestTimeoutMs: 2_000 });
    const start = client.start({ name: "vitest", version: "1" });
    const initializeRequest = nextMessage(backendInput);
    backendOutput.write(
      encodeFrame({ jsonrpc: "2.0", method: "server.ready", params: { protocolVersion: 1 } }),
    );
    await initializeRequest;
    backendOutput.destroy(new Error("backend crashed"));
    await expect(start).rejects.toThrow("backend crashed");
  });
});

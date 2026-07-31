import { describe, expect, it } from "vitest";

import {
  FrameDecoder,
  FrameProtocolError,
  FrameTooLargeError,
  MAX_CONTENT_LENGTH,
  encodeFrame,
} from "../../src/rpc/framing.js";

describe("framed JSON-RPC transport", () => {
  it("uses UTF-8 byte length rather than JavaScript character count", () => {
    const message = { jsonrpc: "2.0", id: 1, method: "echo", params: { text: "로봇" } };
    const frame = encodeFrame(message);
    const separator = frame.indexOf("\r\n\r\n");
    const header = frame.subarray(0, separator).toString("ascii");
    const payload = frame.subarray(separator + 4);
    expect(header).toBe(`Content-Length: ${payload.byteLength}`);
    expect(JSON.parse(payload.toString("utf8"))).toEqual(message);
  });

  it("buffers fragmented frames and emits multiple complete messages", () => {
    const decoder = new FrameDecoder();
    const first = encodeFrame({ jsonrpc: "2.0", id: 1, result: "first" });
    const second = encodeFrame({ jsonrpc: "2.0", id: 2, result: "second" });
    expect(decoder.push(first.subarray(0, 7))).toEqual([]);
    expect(decoder.push(Buffer.concat([first.subarray(7), second]))).toEqual([
      { jsonrpc: "2.0", id: 1, result: "first" },
      { jsonrpc: "2.0", id: 2, result: "second" },
    ]);
  });

  it("accepts unknown headers but rejects malformed lengths", () => {
    const payload = Buffer.from('{"jsonrpc":"2.0","id":1,"result":true}');
    const frame = Buffer.concat([
      Buffer.from(`X-DatasetUI: 1\r\nContent-Length: ${payload.length}\r\n\r\n`),
      payload,
    ]);
    expect(new FrameDecoder().push(frame)).toEqual([
      { jsonrpc: "2.0", id: 1, result: true },
    ]);
    expect(() =>
      new FrameDecoder().push(Buffer.from("Content-Length: nope\r\n\r\n")),
    ).toThrow(FrameProtocolError);
  });

  it("rejects oversized declared payloads before buffering a body", () => {
    expect(() =>
      new FrameDecoder().push(
        Buffer.from(`Content-Length: ${MAX_CONTENT_LENGTH + 1}\r\n\r\n`),
      ),
    ).toThrow(FrameTooLargeError);
  });
});

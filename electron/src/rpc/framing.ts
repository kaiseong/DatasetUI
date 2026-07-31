export const MAX_CONTENT_LENGTH = 16 * 1024 * 1024;
const MAX_HEADER_LENGTH = 8 * 1024;
const DELIMITER = Buffer.from("\r\n\r\n", "ascii");

export class FrameProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FrameProtocolError";
  }
}

export class FrameTooLargeError extends FrameProtocolError {
  constructor(message: string) {
    super(message);
    this.name = "FrameTooLargeError";
  }
}

export type RpcMessage = Record<string, unknown>;

export function encodeFrame(message: RpcMessage): Buffer {
  const payload = Buffer.from(JSON.stringify(message), "utf8");
  if (payload.byteLength > MAX_CONTENT_LENGTH) {
    throw new FrameTooLargeError(
      `payload length ${payload.byteLength} exceeds maximum ${MAX_CONTENT_LENGTH}`,
    );
  }
  return Buffer.concat([
    Buffer.from(`Content-Length: ${payload.byteLength}\r\n\r\n`, "ascii"),
    payload,
  ]);
}

export class FrameDecoder {
  private buffer = Buffer.alloc(0);

  push(chunk: Buffer | Uint8Array): RpcMessage[] {
    if (chunk.byteLength > 0) {
      this.buffer = Buffer.concat([this.buffer, Buffer.from(chunk)]);
    }
    const messages: RpcMessage[] = [];
    while (true) {
      const delimiterIndex = this.buffer.indexOf(DELIMITER);
      if (delimiterIndex < 0) {
        if (this.buffer.byteLength > MAX_HEADER_LENGTH) {
          throw new FrameProtocolError(`frame headers exceed ${MAX_HEADER_LENGTH} bytes`);
        }
        break;
      }
      if (delimiterIndex > MAX_HEADER_LENGTH) {
        throw new FrameProtocolError(`frame headers exceed ${MAX_HEADER_LENGTH} bytes`);
      }
      const headerBlock = this.buffer.subarray(0, delimiterIndex).toString("ascii");
      const headers = new Map<string, string>();
      for (const line of headerBlock.split("\r\n")) {
        const separator = line.indexOf(":");
        if (separator <= 0) {
          throw new FrameProtocolError("malformed frame header");
        }
        const name = line.slice(0, separator).trim().toLowerCase();
        if (headers.has(name)) {
          throw new FrameProtocolError(`duplicate frame header: ${name}`);
        }
        headers.set(name, line.slice(separator + 1).trim());
      }
      const rawLength = headers.get("content-length");
      if (rawLength === undefined || !/^(0|[1-9][0-9]*)$/.test(rawLength)) {
        throw new FrameProtocolError("Content-Length must be a non-negative integer");
      }
      const contentLength = Number(rawLength);
      if (!Number.isSafeInteger(contentLength)) {
        throw new FrameProtocolError("Content-Length exceeds JavaScript safe integer range");
      }
      if (contentLength > MAX_CONTENT_LENGTH) {
        throw new FrameTooLargeError(
          `declared payload exceeds maximum ${MAX_CONTENT_LENGTH} bytes`,
        );
      }
      const bodyStart = delimiterIndex + DELIMITER.byteLength;
      const frameEnd = bodyStart + contentLength;
      if (this.buffer.byteLength < frameEnd) {
        break;
      }
      const body = this.buffer.subarray(bodyStart, frameEnd);
      this.buffer = this.buffer.subarray(frameEnd);
      let value: unknown;
      try {
        value = JSON.parse(body.toString("utf8"));
      } catch (error) {
        throw new FrameProtocolError(`frame body is not valid JSON: ${String(error)}`);
      }
      if (value === null || Array.isArray(value) || typeof value !== "object") {
        throw new FrameProtocolError("JSON-RPC frame body must be an object");
      }
      messages.push(value as RpcMessage);
    }
    return messages;
  }
}

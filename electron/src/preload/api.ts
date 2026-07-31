export interface DatasetEditorApi {
  ping(): Promise<unknown>;
  report(): Promise<unknown>;
}

export type Invoke = (channel: string) => Promise<unknown>;

export function createDatasetEditorApi(invoke: Invoke): DatasetEditorApi {
  return Object.freeze({
    ping: () => invoke("rpc:ping"),
    report: () => invoke("rpc:report"),
  });
}

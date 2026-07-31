import type { DatasetEditorApi } from "../../preload/api";

declare global {
  interface Window {
    datasetEditor: DatasetEditorApi;
  }
}

export {};

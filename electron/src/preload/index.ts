import { contextBridge, ipcRenderer } from "electron";

import { createDatasetEditorApi } from "./api.js";

const api = createDatasetEditorApi((channel) => ipcRenderer.invoke(channel));
contextBridge.exposeInMainWorld("datasetEditor", api);

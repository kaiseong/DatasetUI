import { contextBridge, ipcRenderer } from "electron";

import { createDatasetEditorApi } from "./api.js";

const api = createDatasetEditorApi((channel, params?) => ipcRenderer.invoke(channel, params));
contextBridge.exposeInMainWorld("datasetEditor", api);

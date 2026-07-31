import { app, BrowserWindow, ipcMain, session, type IpcMainInvokeEvent } from "electron";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { PythonBackend } from "./python-backend.js";
import { createSecureWebPreferences, installWebContentsGuards, isAllowedNavigation } from "./security.js";

const isE2e = process.argv.includes("--e2e");
const isRpcSmoke = process.argv.includes("--rpc-smoke");
if (isE2e || isRpcSmoke) app.disableHardwareAcceleration();

let backend: PythonBackend | undefined;
let mainWindow: BrowserWindow | undefined;
let applicationUrl = "";
let quitting = false;

function rootPath(): string {
  return app.isPackaged ? process.resourcesPath : resolve(import.meta.dirname, "../..");
}

function pythonExecutable(root: string): string {
  return process.env.LEROBOT_DATASET_EDITOR_PYTHON ??
    (app.isPackaged ? "python3.12" : join(root, ".venv/bin/python"));
}

function assertTrustedSender(event: IpcMainInvokeEvent): void {
  const senderUrl = event.senderFrame?.url ?? "";
  if (!isAllowedNavigation(senderUrl, applicationUrl)) {
    throw new Error("Blocked IPC request from an untrusted renderer");
  }
}

function registerIpcHandlers(activeBackend: PythonBackend): void {
  ipcMain.removeHandler("rpc:ping");
  ipcMain.removeHandler("rpc:report");
  ipcMain.handle("rpc:ping", (event) => {
    assertTrustedSender(event);
    return activeBackend.request("system.ping");
  });
  ipcMain.handle("rpc:report", (event) => {
    assertTrustedSender(event);
    return activeBackend.request("report.get");
  });
}

function createMainWindow(): BrowserWindow {
  const preload = join(import.meta.dirname, "../preload/index.cjs");
  const renderer = join(import.meta.dirname, "../renderer/index.html");
  applicationUrl = pathToFileURL(renderer).href;
  const window = new BrowserWindow({
    width: 1100,
    height: 760,
    minWidth: 800,
    minHeight: 600,
    show: false,
    autoHideMenuBar: true,
    title: "DatasetUI",
    backgroundColor: "#0d1117",
    webPreferences: createSecureWebPreferences(preload),
  });
  installWebContentsGuards(window.webContents, () => applicationUrl);
  window.once("ready-to-show", () => window.show());
  window.on("closed", () => {
    if (mainWindow === window) mainWindow = undefined;
  });
  void window.loadURL(applicationUrl);
  return window;
}

async function stopBackend(): Promise<void> {
  if (quitting) return;
  quitting = true;
  await backend?.stop();
}

async function bootstrap(): Promise<void> {
  const root = rootPath();
  backend = new PythonBackend({
    appRoot: root,
    pythonExecutable: pythonExecutable(root),
    requestTimeoutMs: 30_000,
  });
  await backend.start();

  if (isRpcSmoke) {
    const ping = await backend.request<Record<string, unknown>>("system.ping");
    process.stdout.write(`DATASETUI_RPC_SMOKE=${JSON.stringify(ping)}\n`);
    await stopBackend();
    app.quit();
    return;
  }

  registerIpcHandlers(backend);
  mainWindow = createMainWindow();
}

app.on("web-contents-created", (_event, contents) => {
  contents.on("will-attach-webview", (event) => event.preventDefault());
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (!mainWindow && backend) mainWindow = createMainWindow();
});

app.on("before-quit", (event) => {
  if (!quitting && backend) {
    event.preventDefault();
    void stopBackend().finally(() => app.quit());
  }
});

void app.whenReady().then(async () => {
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false);
  });
  session.defaultSession.setPermissionCheckHandler(() => false);
  try {
    await bootstrap();
  } catch (error) {
    console.error("DatasetUI startup failed", error);
    await stopBackend();
    app.exit(1);
  }
});

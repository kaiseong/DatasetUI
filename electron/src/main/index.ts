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
  ipcMain.removeHandler("rpc:project.register");
  ipcMain.removeHandler("rpc:project.list");
  ipcMain.removeHandler("rpc:project.get");
  ipcMain.removeHandler("rpc:project.update");
  ipcMain.removeHandler("rpc:project.remove");
  ipcMain.removeHandler("rpc:runtime.doctor");
  ipcMain.removeHandler("rpc:runtime.select");
  ipcMain.removeHandler("rpc:dataset.open");
  ipcMain.removeHandler("rpc:dataset.browse");
  ipcMain.removeHandler("rpc:dataset.validate");
  ipcMain.handle("rpc:ping", (event) => {
    assertTrustedSender(event);
    return activeBackend.request("system.ping");
  });
  ipcMain.handle("rpc:report", (event) => {
    assertTrustedSender(event);
    return activeBackend.request("report.get");
  });
  ipcMain.handle("rpc:project.register", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("project.register", params);
  });
  ipcMain.handle("rpc:project.list", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("project.list", params);
  });
  ipcMain.handle("rpc:project.get", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("project.get", params);
  });
  ipcMain.handle("rpc:project.update", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("project.update", params);
  });
  ipcMain.handle("rpc:project.remove", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("project.remove", params);
  });
  ipcMain.handle("rpc:runtime.doctor", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("runtime.doctor", params);
  });
  ipcMain.handle("rpc:runtime.select", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("runtime.select", params);
  });
  ipcMain.handle("rpc:dataset.open", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("dataset.open", params);
  });
  ipcMain.handle("rpc:dataset.browse", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("dataset.browse", params);
  });
  ipcMain.handle("rpc:dataset.validate", (event, params) => {
    assertTrustedSender(event);
    return activeBackend.request("dataset.validate", params);
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
    show: isE2e,
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

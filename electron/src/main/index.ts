import {
  app,
  BrowserWindow,
  ipcMain,
  net,
  protocol,
  session,
  shell,
  type IpcMainInvokeEvent,
} from "electron";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { PythonBackend } from "./python-backend.js";
import {
  createSecureWebPreferences,
  installWebContentsGuards,
  isAllowedExternalUrl,
  isAllowedNavigation,
  parseDatasetMediaUrl,
} from "./security.js";

protocol.registerSchemesAsPrivileged([
  {
    scheme: "datasetui-media",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
    },
  },
]);

const isE2e = process.argv.includes("--e2e");
const isRpcSmoke = process.argv.includes("--rpc-smoke");
const HUB_IMPORT_TIMEOUT_MS = 30 * 60 * 1000;
if (isE2e || isRpcSmoke) app.disableHardwareAcceleration();

let backend: PythonBackend | undefined;
let mainWindow: BrowserWindow | undefined;
let applicationUrl = "";
let quitting = false;
let mediaProtocolRegistered = false;

const RPC_CHANNELS = {
  "rpc:ping": "system.ping",
  "rpc:report": "report.get",
  "rpc:project.register": "project.register",
  "rpc:project.list": "project.list",
  "rpc:project.get": "project.get",
  "rpc:project.update": "project.update",
  "rpc:project.remove": "project.remove",
  "rpc:runtime.doctor": "runtime.doctor",
  "rpc:runtime.select": "runtime.select",
  "rpc:dataset.open": "dataset.open",
  "rpc:dataset.browse": "dataset.browse",
  "rpc:dataset.validate": "dataset.validate",
  "rpc:dataset.summary": "dataset.summary",
  "rpc:dataset.episode": "dataset.episode",
  "rpc:dataset.analytics": "dataset.analytics",
  "rpc:progress.read": "progress.read",
  "rpc:replay.map": "replay.map",
  "rpc:hub.search": "hub.search",
  "rpc:hub.info": "hub.info",
  "rpc:hub.import": "hub.import",
  "rpc:annotations.list": "annotations.list",
  "rpc:annotations.save": "annotations.save",
  "rpc:annotations.export": "annotations.export",
} as const;

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
  for (const [channel, method] of Object.entries(RPC_CHANNELS)) {
    ipcMain.removeHandler(channel);
    ipcMain.handle(channel, (event, params: Record<string, unknown> | unknown[] | undefined) => {
      assertTrustedSender(event);
      return activeBackend.request(
        method,
        params,
        method === "hub.import" ? HUB_IMPORT_TIMEOUT_MS : undefined,
      );
    });
  }
  ipcMain.removeHandler("app:open-external");
  ipcMain.handle("app:open-external", async (event, params: unknown) => {
    assertTrustedSender(event);
    const url =
      params && typeof params === "object" && typeof (params as { url?: unknown }).url === "string"
        ? (params as { url: string }).url
        : "";
    if (!isAllowedExternalUrl(url)) throw new Error("Blocked unsafe external URL");
    await shell.openExternal(url);
    return { opened: true };
  });
}

function registerMediaProtocol(activeBackend: PythonBackend): void {
  if (mediaProtocolRegistered) return;
  mediaProtocolRegistered = true;
  protocol.handle("datasetui-media", async (request) => {
    const parsed = parseDatasetMediaUrl(request.url);
    if (!parsed) return new Response("Not found", { status: 404 });
    try {
      const resolved = await activeBackend.request<{ path?: unknown }>("media.resolve", {
        project_id: parsed.projectId,
        relative_path: parsed.relativePath,
      });
      if (typeof resolved.path !== "string") {
        return new Response("Not found", { status: 404 });
      }
      return net.fetch(pathToFileURL(resolved.path).href, {
        bypassCustomProtocolHandlers: true,
      });
    } catch {
      return new Response("Not found", { status: 404 });
    }
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
  registerMediaProtocol(backend);
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

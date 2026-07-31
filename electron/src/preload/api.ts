export interface ProjectRegisterParams {
  name: string;
  source_path: string;
  target_format: "v2.1" | "v3";
  runtime_mode: "embedded" | "external";
  output_path?: string;
  selected_revision?: string;
  python_path?: string;
  ffmpeg_path?: string;
}

export interface ProjectUpdateParams {
  id: string;
  name?: string;
  source_path?: string;
  output_path?: string;
  target_format?: "v2.1" | "v3";
  runtime_mode?: "embedded" | "external";
  python_path?: string;
  ffmpeg_path?: string;
  selected_revision?: string;
}

export interface RuntimeDoctorParams {
  project_id?: string;
  runtime_mode?: "embedded" | "external";
  python_path?: string;
  ffmpeg_path?: string;
  device_policy?: "auto" | "cpu-only" | "gpu-only";
}

export interface RuntimeSelectParams {
  project_id: string;
  runtime_mode: "embedded" | "external";
  python_path?: string;
  ffmpeg_path?: string;
  device_policy?: "auto" | "cpu-only" | "gpu-only";
}

export interface DatasetEditorApi {
  ping(): Promise<unknown>;
  report(): Promise<unknown>;
  projectRegister(params: ProjectRegisterParams): Promise<unknown>;
  projectList(params?: { limit?: number }): Promise<unknown>;
  projectGet(params: { id: string }): Promise<unknown>;
  projectUpdate(params: ProjectUpdateParams): Promise<unknown>;
  projectRemove(params: { id: string }): Promise<unknown>;
  runtimeDoctor(params?: RuntimeDoctorParams): Promise<unknown>;
  runtimeSelect(params: RuntimeSelectParams): Promise<unknown>;
}

export type Invoke = (channel: string, params?: unknown) => Promise<unknown>;

export function createDatasetEditorApi(invoke: Invoke): DatasetEditorApi {
  return Object.freeze({
    ping: () => invoke("rpc:ping"),
    report: () => invoke("rpc:report"),
    projectRegister: (params: ProjectRegisterParams) => invoke("rpc:project.register", params),
    projectList: (params?: { limit?: number }) => invoke("rpc:project.list", params),
    projectGet: (params: { id: string }) => invoke("rpc:project.get", params),
    projectUpdate: (params: ProjectUpdateParams) => invoke("rpc:project.update", params),
    projectRemove: (params: { id: string }) => invoke("rpc:project.remove", params),
    runtimeDoctor: (params?: RuntimeDoctorParams) => invoke("rpc:runtime.doctor", params),
    runtimeSelect: (params: RuntimeSelectParams) => invoke("rpc:runtime.select", params),
  });
}

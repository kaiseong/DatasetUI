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

export interface DatasetOpenParams {
  project_id: string;
}

export interface DatasetBrowseParams {
  project_id: string;
  offset?: number;
  limit?: number;
}

export interface DatasetValidateParams {
  project_id: string;
}

export interface ProjectDatasetParams {
  project_id: string;
}

export interface DatasetEpisodeParams extends ProjectDatasetParams {
  episode_index: number;
}

export interface ProgressReadParams extends DatasetEpisodeParams {}

export interface ReplayMapParams extends ProjectDatasetParams {
  episode_index?: number;
  values: number[];
  joint_names?: string[];
  urdf_joints?: string[];
  joint_ranges?: Array<{ min: number; max: number }>;
}

export interface HubSearchParams {
  query: string;
  token?: string;
}

export interface HubInfoParams {
  repo_id: string;
  token?: string;
  revision?: string;
}

export interface HubImportParams extends HubInfoParams {
  destination: string;
}

export interface AnnotationsListParams extends DatasetEpisodeParams {}

export interface AnnotationsSaveParams extends DatasetEpisodeParams {
  atoms: unknown[];
}

export interface AnnotationsExportParams extends ProjectDatasetParams {
  output_path: string;
  episode_index?: number;
}

export interface ExternalOpenParams {
  url: string;
}

export interface MediaUrlParams extends ProjectDatasetParams {
  relative_path: string;
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
  datasetOpen(params: DatasetOpenParams): Promise<unknown>;
  datasetBrowse(params: DatasetBrowseParams): Promise<unknown>;
  datasetValidate(params: DatasetValidateParams): Promise<unknown>;
  datasetSummary(params: ProjectDatasetParams): Promise<unknown>;
  datasetEpisode(params: DatasetEpisodeParams): Promise<unknown>;
  datasetAnalytics(params: ProjectDatasetParams): Promise<unknown>;
  progressRead(params: ProgressReadParams): Promise<unknown>;
  replayMap(params: ReplayMapParams): Promise<unknown>;
  hubSearch(params: HubSearchParams): Promise<unknown>;
  hubInfo(params: HubInfoParams): Promise<unknown>;
  hubImport(params: HubImportParams): Promise<unknown>;
  annotationsList(params: AnnotationsListParams): Promise<unknown>;
  annotationsSave(params: AnnotationsSaveParams): Promise<unknown>;
  annotationsExport(params: AnnotationsExportParams): Promise<unknown>;
  openExternal(params: ExternalOpenParams): Promise<unknown>;
  mediaUrl(params: MediaUrlParams): string;
}

export type Invoke = (channel: string, params?: unknown) => Promise<unknown>;

export function buildMediaUrl(params: MediaUrlParams): string {
  if (!/^[A-Za-z0-9_-]{1,128}$/.test(params.project_id)) {
    throw new Error("Invalid project id");
  }
  const normalized = params.relative_path.replaceAll("\\", "/");
  const parts = normalized.split("/");
  if (
    normalized.startsWith("/") ||
    parts.length === 0 ||
    parts.some((part) => !part || part === "." || part === ".." || part.includes("\0"))
  ) {
    throw new Error("Invalid dataset media path");
  }
  return `datasetui-media://${params.project_id}/${parts.map(encodeURIComponent).join("/")}`;
}

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
    datasetOpen: (params: DatasetOpenParams) => invoke("rpc:dataset.open", params),
    datasetBrowse: (params: DatasetBrowseParams) => invoke("rpc:dataset.browse", params),
    datasetValidate: (params: DatasetValidateParams) => invoke("rpc:dataset.validate", params),
    datasetSummary: (params: ProjectDatasetParams) => invoke("rpc:dataset.summary", params),
    datasetEpisode: (params: DatasetEpisodeParams) => invoke("rpc:dataset.episode", params),
    datasetAnalytics: (params: ProjectDatasetParams) => invoke("rpc:dataset.analytics", params),
    progressRead: (params: ProgressReadParams) => invoke("rpc:progress.read", params),
    replayMap: (params: ReplayMapParams) => invoke("rpc:replay.map", params),
    hubSearch: (params: HubSearchParams) => invoke("rpc:hub.search", params),
    hubInfo: (params: HubInfoParams) => invoke("rpc:hub.info", params),
    hubImport: (params: HubImportParams) => invoke("rpc:hub.import", params),
    annotationsList: (params: AnnotationsListParams) => invoke("rpc:annotations.list", params),
    annotationsSave: (params: AnnotationsSaveParams) => invoke("rpc:annotations.save", params),
    annotationsExport: (params: AnnotationsExportParams) => invoke("rpc:annotations.export", params),
    openExternal: (params: ExternalOpenParams) => invoke("app:open-external", params),
    mediaUrl: (params: MediaUrlParams) => buildMediaUrl(params),
  });
}

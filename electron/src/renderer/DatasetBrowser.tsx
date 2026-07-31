import { useCallback, useEffect, useState } from "react";

interface ProjectSummary {
  id: string;
  name: string;
  target_format: "v2.1" | "v3";
}

interface DatasetDocument {
  source_path: string;
  version: { family: string; version: string; raw: string };
  features: Array<{
    name: string;
    dtype: string;
    shape: number[];
    names: string[] | null;
  }>;
  episodes: Array<{
    index: number;
    length: number;
    chunk_index: number;
    file_index: number;
    tasks: string[];
    frame_start: number;
    frame_end: number;
  }>;
  total_frames: number;
  fps: number;
  tasks: string[];
  has_annotations: boolean;
  annotation_styles: string[] | null;
  validation: { valid: boolean; errors: string[]; warnings: string[] };
}

function parseProjects(value: unknown): ProjectSummary[] {
  if (!value || typeof value !== "object") throw new Error("Invalid project list response");
  const projects = (value as { projects?: unknown }).projects;
  if (!Array.isArray(projects)) throw new Error("Invalid project list response");
  return projects.map((item) => {
    if (!item || typeof item !== "object") throw new Error("Invalid project record");
    const project = item as Partial<ProjectSummary>;
    if (
      typeof project.id !== "string" ||
      typeof project.name !== "string" ||
      (project.target_format !== "v2.1" && project.target_format !== "v3")
    ) {
      throw new Error("Invalid project record");
    }
    return project as ProjectSummary;
  });
}

function isDocument(value: unknown): value is DatasetDocument {
  if (!value || typeof value !== "object") return false;
  const document = value as Partial<DatasetDocument>;
  return (
    typeof document.source_path === "string" &&
    typeof document.version?.version === "string" &&
    Array.isArray(document.features) &&
    document.features.every((feature) =>
      typeof feature?.name === "string" &&
      typeof feature.dtype === "string" &&
      Array.isArray(feature.shape),
    ) &&
    Array.isArray(document.episodes) &&
    document.episodes.every((episode) =>
      typeof episode?.index === "number" &&
      typeof episode.length === "number" &&
      typeof episode.frame_start === "number" &&
      typeof episode.frame_end === "number",
    ) &&
    typeof document.total_frames === "number" &&
    typeof document.fps === "number" &&
    Array.isArray(document.tasks) &&
    typeof document.validation?.valid === "boolean"
  );
}

function DatasetCard({ projectId, label }: { projectId: string; label: "left" | "right" }): React.JSX.Element {
  const [document, setDocument] = useState<DatasetDocument>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!projectId) {
      setDocument(undefined);
      setError(undefined);
      return;
    }
    let active = true;
    setDocument(undefined);
    setError(undefined);
    void window.datasetEditor.datasetOpen({ project_id: projectId })
      .then((value) => {
        if (!isDocument(value)) throw new Error("Invalid dataset document response");
        if (active) setDocument(value);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { active = false; };
  }, [projectId]);

  return (
    <article className="dataset-card" data-testid={`dataset-card-${label}`}>
      <h3 className="dataset-card-title">{label === "left" ? "Left dataset" : "Right dataset"}</h3>
      {error ? <p className="dataset-card-error" data-testid={`dataset-error-${label}`}>{error}</p> : null}
      {!projectId ? <p className="dataset-card-status">Select a project.</p> : null}
      {projectId && !document && !error ? <p className="dataset-card-status">Loading metadata…</p> : null}
      {document ? (
        <div className="dataset-card-content">
          <div className="dataset-card-row"><span>Version</span><strong data-testid={`dataset-version-${label}`}>{document.version.version}</strong></div>
          <div className="dataset-card-row"><span>Episodes</span><strong data-testid={`dataset-episodes-${label}`}>{document.episodes.length}</strong></div>
          <div className="dataset-card-row"><span>Total frames</span><strong>{document.total_frames}</strong></div>
          <div className="dataset-card-row"><span>FPS</span><strong>{document.fps}</strong></div>
          <div className="dataset-card-row"><span>Features</span><strong>{document.features.length}</strong></div>
          <div className="dataset-card-row"><span>Tasks</span><strong>{document.tasks.length}</strong></div>
          {document.has_annotations ? (
            <div className="dataset-card-row"><span>Annotations</span><strong>{document.annotation_styles?.join(", ") ?? "yes"}</strong></div>
          ) : null}
          <section className="dataset-reference-block" aria-label={`${label} feature schema`}>
            <h4>Feature schema</h4>
            <ul className="dataset-reference-list" data-testid={`dataset-schema-${label}`}>
              {document.features.slice(0, 8).map((feature) => (
                <li key={feature.name}>
                  <code>{feature.name}</code>
                  <span>{feature.dtype}{feature.shape.length ? ` [${feature.shape.join(" × ")}]` : ""}</span>
                </li>
              ))}
            </ul>
            {document.features.length > 8 ? (
              <p className="dataset-reference-more">+ {document.features.length - 8} more features</p>
            ) : null}
          </section>
          <section className="dataset-reference-block" aria-label={`${label} episode references`}>
            <h4>Episode references</h4>
            <ol className="dataset-reference-list dataset-episode-list">
              {document.episodes.slice(0, 5).map((episode) => (
                <li key={episode.index} data-testid={`dataset-episode-${label}-${episode.index}`}>
                  <code>#{episode.index}</code>
                  <span>Frames {episode.frame_start}–{episode.frame_end}</span>
                  <small>chunk {episode.chunk_index} · file {episode.file_index}</small>
                </li>
              ))}
            </ol>
            {document.episodes.length > 5 ? (
              <p className="dataset-reference-more">Showing 5 of {document.episodes.length}</p>
            ) : null}
          </section>
          <div className="dataset-card-row">
            <span>Metadata status</span>
            <strong className={document.validation.valid ? "badge-valid" : "badge-invalid"}>
              {document.validation.valid ? "ready" : `${document.validation.errors.length} error(s)`}
            </strong>
          </div>
        </div>
      ) : null}
    </article>
  );
}

export function DatasetBrowser(): React.JSX.Element {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [leftProjectId, setLeftProjectId] = useState("");
  const [rightProjectId, setRightProjectId] = useState("");
  const [error, setError] = useState<string>();

  const refresh = useCallback(async (): Promise<void> => {
    try {
      setProjects(parseProjects(await window.datasetEditor.projectList({ limit: 100 })));
      setError(undefined);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  return (
    <section className="dataset-browser" data-testid="dataset-browser" aria-labelledby="dataset-browser-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Read-only metadata</p>
          <h2 id="dataset-browser-heading">Dataset browser</h2>
        </div>
        <button data-testid="dataset-refresh" type="button" onClick={() => void refresh()}>Refresh projects</button>
      </div>
      <div className="dataset-browser-toolbar">
        <label>
          Left project
          <select data-testid="dataset-left-select" value={leftProjectId} onChange={(event) => setLeftProjectId(event.target.value)}>
            <option value="">Select…</option>
            {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
          </select>
        </label>
        <label>
          Right project
          <select data-testid="dataset-right-select" value={rightProjectId} onChange={(event) => setRightProjectId(event.target.value)}>
            <option value="">Select…</option>
            {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
          </select>
        </label>
      </div>
      {error ? <p className="dataset-card-error" role="alert">{error}</p> : null}
      <div className="dataset-browser-grid">
        <DatasetCard projectId={leftProjectId} label="left" />
        <DatasetCard projectId={rightProjectId} label="right" />
      </div>
    </section>
  );
}

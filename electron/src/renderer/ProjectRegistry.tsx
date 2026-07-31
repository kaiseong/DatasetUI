import { type FormEvent, useCallback, useEffect, useState } from "react";

interface Project {
  id: string;
  name: string;
  source_path: string;
  target_format: "v2.1" | "v3";
  runtime_mode: "embedded" | "external";
  python_path: string | null;
  ffmpeg_path: string | null;
  path_health?: { status: "ok" | "missing" | "read_only" };
}

function isProject(value: unknown): value is Project {
  if (!value || typeof value !== "object") return false;
  const project = value as Partial<Project>;
  return (
    typeof project.id === "string" &&
    typeof project.name === "string" &&
    typeof project.source_path === "string" &&
    (project.target_format === "v2.1" || project.target_format === "v3") &&
    (project.runtime_mode === "embedded" || project.runtime_mode === "external")
  );
}

function projectsFrom(value: unknown): Project[] {
  if (!value || typeof value !== "object") throw new Error("Invalid project list response");
  const projects = (value as { projects?: unknown }).projects;
  if (!Array.isArray(projects) || !projects.every(isProject)) {
    throw new Error("Invalid project list response");
  }
  return projects;
}

function doctorCompatible(value: unknown): boolean {
  return Boolean(
    value && typeof value === "object" && (value as { compatible?: unknown }).compatible === true,
  );
}

export function ProjectRegistry(): React.JSX.Element {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selected, setSelected] = useState<Project>();
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState<"v2.1" | "v3">("v3");
  const [runtimeMode, setRuntimeMode] = useState<"embedded" | "external">("embedded");
  const [pythonPath, setPythonPath] = useState("");
  const [ffmpegPath, setFfmpegPath] = useState("");
  const [compatibility, setCompatibility] = useState<"unknown" | "compatible" | "incompatible">("unknown");
  const [registering, setRegistering] = useState(false);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [error, setError] = useState<string>();

  const refresh = useCallback(async (): Promise<Project[]> => {
    const next = projectsFrom(await window.datasetEditor.projectList({ limit: 100 }));
    setProjects(next);
    return next;
  }, []);

  useEffect(() => {
    void refresh().catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  }, [refresh]);

  async function choose(project: Project): Promise<void> {
    setError(undefined);
    try {
      const opened = await window.datasetEditor.projectGet({ id: project.id });
      if (!isProject(opened)) throw new Error("Invalid project response");
      setSelected(opened);
      setRuntimeMode(opened.runtime_mode);
      setPythonPath(opened.python_path ?? "");
      setFfmpegPath(opened.ffmpeg_path ?? "");
      setCompatibility("unknown");
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function register(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (registering) return;
    setRegistering(true);
    setError(undefined);
    try {
      await window.datasetEditor.projectRegister({
        name,
        source_path: source,
        target_format: target,
        runtime_mode: "embedded",
      });
      setName("");
      setSource("");
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRegistering(false);
    }
  }

  async function applyRuntime(): Promise<void> {
    if (!selected || runtimeBusy) return;
    setRuntimeBusy(true);
    setError(undefined);
    setCompatibility("unknown");
    try {
      const params = runtimeMode === "external"
        ? {
            project_id: selected.id,
            runtime_mode: runtimeMode,
            python_path: pythonPath,
            ffmpeg_path: ffmpegPath,
            device_policy: "auto" as const,
          }
        : {
            project_id: selected.id,
            runtime_mode: runtimeMode,
            device_policy: "auto" as const,
          };
      const updated = await window.datasetEditor.runtimeSelect(params);
      if (!isProject(updated)) throw new Error("Invalid runtime selection response");
      const doctor = await window.datasetEditor.runtimeDoctor({
        project_id: selected.id,
        device_policy: "auto",
      });
      setCompatibility(doctorCompatible(doctor) ? "compatible" : "incompatible");
      setSelected(updated);
      setProjects((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (reason) {
      setCompatibility("incompatible");
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRuntimeBusy(false);
    }
  }

  return (
    <section className="project-panel" aria-labelledby="projects-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Local projects</p>
          <h2 id="projects-heading">Project registry</h2>
        </div>
        <strong data-testid="project-count">{projects.length}</strong>
      </div>

      <form className="project-form" aria-busy={registering} onSubmit={(event) => void register(event)}>
        <label>
          Name
          <input data-testid="project-name" disabled={registering} value={name} onChange={(event) => setName(event.target.value)} required />
        </label>
        <label>
          Source dataset
          <input data-testid="project-source" disabled={registering} value={source} onChange={(event) => setSource(event.target.value)} required />
        </label>
        <label>
          Preferred output
          <select data-testid="project-target" disabled={registering} value={target} onChange={(event) => setTarget(event.target.value as "v2.1" | "v3")}>
            <option value="v2.1">LeRobot v2.1</option>
            <option value="v3">LeRobot v3</option>
          </select>
        </label>
        <button data-testid="project-register" disabled={registering} type="submit">
          {registering ? "Registering…" : "Register project"}
        </button>
      </form>

      <div className="project-grid">
        <ul className="project-list" aria-label="Recent projects">
          {projects.map((project) => (
            <li key={project.id}>
              <button type="button" onClick={() => void choose(project)} aria-pressed={selected?.id === project.id}>
                <span>{project.name}</span>
                <small>{project.target_format} · {project.path_health?.status ?? "unknown"}</small>
              </button>
            </li>
          ))}
        </ul>

        <div className="runtime-card" aria-busy={runtimeBusy}>
          <h3>Runtime</h3>
          {selected ? (
            <>
              <p>{selected.name}: <strong data-testid="selected-runtime">{selected.runtime_mode}</strong></p>
              <label>
                Mode
                <select data-testid="runtime-mode" disabled={runtimeBusy} value={runtimeMode} onChange={(event) => setRuntimeMode(event.target.value as "embedded" | "external")}>
                  <option value="embedded">Embedded</option>
                  <option value="external">External</option>
                </select>
              </label>
              {runtimeMode === "external" ? (
                <>
                  <label>
                    Python 3.12
                    <input data-testid="runtime-python" disabled={runtimeBusy} value={pythonPath} onChange={(event) => setPythonPath(event.target.value)} required />
                  </label>
                  <label>
                    FFmpeg
                    <input data-testid="runtime-ffmpeg" disabled={runtimeBusy} value={ffmpegPath} onChange={(event) => setFfmpegPath(event.target.value)} required />
                  </label>
                </>
              ) : null}
              <button data-testid="runtime-apply" disabled={runtimeBusy} type="button" onClick={() => void applyRuntime()}>
                {runtimeBusy ? "Validating…" : "Validate and switch"}
              </button>
              <output data-testid="runtime-compatible">{compatibility}</output>
            </>
          ) : <p>Select a project to configure its runtime.</p>}
        </div>
      </div>
      {error ? <p className="error" role="alert">{error}</p> : null}
    </section>
  );
}

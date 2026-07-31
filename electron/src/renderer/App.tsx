import { useEffect, useState } from "react";

import { DatasetBrowser } from "./DatasetBrowser";
import { ProjectRegistry } from "./ProjectRegistry";

interface Report {
  status: string;
  dataset_editor_version: string;
  space_parity: { total: number };
  errors: string[];
}

function isReport(value: unknown): value is Report {
  if (!value || typeof value !== "object") return false;
  const report = value as Partial<Report>;
  return (
    typeof report.status === "string" &&
    typeof report.dataset_editor_version === "string" &&
    typeof report.space_parity?.total === "number" &&
    Array.isArray(report.errors)
  );
}

export function App(): React.JSX.Element {
  const [report, setReport] = useState<Report>();
  const [error, setError] = useState<string>();
  const [evalBlocked] = useState(() => {
    try {
      window.eval("globalThis.__datasetUiRendererEvalExecuted = true");
      return false;
    } catch {
      return true;
    }
  });

  useEffect(() => {
    let active = true;
    void window.datasetEditor
      .report()
      .then((value) => {
        if (!isReport(value)) throw new Error("Backend returned an invalid report");
        if (active) setReport(value);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <main>
      <header>
        <p className="eyebrow">LeRobot Dataset Editor</p>
        <h1>DatasetUI</h1>
        <p className="subtitle">Secure local projects and runtime environments</p>
      </header>
      <section className="status-card" aria-live="polite">
        <div>
          <span>Backend status</span>
          <strong data-testid="backend-status">{error ? "error" : (report?.status ?? "loading")}</strong>
        </div>
        <div>
          <span>Editor version</span>
          <strong data-testid="dataset-editor-version">{report?.dataset_editor_version ?? "—"}</strong>
        </div>
        <div>
          <span>Space capabilities mapped</span>
          <strong data-testid="space-count">{report?.space_parity.total ?? "—"}</strong>
        </div>
      </section>
      {error ? <p className="error">{error}</p> : null}
      <ProjectRegistry />
      <DatasetBrowser />
      <output data-testid="csp-eval-blocked" hidden>{String(evalBlocked)}</output>
    </main>
  );
}

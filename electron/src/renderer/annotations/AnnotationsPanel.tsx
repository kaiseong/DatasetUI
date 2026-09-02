import { useEffect, useMemo, useState } from "react";

import { createAnnotationAdapter, type AnnotationStorageAdapter } from "./adapter";
import { AnnotationOverlayCanvas } from "./AnnotationOverlayCanvas";
import { AnnotationsTimeline } from "./AnnotationsTimeline";
import { buildInterjection, buildPersistent, buildSpeech, buildVqaPair } from "./constructors";
import type { NormalizedDraw } from "./geometry";
import { clampAtom, envelope, snapToTimestamp } from "./model";
import { isSpeech, parseVqa, type AnnotationStyle, type EditableAnnotationAtom, type VqaAnswer, type VqaKind } from "./types";
import "./annotations.css";

export interface AnnotationsPanelProps {
  projectId: string;
  episodeIndex: number;
  timestamps: readonly number[];
  cameras: readonly string[];
  media?: ReadonlyArray<{ camera: string; url: string; kind: "video" | "image" | "depth"; from?: number }>;
  currentTime: number;
  duration: number;
  onSeek(time: number): void;
  onPause?(): void;
  isPlaying?: boolean;
  adapter?: AnnotationStorageAdapter;
}

type QuickKind = "task_aug" | "subtask" | "plan" | "memory" | "interjection" | "speech" | VqaKind;

export function AnnotationsPanel({
  projectId,
  episodeIndex,
  timestamps,
  cameras,
  media = [],
  currentTime,
  duration,
  onSeek,
  onPause,
  isPlaying = false,
  adapter: suppliedAdapter,
}: AnnotationsPanelProps): React.JSX.Element {
  const adapter = useMemo(() => suppliedAdapter ?? createAnnotationAdapter(), [suppliedAdapter]);
  const [atoms, setAtoms] = useState<EditableAnnotationAtom[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [kind, setKind] = useState<QuickKind>("subtask");
  const [text, setText] = useState("");
  const [camera, setCamera] = useState(cameras[0] ?? "");
  const [question, setQuestion] = useState("What is shown?");
  const [detailA, setDetailA] = useState("");
  const [detailB, setDetailB] = useState("");
  const [message, setMessage] = useState("Loading annotations…");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setMessage("Loading annotations…");
    void adapter.load(projectId, episodeIndex).then((loaded) => {
      if (!active) return;
      setAtoms(loaded.map((atom) => clampAtom(atom, duration, timestamps)));
      setMessage(loaded.length ? `${loaded.length} annotations loaded` : "No annotations yet");
    }).catch((reason: unknown) => {
      if (active) {
        setError(reason instanceof Error ? reason.message : String(reason));
        setMessage("Unable to load annotations");
      }
    });
    return () => { active = false; };
  }, [adapter, duration, episodeIndex, projectId, timestamps]);

  useEffect(() => {
    if (!cameras.includes(camera)) setCamera(cameras[0] ?? "");
  }, [camera, cameras]);

  const selected = atoms.find((atom) => atom.id === selectedId) ?? null;
  const selectedMedia = media.find((track) => track.camera === camera);
  const eventTime = snapToTimestamp(timestamps, Math.max(0, Math.min(duration, currentTime)));

  function append(newAtoms: ReturnType<typeof envelope>[]): void {
    setAtoms((current) => [...current, ...newAtoms]);
    setSelectedId(newAtoms.at(-1)?.id ?? null);
    setError(null);
  }

  function addQuick(): void {
    try {
      if (["task_aug", "subtask", "plan", "memory"].includes(kind)) {
        append([envelope(buildPersistent(kind as "task_aug" | "subtask" | "plan" | "memory", text, eventTime))]);
      } else if (kind === "interjection") {
        append([envelope(buildInterjection(text, eventTime))]);
      } else if (kind === "speech") {
        append([envelope(buildSpeech(text, eventTime))]);
      } else if (isVqa(kind)) {
        append(buildVqaPair(kind, question, buildSimpleVqa(kind, text, detailA, detailB), eventTime, camera).map((atom) => envelope(atom)));
      } else {
        throw new Error("Unsupported annotation type");
      }
      setText("");
      setMessage("Unsaved changes");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function save(): Promise<void> {
    try {
      const location = await adapter.save(projectId, episodeIndex, atoms);
      setMessage(location === "backend" ? "Draft saved" : "Backend offline — draft saved for this session");
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function exportDataset(): Promise<void> {
    try {
      await save();
      const suggested = `${projectId}-annotated`;
      const outputPath = window.prompt("Export directory (must not already exist)", suggested)?.trim();
      if (!outputPath) {
        setMessage("Export cancelled");
        return;
      }
      await adapter.exportDataset(projectId, outputPath);
      setMessage("Annotated dataset exported without modifying the source");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function onDraw(draw: NormalizedDraw): void {
    try {
      const label = text.trim() || "object";
      const answer: VqaAnswer = draw.kind === "bbox"
        ? { detections: [{ label, bbox_format: "xyxy", bbox: draw.bbox }] }
        : { label, point_format: "xy", point: draw.point };
      append(buildVqaPair(draw.kind, question, answer, eventTime, camera).map((atom) => envelope(atom)));
      setKind(draw.kind);
      setMessage("Visual annotation added; save the draft when ready");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  return (
    <section className="annotations-panel" aria-labelledby="annotations-title" data-testid="annotations-panel">
      <header className="annotations-header">
        <div><h2 id="annotations-title">Language annotations</h2><p>Episode {episodeIndex} · v3.1 canonical atoms</p></div>
        <div className="annotation-actions">
          <button type="button" data-testid="annotations-save" onClick={() => void save()}>Save draft</button>
          <button type="button" data-testid="annotations-export" onClick={() => void exportDataset()}>Export copy</button>
        </div>
      </header>
      <p className="annotation-status" aria-live="polite">{message}</p>
      {error ? <p className="annotation-error" role="alert">{error}</p> : null}

      <AnnotationsTimeline atoms={atoms} selectedId={selectedId} duration={duration} timestamps={timestamps}
        onSelect={setSelectedId} onSeek={onSeek} {...(onPause ? { onPause } : {})}
        onMove={(id, start, end) => {
          try {
            setAtoms((current) => {
              const selectedAtom = current.find((atom) => atom.id === id);
              if (!selectedAtom) return current;
              if (end === undefined) {
                return current.map((atom) => atom.id === id
                  ? clampAtom({ ...atom, timestamp: start }, duration, timestamps)
                  : atom);
              }
              if (selectedAtom.style !== "subtask" && selectedAtom.style !== "plan") throw new Error("Only subtask and plan spans are resizable");
              const next = current
                .filter((atom) => atom.style === selectedAtom.style && atom.timestamp > selectedAtom.timestamp)
                .sort((a, b) => a.timestamp - b.timestamp)[0];
              if (!next) throw new Error("The final subtask ends with the episode");
              const movedStart = clampAtom({ ...selectedAtom, timestamp: start }, duration, timestamps);
              const movedBoundary = clampAtom({ ...next, timestamp: end }, duration, timestamps);
              if (movedBoundary.timestamp < movedStart.timestamp) throw new Error("Annotation range is invalid");
              return current.map((atom) => atom.id === selectedAtom.id ? movedStart : atom.id === next.id ? movedBoundary : atom);
            });
            setMessage("Unsaved changes");
          } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
        }}
        onCreateRange={(start, end) => {
          // Canonical persistent annotations store activation timestamps. The
          // span continues until the next subtask activation or episode end.
          if (!(end > start)) { setError("Annotation range is invalid"); return; }
          append([
            envelope(buildPersistent("subtask", "New subtask", snapToTimestamp(timestamps, start))),
            envelope(buildPersistent("subtask", "Next subtask", snapToTimestamp(timestamps, end))),
          ]);
          setMessage("Unsaved changes");
        }} />

      <div className="annotation-workspace">
        <div className="annotation-create">
          <h3>Create annotation</h3>
          <label>Type<select aria-label="Annotation type" data-testid="annotation-kind" value={kind} onChange={(event) => setKind(event.target.value as QuickKind)}>
            <option value="task_aug">Task augmentation</option><option value="subtask">Subtask</option><option value="plan">Plan</option>
            <option value="memory">Memory</option><option value="interjection">Interjection</option><option value="speech">Speech tool call</option>
            <option value="bbox">VQA bbox</option><option value="keypoint">VQA keypoint</option><option value="count">VQA count</option>
            <option value="attribute">VQA attribute</option><option value="spatial">VQA spatial</option>
          </select></label>
          {isVqa(kind) ? <>
            <label>Camera<select aria-label="VQA camera" value={camera} onChange={(event) => setCamera(event.target.value)}>{cameras.map((key) => <option key={key}>{key}</option>)}</select></label>
            <label>Question<input aria-label="VQA question" value={question} onChange={(event) => setQuestion(event.target.value)} /></label>
          </> : null}
          <label>{kind === "speech" ? "Speech text" : kind === "count" ? "Object label" : kind === "spatial" ? "Subject" : "Label / content"}
            <input aria-label="Annotation content" data-testid="annotation-content" value={text} onChange={(event) => setText(event.target.value)} />
          </label>
          {kind === "count" ? <label>Count<input aria-label="Count" type="number" min="0" value={detailA} onChange={(event) => setDetailA(event.target.value)} /></label> : null}
          {kind === "attribute" ? <><label>Attribute<input aria-label="Attribute" value={detailA} onChange={(event) => setDetailA(event.target.value)} /></label><label>Value<input aria-label="Attribute value" value={detailB} onChange={(event) => setDetailB(event.target.value)} /></label></> : null}
          {kind === "spatial" ? <><label>Relation<input aria-label="Spatial relation" value={detailA} onChange={(event) => setDetailA(event.target.value)} /></label><label>Object<input aria-label="Spatial object" value={detailB} onChange={(event) => setDetailB(event.target.value)} /></label></> : null}
          <button type="button" data-testid="annotation-add" disabled={isVqa(kind) && !camera} onClick={addQuick}>Add at {eventTime.toFixed(3)}s</button>
          <p className="annotation-hint">Draw on the canvas for bbox/keypoint. A drag over 4 px makes a box; a click makes a point.</p>
        </div>

        <div className="annotation-visual">
          {cameras.length ? <AnnotationOverlayCanvas camera={camera} atoms={atoms} currentTime={currentTime} selectedId={selectedId}
            paused={!isPlaying} drawEnabled={Boolean(selectedMedia)} onDraw={onDraw}
            {...(selectedMedia ? { sourceUrl: selectedMedia.url, sourceKind: selectedMedia.kind, sourceTimeOffset: selectedMedia.from ?? 0 } : {})}
            onMoveGeometry={(id, answer) => {
              setAtoms((current) => current.map((atom) => atom.id === id ? { ...atom, content: JSON.stringify(answer) } : atom));
              setMessage("Visual annotation moved; save the draft when ready");
            }} /> : <p className="annotation-empty">No camera streams are available for visual grounding.</p>}
        </div>

        <div className="annotation-list" aria-label="Annotation list">
          <h3>Atoms ({atoms.length})</h3>
          {atoms.length === 0 ? <p className="annotation-empty">Create the first persistent or event annotation.</p> : atoms.map((atom) => (
            <article key={atom.id} className={selectedId === atom.id ? "selected" : ""}>
              <button type="button" onClick={() => { setSelectedId(atom.id); onPause?.(); onSeek(atom.timestamp); }}>
                <strong>{atom.style ?? "speech"}</strong><span>{atom.timestamp.toFixed(3)}s · {atom.role}</span>
                <small>{isSpeech(atom) ? String(atom.tool_calls?.[0]?.function.arguments.text ?? "") : atom.style === "vqa" && atom.role === "assistant" ? summarizeVqa(atom.content) : atom.content}</small>
              </button>
              <button type="button" className="delete" aria-label={`Delete ${atom.style ?? "speech"} annotation`} onClick={() => {
                setAtoms((current) => current.filter((candidate) => candidate.id !== atom.id));
                if (selectedId === atom.id) setSelectedId(null);
                setMessage("Unsaved changes");
              }}>Delete</button>
            </article>
          ))}
          {selected && selected.style === "vqa" && selected.role === "assistant"
            ? <SelectedVqaEditor atom={selected} onChange={(answer) => {
                setAtoms((current) => current.map((atom) => atom.id === selected.id ? { ...atom, content: JSON.stringify(answer) } : atom));
                setMessage("Unsaved changes");
              }} />
            : selected ? <label>Edit selected content<input aria-label="Edit selected annotation content" value={selected.content ?? ""} disabled={isSpeech(selected)}
                onChange={(event) => setAtoms((current) => current.map((atom) => atom.id === selected.id ? { ...atom, content: event.target.value } : atom))} /></label> : null}
        </div>
      </div>
    </section>
  );
}

function isVqa(kind: QuickKind): kind is VqaKind { return ["bbox", "keypoint", "count", "attribute", "spatial"].includes(kind); }

function buildSimpleVqa(kind: VqaKind, label: string, a: string, b: string): VqaAnswer {
  if (kind === "count") return { label: label.trim(), count: Number(a) };
  if (kind === "attribute") return { label: label.trim(), attribute: a.trim(), value: b.trim() };
  if (kind === "spatial") return { subject: label.trim(), relation: a.trim(), object: b.trim() };
  if (kind === "bbox") return { detections: [{ label: label.trim(), bbox_format: "xyxy", bbox: [0.25, 0.25, 0.75, 0.75] }] };
  return { label: label.trim(), point_format: "xy", point: [0.5, 0.5] };
}

function summarizeVqa(content: string | null): string {
  const answer = parseVqa(content);
  if (!answer) return "Invalid VQA answer";
  if ("detections" in answer) return `${answer.detections.length} detection(s)`;
  if ("point" in answer) return `${answer.label} keypoint`;
  if ("count" in answer) return `${answer.label}: ${answer.count}`;
  if ("attribute" in answer) return `${answer.label} ${answer.attribute}: ${answer.value}`;
  return `${answer.subject} ${answer.relation} ${answer.object}`;
}

function SelectedVqaEditor({ atom, onChange }: { atom: EditableAnnotationAtom; onChange(answer: VqaAnswer): void }): React.JSX.Element {
  const answer = parseVqa(atom.content);
  if (!answer) return <p className="annotation-error">This VQA answer is invalid and cannot be edited.</p>;
  if ("count" in answer) return <label>Edit count<input aria-label="Edit VQA count" type="number" min="0" value={answer.count}
    onChange={(event) => onChange({ ...answer, count: Math.max(0, Math.trunc(Number(event.target.value))) })} /></label>;
  if ("attribute" in answer) return <><label>Edit attribute<input aria-label="Edit VQA attribute" value={answer.attribute}
    onChange={(event) => onChange({ ...answer, attribute: event.target.value })} /></label><label>Edit value<input aria-label="Edit VQA attribute value" value={answer.value}
    onChange={(event) => onChange({ ...answer, value: event.target.value })} /></label></>;
  if ("subject" in answer) return <><label>Edit relation<input aria-label="Edit VQA relation" value={answer.relation}
    onChange={(event) => onChange({ ...answer, relation: event.target.value })} /></label><label>Edit object<input aria-label="Edit VQA object" value={answer.object}
    onChange={(event) => onChange({ ...answer, object: event.target.value })} /></label></>;
  return <p className="annotation-hint">Drag this {"detections" in answer ? "box" : "keypoint"} on the camera canvas to move it.</p>;
}

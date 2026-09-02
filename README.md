# DatasetUI

DatasetUI is a secure desktop viewer and annotation editor for LeRobot datasets. It combines a sandboxed Electron/React interface with a fixed-method Python JSON-RPC backend and supports local datasets as well as transiently authenticated Hugging Face Hub imports.

The implementation tracks all **37 capabilities** in `contracts/space-parity.yaml` against the pinned `lerobot/visualize_dataset` revision `d724744111cae6feb9a2194e607e71749813a97a`. Every capability has an exact executable Vitest title and is marked `parity-tested`; upstream-derived numeric constants are recorded in `contracts/space-parity-goldens.json`.

## Implemented capabilities

### Dataset access and playback

- Local project registry plus Hub dataset search, information lookup, authenticated import, and registration. Hub tokens are transient and are never persisted in project state.
- Episode sidebar with 100-item pagination, direct and keyboard navigation, synchronized multi-camera playback, consolidated-video segment seeking/looping, and per-camera hide/enlarge controls.
- Grayscale and depth colormaps, synchronized action/state charts, chart grouping and click-to-seek.
- Dataset summary statistics, episode-length histogram, paginated/lazy first-last frame gallery, persistent episode flags with ID and LeRobot CLI export, and duration/low-movement filters.

### Analysis, progress, and replay

- Action autocorrelation and chunk suggestion, normalized velocity/activity and discrete-motor diagnostics, jerk ranking, cross-episode variance heatmap, demonstrator-speed CV, and state/action lag alignment.
- Optional `sarm_progress.parquet` / `srm_progress.parquet` overlays.
- Replay mappings for SO100/SO101/SO follower, OpenArm, and Unitree G1, including unit conversion, gripper handling, orbit interaction, episode switching, and one-second end-effector trails.
- The pinned Space Doctor link and a local runtime doctor.

### Annotations

- Canonical persistent `task_aug`, `subtask`, `plan`, and `memory` atoms.
- Frame-snapped interjection/speech pairs and canonical `say` tool calls.
- VQA bbox, keypoint, count, attribute, and spatial-relation answers.
- Editable timeline spans/ticks, inspect/delete flows, camera-specific overlays, bbox draw/drag, and keypoint placement/movement.
- Safe annotation list/save/export through the fixed RPC surface; source data is not overwritten implicitly.

## Supported data and compatibility

- LeRobot v2.1 (`lerobot==0.3.3`, canonical marker `v2.1`).
- Core LeRobot v3 (`lerobot==0.6.0`, canonical marker `v3.0`).
- Annotated v3.1 Arrow `list<struct>` language columns.
- Read-only inspection of legacy `v1.<integer>` and `v2.0` layouts through the version-neutral `DatasetDocument`.

The derived metadata index lives under the application XDG cache directory. Opening and indexing inspect metadata, Parquet footers/schemas, and media metadata without decoding data row groups or video frames. Explicit validation additionally checks episode/frame totals, schemas, timestamps, and referenced image/video files.

## Security model

Electron launches only `python -m lerobot_dataset_editor.rpc` with `shell: false`, a fixed working directory, and a secret-free environment allowlist. The framed JSON-RPC transport is capped at 16 MiB and exposes only the methods frozen in `contracts/rpc-transport.schema.json`; there is no generic command or IPC channel.

The renderer uses context isolation, sandboxing, web security, no renderer Node access, no webviews, no insecure content, no popup/navigation escape, and a CSP without `unsafe-inline` or `unsafe-eval`. The frozen preload bridge exposes fixed typed methods only. Dataset media is served through a validated `datasetui-media:` protocol confined to registered project roots. External navigation is restricted to the exact pinned Doctor URL shape.

Registry and cache state is kept outside datasets:

```text
$XDG_CONFIG_HOME/lerobot-dataset-editor/
$XDG_DATA_HOME/lerobot-dataset-editor/projects.sqlite
$XDG_STATE_HOME/lerobot-dataset-editor/
$XDG_CACHE_HOME/lerobot-dataset-editor/
```

## Setup and verification

Requirements: Node.js 22.12+, npm 11, and Python 3.12.

```bash
cd /home/kgs/workspace/DatasetUI
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
npm ci
```

Run the complete local verification gate:

```bash
npm run verify          # TypeScript, all Node/Vitest tests, Python tests, production build
npm run test:electron   # Real Electron end-to-end tests (requires a display or xvfb-run)
npm audit --audit-level=high
```

Focused commands:

```bash
npm run typecheck
npm run test:node
npm run test:python
npm run test:parity     # all 37 contract mappings plus backend parity goldens
npm run build
```

CI runs the same verification, real Electron tests under Xvfb, and a high-severity dependency audit on Ubuntu 24.04.

## AppImage feasibility artifact

```bash
npm run appimage:spike
```

This builds and verifies `release/DatasetUI-0.2.0-linux-x86_64.AppImage`, including packaged sources, contracts, fixtures, and an RPC smoke launch. The current artifact remains a `host-python-spike` (`self_contained=false`): it requires host Python 3.12 with the locked DatasetUI Python dependencies. This packaging limitation does not affect the 37 implemented application capabilities, but a fully bundled Python runtime is still required for a self-contained production AppImage.

## Repository layout

```text
contracts/                              schemas, RPC surface, 37-item parity inventory/goldens
electron/src/main/                      backend lifecycle, media protocol, trusted IPC
electron/src/preload/                   frozen typed renderer bridge
electron/src/renderer/parity/           viewer, analytics, progress, replay, Doctor
electron/src/renderer/annotations/      annotation editor, timeline, overlays
electron/tests/parity/                  exact capability-title and model tests
python/src/lerobot_dataset_editor/
  dataset/                              version-neutral documents, index, validation
  parity/                               viewer analytics, Hub, progress, replay, annotations
  registry/                             XDG SQLite project registry
  rpc/                                  bounded framed JSON-RPC server
  runtime/                              embedded/external runtime doctor and device selection
tests/parity/                           backend numeric/security parity tests
fixtures/                               deterministic contract-derived fixtures
tests/fixtures/official/                official-writer-generated compatibility goldens
third_party/visualize_dataset.*         pinned upstream revision and notice
```

## Dependency policy

Runtime Python dependencies are exact-pinned in `pyproject.toml`/`uv.lock`; desktop dependencies are exact-pinned in `package.json`/`package-lock.json`. No source dataset is modified by open, browse, index, or validate operations. Generated reports, registry data, indexes, annotation exports, and package artifacts are written only to explicit output or application-state locations.

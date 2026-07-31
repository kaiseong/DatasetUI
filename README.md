# LeRobot Dataset Editor

Standalone LeRobot dataset inspection/editing application under development in `/home/kgs/DatasetUI`. Task 1 freezes format, annotation, Space-parity, fixture, report, and existing `dataset_tools.sh` contracts. Task 2 provides the secure Electron↔Python vertical slice and Ubuntu x86_64 AppImage feasibility spike. Task 3 adds the local project registry and embedded/external runtime doctor. Task 4 adds read-only version-neutral documents, explicit structural validation, an incremental metadata index, and a side-by-side schema/episode browser. The characterized launcher and Python tools remain vendored byte-for-byte; the original `/home/kgs/lerobot-rby1` files are unmodified.

## Frozen baselines

- LeRobot v2.1: official `lerobot==0.3.3`; canonical marker `v2.1`.
- LeRobot v3: official `lerobot==0.6.0`; core layout marker `v3.0`.
- Annotated v3: v3.1 Arrow `list<struct>` language columns.
- Space parity: revision `d724744111cae6feb9a2194e607e71749813a97a`, with 37 explicitly planned capabilities and executable test IDs.
- v2.1 editor annotations remain outside the strict core in `meta/lerobot_annotations.json`.

Image-backed official datasets use `video_path: null`; contracts also accept canonical video templates for video-backed datasets. Official v0.3.3 does not always emit global `meta/stats.json`, so the report identifies that optional extension explicitly.

## Reproducible setup and validation

Python dependencies are exact-locked in `uv.lock`:

```bash
cd /home/kgs/DatasetUI
uv sync --python /usr/bin/python3.12 --extra test --frozen
uv run --extra test pytest -q
uv run --extra test dataset-editor-report --regenerate-fixtures
```

Desktop dependencies are exact-pinned in `package.json` and `package-lock.json`:

```bash
npm ci
npm run typecheck
npm run test:node       # framing/client/security plus real Python subprocess
npm run test:electron   # production build plus real Electron E2E
```

The report fails on invalid schemas/fixtures, Space drift, source-fingerprint drift, or stale official compatibility evidence. It conforms to `contracts/rpc.schema.json`.

## Secure Electron↔Python boundary

Electron starts only the fixed backend command `python -m lerobot_dataset_editor.rpc`. It uses `shell: false`, a fixed working directory, and an environment allowlist: `HOME`, `LANG`, `LC_ALL`, `PATH`, and the four safe XDG location variables. It adds fixed `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, and packaged-source `PYTHONPATH`; credentials and tokens are not copied. Stdout is framed protocol only and diagnostics use stderr.

`contracts/rpc-transport.schema.json` freezes protocol version 1 and `Content-Length: N\r\n\r\n<payload>`. Length is UTF-8 bytes, payloads are capped at 16 MiB, headers at 8 KiB, batches are rejected, and IDs correlate requests. The fixed surface includes initialize/shutdown/ping/report, project register/list/get/update/remove, runtime doctor/select, and dataset open/browse/validate. No generic method or IPC channel is exposed.

The sandboxed CommonJS preload exposes one frozen `window.datasetEditor` object containing only those fixed methods. Main validates sender URLs. The window enables context isolation, sandboxing, and web security; disables renderer Node, webviews, insecure content, popups, external navigation, and permissions; and uses a CSP without `unsafe-inline` or `unsafe-eval`.

## Project registry and runtime management

Registry state never lives inside a dataset:

```text
$XDG_CONFIG_HOME/lerobot-dataset-editor/
$XDG_DATA_HOME/lerobot-dataset-editor/projects.sqlite
$XDG_STATE_HOME/lerobot-dataset-editor/
$XDG_CACHE_HOME/lerobot-dataset-editor/
```

The SQLite registry uses versioned idempotent migrations, WAL, private application directories, parameterized queries, and an explicit newer-schema refusal. It stores local source/output references, selected revision, an explicit preferred target (`v2.1` or `v3`), recent-open ordering, and embedded/external runtime paths. It never provides a token field and rejects detected Hub tokens. Removing or opening a project changes registry state only; it never deletes or writes source data. Missing and read-only source paths are reported rather than repaired silently.

The renderer can register multiple projects with an explicit target and switch runtimes. An external runtime is persisted only after a fixed-command handshake confirms Python 3.12, exact `lerobot==0.6.0`, and executable FFmpeg/codec reporting. Probe subprocesses receive a secret-free environment. The doctor reports Python, LeRobot, FFmpeg/codecs, CPU, Torch/CUDA runtime, NVIDIA driver/GPU/VRAM, compatibility issues, and device choice. `auto` falls back to CPU on missing/import/init/VRAM failures; `gpu-only` raises an explicit error and never falls back. Task 3 does not claim that production runtimes are bundled—that remains Task 22.

## Version-neutral read-only dataset browser

Task 4 reads legacy `v1.<integer>`, `v2.0`, `v2.1`, and core `v3.0` datasets through one immutable `DatasetDocument`; a core-v3 dataset with v3.1 language features is reported as the annotated `v3.1` extension. Unknown or future core versions fail closed. The document contains canonical source/version data, feature specs, episode file references, cumulative half-open frame ranges, media/task/annotation references, provenance, and validation status. The renderer can select two registered projects independently and displays their common feature schemas and bounded episode-reference lists side by side.

The derived index lives at `$XDG_CACHE_HOME/lerobot-dataset-editor/index.sqlite`, uses WAL and versioned cache-only migrations, and fingerprints metadata, data Parquet, videos, and provenance. It reads Parquet footers/schemas plus small episode/task metadata only; dataset open and index build never read data row groups. Video indexing invokes fixed, shell-free `ffprobe -v error -print_format json -show_format -show_streams <video>` and never requests frames or decode intervals. Any indexed source-file stat change invalidates the cached document.

`dataset.validate` is deliberately explicit because it may read timestamp columns. It checks zero-based unique contiguous episodes, declared episode/frame totals, every metadata and data Parquet footer, every data-file schema and primitive feature type, finite monotonic timestamps across row groups/files, and referenced image/video media. Opening, indexing, browsing, and validating are source-preserving: SQLite state is written only under XDG application directories, never into a dataset. Task 4 provides no playback, charts, frame navigation, or writes; those begin in later tasks.

## Ubuntu x86_64 AppImage feasibility spike

```bash
npm run appimage:spike
```

This builds `release/DatasetUI-0.2.0-linux-x86_64.AppImage`, extracts it with `--appimage-extract` without FUSE, checks packaged sources/contracts/fixtures/tools, and launches extracted `AppRun --rpc-smoke`. Retained evidence is `appimage-spike/evidence/spike-result.json`, validated by `contracts/appimage-spike.schema.json`; generated artifacts/extraction trees are ignored.

**Task 2 limitation:** this artifact remains `host-python-spike`, `self_contained=false`. It needs host Python 3.12 with locked DatasetUI dependencies. Full/Lightweight production Python bundling, production icons/metadata, and final dependency pruning are Task 22.

## Fixture provenance

Two fixture classes remain intentionally separate:

1. `fixtures/`: deterministic **CONTRACT-DERIVED** editor fixtures.
2. `tests/fixtures/official/`: small **official-writer-generated** `v21_v033` and `v30_v060` goldens.

Official fixture manifests record generation/dependency provenance and hashes. `contracts/official-compatibility.json` binds full-loader evidence to exact tree hashes. Compatibility adapters replace only unused Torch/Torchvision/Safetensors import surfaces; real official LeRobot metadata/data code, PyArrow, Hugging Face Datasets, Pillow, and PyAV performed writer/loader work. See `scripts/compat/README.md`.

## Layout

```text
DatasetUI/
├── appimage-spike/evidence/
├── contracts/
│   ├── appimage-spike.schema.json
│   ├── project-registry.schema.json
│   ├── rpc-transport.schema.json
│   ├── lerobot-v21.schema.json
│   ├── lerobot-v30.schema.json
│   ├── language-v31.schema.json
│   └── space-parity.yaml
├── electron/
│   ├── src/main/                      # backend lifecycle and trusted fixed IPC
│   ├── src/preload/                   # frozen typed bridge
│   ├── src/renderer/                  # strict-CSP React shell/project UI
│   └── tests/                         # Node and real-Electron tests
├── fixtures/
├── python/src/lerobot_dataset_editor/
│   ├── dataset/                       # common documents, validation, lazy XDG index
│   ├── registry/                      # XDG SQLite projects and migrations
│   ├── runtime/                       # embedded/external doctor and devices
│   ├── rpc/                           # bounded framed JSON-RPC server
│   ├── characterization.py
│   ├── cli.py
│   ├── fixtures.py
│   └── schemas.py
├── scripts/compat/
├── tests/{appimage,registry,rpc,runtime,compat}/
├── tools/lerobot_dataset_tools/
├── dataset_tools.sh
├── electron-builder.yml
├── package.json
├── package-lock.json
├── pyproject.toml
└── uv.lock
```

## Exact direct dependencies

Python runtime: `jsonschema==4.23.0`, `pyarrow==20.0.0`, `pyyaml==6.0.2`. Python test: `pytest==8.3.5`, `pytest-timeout==2.3.1`. Build backend: `hatchling==1.27.0`.

Desktop runtime: `react==19.2.8`, `react-dom==19.2.8`. Build/test pins include `electron==43.2.0`, `electron-builder==26.15.3`, `electron-vite==5.0.0`, `vite==7.3.6`, `vitest==4.1.10`, `typescript==7.0.2`, and `playwright-core==1.62.1`. The lockfile overrides transitive `brace-expansion` to patched exact `5.0.9`.

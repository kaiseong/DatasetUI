# LeRobot Dataset Editor

Standalone LeRobot dataset inspection/editing application under development in `/home/kgs/DatasetUI`. Task 1 freezes the format, annotation, Space-parity, fixture, RPC-report, and existing `dataset_tools.sh` behavior contracts. Task 2 adds a secure Electron↔Python vertical slice and an Ubuntu x86_64 AppImage feasibility spike. The characterized launcher and Python tools are vendored here byte-for-byte; the original `/home/kgs/lerobot-rby1` files remain unmodified.

## Frozen baselines

- LeRobot v2.1: official `lerobot==0.3.3`; canonical marker `v2.1`.
- LeRobot v3: official `lerobot==0.6.0`; core layout marker `v3.0`.
- Annotated v3: v3.1 Arrow `list<struct>` language columns.
- Space parity: revision `d724744111cae6feb9a2194e607e71749813a97a`, with 37 explicitly planned capabilities and future executable test IDs.
- v2.1 editor annotations remain outside the strict core in `meta/lerobot_annotations.json`.

Image-backed official datasets use `video_path: null`; the contracts also accept the canonical video templates for video-backed datasets. Official v0.3.3 always writes `info.json`, `episodes.jsonl`, `episodes_stats.jsonl`, and `tasks.jsonl` in the exercised writer flow, but does not always emit global `meta/stats.json`. The contract-derived v2.1 fixture includes global stats as an explicit editor fixture extension, and the report exposes whether that optional file is present.

## Reproducible setup and validation

Python is packaged from `python/src/lerobot_dataset_editor` and dependencies are locked in `uv.lock` with exact direct pins.

```bash
cd /home/kgs/DatasetUI
uv sync --python /usr/bin/python3.12 --extra test --frozen
uv run --extra test pytest -q
uv run --extra test dataset-editor-report --regenerate-fixtures
```

The report exits nonzero if a schema, fixture, Space pin, source fingerprint, golden fixture, or hash-bound official compatibility record is invalid. It conforms to `contracts/rpc.schema.json`.

The desktop dependencies are exact-pinned in `package.json` and `package-lock.json`.

```bash
npm ci
npm run typecheck
npm run test:node       # TypeScript framing/client/security + real Python subprocess
npm run test:electron   # production build + real Electron renderer/security E2E
```

## Secure Electron↔Python vertical slice

Electron starts the backend with fixed arguments only:

```text
python -m lerobot_dataset_editor.rpc
```

The subprocess uses `shell: false`, a fixed working directory, and an environment allowlist (`HOME`, `LANG`, `LC_ALL`, `PATH`) plus forced `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, and packaged-source `PYTHONPATH`. Parent credentials and tokens are not copied. Stdout is reserved for framed JSON-RPC; Python diagnostics go to stderr.

`contracts/rpc-transport.schema.json` freezes protocol version 1 and the `Content-Length: N\r\n\r\n<payload>` transport. Length is measured in UTF-8 bytes, payloads are capped at 16 MiB, headers at 8 KiB, batches are rejected, and requests are correlated by ID. The fixed method surface is `initialize`, `system.ping`, `report.get`, and `shutdown`; the server emits `server.ready` and `server.exit` notifications.

The renderer has no Node integration. Its sandboxed CommonJS preload exposes one frozen object, `window.datasetEditor`, with only `ping()` and `report()`. Main-process handlers validate the sender URL and provide no arbitrary IPC channel. The window enables context isolation, sandboxing, and web security; disables Node in frames/workers, webviews, insecure content, popups, external navigation, and all permissions; and uses a CSP without `unsafe-inline` or `unsafe-eval`.

## Ubuntu x86_64 AppImage feasibility spike

Run the complete spike after the Python and Node setup:

```bash
npm run appimage:spike
```

This command builds `release/DatasetUI-0.2.0-linux-x86_64.AppImage`, extracts it with `--appimage-extract` (so FUSE is not required), checks the packaged Python sources/contracts/fixtures/tools, and launches the extracted `AppRun --rpc-smoke`. The smoke path starts host Python, completes initialize and `system.ping`, shuts down cleanly, and requires the `DATASETUI_RPC_SMOKE=...` marker.

The retained result is `appimage-spike/evidence/spike-result.json`, validated by `contracts/appimage-spike.schema.json`. The generated AppImage and extraction trees are intentionally ignored.

**Task 2 limitation:** this is a `host-python-spike`, not a distributable self-contained application. The verifier defaults to `.venv/bin/python` (or an absolute `LEROBOT_DATASET_EDITOR_PYTHON`) and therefore depends on host Python 3.12 with the locked DatasetUI runtime dependencies already installed. `self_contained` is explicitly `false`; Full/Lightweight production Python bundling, production icons/metadata, and final dependency pruning are deferred to Task 22.

## Fixture provenance

Two fixture classes are intentionally separate:

1. `fixtures/` contains deterministic **CONTRACT-DERIVED** editor fixtures. Their `PROVENANCE.md` files state that they were not produced by an official writer.
2. `tests/fixtures/official/` contains small **official-writer-generated** golden fixtures:
   - `v21_v033`, generated with `LeRobotDataset.create/add_frame/save_episode` from `lerobot==0.3.3`.
   - `v30_v060`, generated with `LeRobotDataset.create/add_frame/save_episode/finalize` from `lerobot==0.6.0`.

Each official fixture has a `PROVENANCE.json` manifest containing Python and dependency versions, generation script, metadata-only adapter disclosure, and SHA-256 hashes. `contracts/official-compatibility.json` binds successful full-loader results to exact fixture tree hashes, so the report cannot claim compatibility after fixture drift.

The lightweight adapters in `tests/compat/stubs/` replace only unused Torch/Torchvision/Safetensors import surfaces; they do not implement model execution, training, CUDA, or encoding. Real official LeRobot data/metadata code, PyArrow/Hugging Face Datasets, Pillow, and (for v0.6.0) PyAV performed the writer/loader work. See `scripts/compat/README.md` for exact generation and opt-in validation commands.

## Layout

```text
DatasetUI/
├── appimage-spike/evidence/           # retained, schema-validated spike result
├── contracts/
│   ├── appimage-spike.schema.json
│   ├── dataset-tools-characterization.json
│   ├── official-compatibility.json
│   ├── lerobot-v21.schema.json
│   ├── lerobot-v30.schema.json
│   ├── language-v31.schema.json
│   ├── rpc.schema.json
│   ├── rpc-transport.schema.json
│   └── space-parity.yaml
├── electron/
│   ├── src/main/                      # backend lifecycle, trusted fixed IPC
│   ├── src/preload/                   # frozen ping/report bridge
│   ├── src/renderer/                  # strict-CSP React shell
│   └── tests/                         # Node and real-Electron tests
├── fixtures/                          # deterministic contract-derived fixtures
├── python/src/lerobot_dataset_editor/
│   ├── rpc/                           # bounded framed JSON-RPC server
│   ├── characterization.py
│   ├── cli.py
│   ├── fixtures.py
│   └── schemas.py
├── scripts/
│   ├── compat/                        # exact official generation/validation
│   └── verify-appimage.mjs            # no-FUSE extraction and RPC smoke
├── tests/
│   ├── appimage/                      # packaging/evidence acceptance contract
│   ├── compat/                        # adapters and opt-in official-loader gate
│   └── fixtures/official/             # official writer goldens
├── third_party/visualize_dataset.REVISION
├── tools/lerobot_dataset_tools/       # vendored exact characterized sources
├── dataset_tools.sh                   # vendored exact launcher
├── electron-builder.yml
├── package.json
├── package-lock.json
├── pyproject.toml
├── uv.lock
└── README.md
```

## Exact direct dependencies

Python runtime: `jsonschema==4.23.0`, `pyarrow==20.0.0`, `pyyaml==6.0.2`. Python test: `pytest==8.3.5`, `pytest-timeout==2.3.1`. Build backend: `hatchling==1.27.0`.

Desktop runtime: `react==19.2.8`, `react-dom==19.2.8`. Desktop build/test pins include `electron==43.2.0`, `electron-builder==26.15.3`, `electron-vite==5.0.0`, `vite==7.3.6`, `vitest==4.1.10`, `typescript==7.0.2`, and `playwright-core==1.62.1`. The lockfile overrides transitive `brace-expansion` to patched exact version `5.0.9`.

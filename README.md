# LeRobot Dataset Editor

Standalone LeRobot dataset inspection/editing application under development in `/home/kgs/DatasetUI`. Task 1 freezes the format, annotation, Space-parity, fixture, RPC, and existing `dataset_tools.sh` behavior contracts that later tasks build on. The characterized launcher and Python tools are vendored here byte-for-byte; the original `/home/kgs/lerobot-rby1` files remain unmodified.

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
├── contracts/
│   ├── dataset-tools-characterization.json
│   ├── official-compatibility.json
│   ├── lerobot-v21.schema.json
│   ├── lerobot-v30.schema.json
│   ├── language-v31.schema.json
│   ├── rpc.schema.json
│   └── space-parity.yaml
├── fixtures/                         # deterministic contract-derived fixtures
├── python/src/lerobot_dataset_editor/
│   ├── characterization.py
│   ├── cli.py
│   ├── fixtures.py
│   └── schemas.py
├── scripts/compat/                   # exact official generation/validation scripts
├── tests/
│   ├── compat/                       # adapters and opt-in official-loader gate
│   └── fixtures/official/            # official writer goldens
├── third_party/visualize_dataset.REVISION
├── tools/lerobot_dataset_tools/       # vendored exact characterized sources
├── dataset_tools.sh                   # vendored exact launcher
├── LICENSE
├── pyproject.toml
├── uv.lock
└── README.md
```

## Exact direct dependencies

Runtime: `jsonschema==4.23.0`, `pyarrow==20.0.0`, `pyyaml==6.0.2`. Test: `pytest==8.3.5`, `pytest-timeout==2.3.1`. Build backend: `hatchling==1.27.0`.

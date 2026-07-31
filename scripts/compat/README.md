# Exact official LeRobot compatibility probes

These scripts generate and validate the Task 1 golden fixtures with the exact supported baselines:

- LeRobot v2.1: `lerobot==0.3.3`
- LeRobot v3.0: `lerobot==0.6.0`

Every generating command requires an explicit `--format` (`v2.1` or `v3`). The checked-in fixtures under `tests/fixtures/official/` were produced by the official writer APIs; their `PROVENANCE.json` files pin Python, dependency versions, adapter use, generation scripts, and SHA-256 hashes.

The adapters in `tests/compat/stubs/` replace only Torch/Torchvision/Safetensors surfaces not used by metadata, Parquet, or embedded-image handling. They do not implement model execution, CUDA, training, or media encoding. Official `lerobot` source files were not modified. v0.6.0 uses real pinned PyAV. The v0.6 script imports the official dataset modules directly to avoid unrelated eager Gym/environment imports from `lerobot.datasets.__init__`.

## Reproduce

Create isolated Python 3.12 environments and install the exact dependency sets recorded in each fixture manifest. Then run:

```bash
PYTHONPATH=tests/compat/stubs/v033 /path/to/v033/python \
  scripts/compat/generate_v21_v033.py --format v2.1 \
  --output /tmp/official-v21 --force

PYTHONPATH=tests/compat/stubs/v060 /path/to/v060/python \
  scripts/compat/generate_v30_v060.py --format v3 \
  --output /tmp/official-v30 --force
```

Validate any local fixture with the full official loader:

```bash
PYTHONPATH=tests/compat/stubs/v033 /path/to/v033/python \
  scripts/compat/validate_official.py --format v2.1 --dataset fixtures/v21_valid

PYTHONPATH=tests/compat/stubs/v060 /path/to/v060/python \
  scripts/compat/validate_official.py --format v3 --dataset fixtures/v30_valid
```

The opt-in pytest gate uses `LEROBOT_V033_PYTHON` and `LEROBOT_V060_PYTHON`; it skips when those exact external environments are unavailable.

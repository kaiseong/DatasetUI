import json
import sys
from types import ModuleType

import numpy as np
import pytest

from test_relative_artifacts import relative_output as _relative_output

from datasetui.relative.artifacts import load_relative_processors
from datasetui.transform_errors import CurationTransformError


relative_output = _relative_output


@pytest.fixture
def fake_public_pipeline(monkeypatch):
    import datasetui.official.runtime as runtime

    gate_calls = []
    monkeypatch.setattr(runtime, "require_runtime", lambda: gate_calls.append(True))

    class Relative:
        def __init__(self, **config):
            self.config = config

        def _build_mask(self, width):
            return [
                name not in self.config["exclude_joints"]
                for name in self.config["action_names"][:width]
            ]

        def __call__(self, action, state):
            self.state = state
            return action - state * self._build_mask(action.shape[-1])

    class Absolute:
        def __init__(self, relative_step):
            self.relative_step = relative_step

        def __call__(self, action):
            return action + self.relative_step.state * self.relative_step._build_mask(
                action.shape[-1]
            )

    class Pipeline:
        @classmethod
        def from_pretrained(cls, root, *, config_filename, overrides=None):
            assert root.name.startswith("datasetui-relative-processors-")
            value = json.loads((root / config_filename).read_text())
            step = value["steps"][0]
            result = cls()
            if step["registry_name"] == "relative_actions_processor":
                result.steps = [Relative(**step["config"])]
            else:
                result.steps = [
                    Absolute(overrides["absolute_actions_processor"]["relative_step"])
                ]
            return result

    module = ModuleType("lerobot.processor")
    module.DataProcessorPipeline = Pipeline
    module.RelativeActionsProcessorStep = Relative
    module.AbsoluteActionsProcessorStep = Absolute
    monkeypatch.setitem(sys.modules, "lerobot.processor", module)
    return gate_calls


def test_loader_uses_private_configs_and_reconnects_pair(
    relative_output, fake_public_pipeline
):
    _, root, _ = relative_output
    relative, absolute = load_relative_processors(root, chunk_size=3)
    assert fake_public_pipeline == [True]
    assert absolute.relative_step is relative
    action = np.array([[3.0], [4.0], [5.0]])
    state = np.array([2.0])
    np.testing.assert_array_equal(absolute(relative(action, state)), action)


@pytest.mark.parametrize("tamper", ["horizon", "class", "statistics", "backup"])
def test_loader_rejects_invalid_contract_before_pipeline(
    relative_output, fake_public_pipeline, tamper
):
    _, root, _ = relative_output
    if tamper == "class":
        path = root / "meta/datasetui_relative_preprocessor.json"
        value = json.loads(path.read_text())
        value["steps"][0]["class"] = "not.allowed"
        path.write_text(json.dumps(value))
    elif tamper == "statistics":
        path = root / "meta/stats.json"
        value = json.loads(path.read_text())
        value["action"]["mean"] = [999]
        path.write_text(json.dumps(value))
    elif tamper == "backup":
        (root / "meta/stats.absolute.json").unlink()
    with pytest.raises((ValueError, CurationTransformError, OSError)):
        load_relative_processors(root, chunk_size=4 if tamper == "horizon" else 3)

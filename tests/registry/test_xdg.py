from __future__ import annotations

from pathlib import Path


def test_xdg_paths_use_lerobot_dataset_editor_namespace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    from lerobot_dataset_editor.registry.xdg import xdg_paths

    paths = xdg_paths()
    assert paths.config_dir == tmp_path / "config" / "lerobot-dataset-editor"
    assert paths.data_dir == tmp_path / "data" / "lerobot-dataset-editor"
    assert paths.state_dir == tmp_path / "state" / "lerobot-dataset-editor"
    assert paths.cache_dir == tmp_path / "cache" / "lerobot-dataset-editor"
    assert paths.registry_db == paths.data_dir / "projects.sqlite"


def test_xdg_paths_follow_home_fallbacks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)

    from lerobot_dataset_editor.registry.xdg import xdg_paths

    paths = xdg_paths()
    assert paths.config_dir == tmp_path / ".config" / "lerobot-dataset-editor"
    assert paths.data_dir == tmp_path / ".local" / "share" / "lerobot-dataset-editor"
    assert paths.state_dir == tmp_path / ".local" / "state" / "lerobot-dataset-editor"
    assert paths.cache_dir == tmp_path / ".cache" / "lerobot-dataset-editor"

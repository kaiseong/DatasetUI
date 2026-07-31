"""XDG paths for DatasetUI state kept outside source datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR = "lerobot-dataset-editor"


@dataclass(frozen=True)
class XdgPaths:
    config_dir: Path
    data_dir: Path
    state_dir: Path
    cache_dir: Path

    @property
    def registry_db(self) -> Path:
        return self.data_dir / "projects.sqlite"


def _base(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else fallback


def xdg_paths() -> XdgPaths:
    """Resolve XDG paths without creating or mutating them."""
    home = Path.home()
    return XdgPaths(
        config_dir=_base("XDG_CONFIG_HOME", home / ".config") / APP_DIR,
        data_dir=_base("XDG_DATA_HOME", home / ".local" / "share") / APP_DIR,
        state_dir=_base("XDG_STATE_HOME", home / ".local" / "state") / APP_DIR,
        cache_dir=_base("XDG_CACHE_HOME", home / ".cache") / APP_DIR,
    )


def ensure_xdg_dirs(*, mode: int = 0o700) -> XdgPaths:
    """Create application-owned XDG directories with private permissions."""
    paths = xdg_paths()
    for directory in (paths.config_dir, paths.data_dir, paths.state_dir, paths.cache_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=mode)
        os.chmod(directory, mode)
    return paths

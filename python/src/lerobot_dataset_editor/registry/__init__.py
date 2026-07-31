"""Project registry — XDG SQLite store for dataset projects."""

from .db import open_registry
from .models import Project
from .repository import (
    get_project,
    list_recent,
    mark_project_opened,
    register_project,
    remove_project,
    update_project,
)
from .schema import RegistryVersionError
from .xdg import XdgPaths, xdg_paths

__all__ = [
    "Project",
    "RegistryVersionError",
    "XdgPaths",
    "get_project",
    "list_recent",
    "mark_project_opened",
    "open_registry",
    "register_project",
    "remove_project",
    "update_project",
    "xdg_paths",
]

"""Stateful JSON-RPC server for the Electron subprocess boundary."""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, BinaryIO

from ..cli import generate_report
from .framing import (
    FrameProtocolError,
    FrameTooLargeError,
    JsonPayloadError,
    encode_frame,
    read_frame,
)
from .protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_NOT_INITIALIZED,
    RpcRequest,
    error_response,
    notification,
    parse_request,
    success_response,
)

PROTOCOL_VERSION = 1
METHODS = [
    "annotations.export",
    "annotations.list",
    "annotations.save",
    "dataset.analytics",
    "dataset.browse",
    "dataset.episode",
    "dataset.open",
    "dataset.summary",
    "dataset.validate",
    "hub.import",
    "hub.info",
    "hub.search",
    "initialize",
    "media.resolve",
    "progress.read",
    "project.get",
    "project.list",
    "project.register",
    "project.remove",
    "project.update",
    "report.get",
    "replay.map",
    "runtime.doctor",
    "runtime.select",
    "shutdown",
    "system.ping",
]


class RpcDispatchError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class RpcServer:
    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_requested = False
        self.started_at = time.monotonic()
        self._registry_conn = None

    def _get_registry(self):
        """Lazy-open the registry database connection."""
        if self._registry_conn is None:
            from ..registry.db import open_registry
            self._registry_conn = open_registry()
        return self._registry_conn

    def dispatch(self, request: RpcRequest) -> Any:
        if request.method == "initialize":
            return self._initialize(request.params)
        if request.method == "shutdown":
            self.shutdown_requested = True
            if self._registry_conn is not None:
                self._registry_conn.close()
                self._registry_conn = None
            return None
        if not self.initialized:
            raise RpcDispatchError(
                SERVER_NOT_INITIALIZED,
                "Server not initialized",
                {"requiredMethod": "initialize"},
            )
        if request.method == "system.ping":
            return {
                "service": "lerobot-dataset-editor",
                "protocolVersion": PROTOCOL_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptimeMs": int((time.monotonic() - self.started_at) * 1000),
            }
        if request.method == "report.get":
            return generate_report()
        if request.method == "project.register":
            return self._project_register(request.params)
        if request.method == "project.list":
            return self._project_list(request.params)
        if request.method == "project.get":
            return self._project_get(request.params)
        if request.method == "project.update":
            return self._project_update(request.params)
        if request.method == "project.remove":
            return self._project_remove(request.params)
        if request.method == "runtime.doctor":
            return self._runtime_doctor(request.params)
        if request.method == "runtime.select":
            return self._runtime_select(request.params)
        if request.method == "dataset.open":
            return self._dataset_open(request.params)
        if request.method == "dataset.browse":
            return self._dataset_browse(request.params)
        if request.method == "dataset.validate":
            return self._dataset_validate(request.params)
        if request.method in {
            "dataset.summary", "dataset.episode", "dataset.analytics", "progress.read",
            "replay.map", "annotations.list", "annotations.save", "annotations.export",
            "media.resolve",
        }:
            return self._parity_dispatch(request.method, request.params)
        if request.method in {"hub.search", "hub.info", "hub.import"}:
            return self._hub_dispatch(request.method, request.params)
        raise RpcDispatchError(
            METHOD_NOT_FOUND,
            "Method not found",
            {"method": request.method},
        )

    def _initialize(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        if self.initialized:
            raise RpcDispatchError(INVALID_REQUEST, "Server is already initialized")
        if not isinstance(params, dict) or params.get("protocolVersion") != PROTOCOL_VERSION:
            raise RpcDispatchError(
                INVALID_PARAMS,
                "initialize requires the supported protocolVersion",
                {"supportedProtocolVersion": PROTOCOL_VERSION},
            )
        client = params.get("client")
        if not isinstance(client, dict) or not isinstance(client.get("name"), str):
            raise RpcDispatchError(
                INVALID_PARAMS,
                "initialize requires client.name",
            )
        self.initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "server": {"name": "lerobot-dataset-editor", "version": "0.1.0"},
            "methods": METHODS,
        }

    def _project_register(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..registry.repository import register_project

        if not isinstance(params, dict):
            raise RpcDispatchError(INVALID_PARAMS, "params must be an object")
        name = params.get("name")
        source_path = params.get("source_path")
        target_format = params.get("target_format")
        runtime_mode = params.get("runtime_mode", "embedded")
        if not isinstance(name, str) or not name.strip():
            raise RpcDispatchError(INVALID_PARAMS, "name must be a non-empty string")
        if not isinstance(source_path, str) or not source_path:
            raise RpcDispatchError(INVALID_PARAMS, "source_path must be a non-empty string")
        if runtime_mode == "external":
            report = self._doctor_from_values(
                mode="external",
                python_path=params.get("python_path"),
                ffmpeg_path=params.get("ffmpeg_path"),
                device_policy=params.get("device_policy", "auto"),
            )
            self._require_compatible(report)
        try:
            return register_project(
                self._get_registry(),
                name=name.strip(),
                source_path=source_path,
                target_format=target_format,
                runtime_mode=runtime_mode,
                output_path=params.get("output_path"),
                selected_revision=params.get("selected_revision"),
                python_path=params.get("python_path"),
                ffmpeg_path=params.get("ffmpeg_path"),
            )
        except (TypeError, ValueError) as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

    def _project_list(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..registry.repository import list_recent

        if params is not None and not isinstance(params, dict):
            raise RpcDispatchError(INVALID_PARAMS, "params must be an object")
        limit = params.get("limit", 20) if isinstance(params, dict) else 20
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise RpcDispatchError(INVALID_PARAMS, "limit must be an integer from 1 to 1000")
        projects = list_recent(self._get_registry(), limit=limit)
        return {"projects": projects, "total": len(projects)}

    def _project_get(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..registry.repository import get_project, mark_project_opened

        if not isinstance(params, dict) or not isinstance(params.get("id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "id is required")
        try:
            mark_project_opened(self._get_registry(), params["id"])
            return get_project(self._get_registry(), params["id"])
        except KeyError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

    def _project_update(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..registry.repository import get_project, update_project

        if not isinstance(params, dict) or not isinstance(params.get("id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "id is required")
        project_id = params["id"]
        updates = {key: value for key, value in params.items() if key != "id"}
        try:
            update_project(self._get_registry(), project_id, **updates)
            return get_project(self._get_registry(), project_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

    def _project_remove(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..registry.repository import get_project, remove_project

        if not isinstance(params, dict) or not isinstance(params.get("id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "id is required")
        try:
            get_project(self._get_registry(), params["id"])
        except KeyError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc
        remove_project(self._get_registry(), params["id"])
        return {"removed": True}

    def _doctor_from_values(
        self,
        *,
        mode: Any,
        python_path: Any,
        ffmpeg_path: Any,
        device_policy: Any,
    ) -> dict[str, Any]:
        from ..runtime.doctor import run_doctor

        if not isinstance(mode, str) or not isinstance(device_policy, str):
            raise RpcDispatchError(INVALID_PARAMS, "runtime mode and device policy must be strings")
        if python_path is not None and not isinstance(python_path, str):
            raise RpcDispatchError(INVALID_PARAMS, "python_path must be a string")
        if ffmpeg_path is not None and not isinstance(ffmpeg_path, str):
            raise RpcDispatchError(INVALID_PARAMS, "ffmpeg_path must be a string")
        try:
            return run_doctor(
                mode=mode,
                python_path=python_path,
                ffmpeg_path=ffmpeg_path,
                device_policy=device_policy,
            )
        except (RuntimeError, ValueError) as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

    @staticmethod
    def _require_compatible(report: dict[str, Any]) -> None:
        if report.get("compatible") is True:
            return
        issues = report.get("issues")
        detail = "; ".join(str(item) for item in issues) if isinstance(issues, list) else "unknown incompatibility"
        raise RpcDispatchError(INVALID_PARAMS, detail)

    def _runtime_doctor(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..registry.repository import get_project

        if params is None:
            values: dict[str, Any] = {}
        elif isinstance(params, dict):
            values = params
        else:
            raise RpcDispatchError(INVALID_PARAMS, "params must be an object")

        project: dict[str, Any] = {}
        project_id = values.get("project_id")
        if project_id is not None:
            if not isinstance(project_id, str):
                raise RpcDispatchError(INVALID_PARAMS, "project_id must be a string")
            try:
                project = get_project(self._get_registry(), project_id)
            except KeyError as exc:
                raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc
        return self._doctor_from_values(
            mode=values.get("runtime_mode", project.get("runtime_mode", "embedded")),
            python_path=values.get("python_path", project.get("python_path")),
            ffmpeg_path=values.get("ffmpeg_path", project.get("ffmpeg_path")),
            device_policy=values.get("device_policy", "auto"),
        )

    def _runtime_select(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..registry.repository import get_project, update_project

        if not isinstance(params, dict) or not isinstance(params.get("project_id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "project_id is required")
        project_id = params["project_id"]
        runtime_mode = params.get("runtime_mode")
        if runtime_mode not in {"embedded", "external"}:
            raise RpcDispatchError(INVALID_PARAMS, "runtime_mode must be 'embedded' or 'external'")
        try:
            get_project(self._get_registry(), project_id)
        except KeyError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

        python_path = params.get("python_path")
        ffmpeg_path = params.get("ffmpeg_path")
        if runtime_mode == "external":
            report = self._doctor_from_values(
                mode=runtime_mode,
                python_path=python_path,
                ffmpeg_path=ffmpeg_path,
                device_policy=params.get("device_policy", "auto"),
            )
            self._require_compatible(report)
        else:
            python_path = None
            ffmpeg_path = None

        try:
            update_project(
                self._get_registry(),
                project_id,
                runtime_mode=runtime_mode,
                python_path=python_path,
                ffmpeg_path=ffmpeg_path,
            )
            return get_project(self._get_registry(), project_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

    def _dataset_open(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..dataset.index import DatasetIndex, _serialize_document, default_index_path
        from ..dataset.version import UnsupportedVersionError
        from ..registry.repository import get_project

        if not isinstance(params, dict) or not isinstance(params.get("project_id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "project_id is required")
        try:
            project = get_project(self._get_registry(), params["project_id"])
        except KeyError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

        source_path = project.get("source_path", "")
        try:
            idx = DatasetIndex(db_path=default_index_path())
            doc = idx.get_or_build(source_path)
            idx.close()
        except UnsupportedVersionError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc
        except Exception as exc:
            raise RpcDispatchError(INTERNAL_ERROR, f"Failed to open dataset: {exc}") from exc

        import json
        return json.loads(_serialize_document(doc))

    def _dataset_browse(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..dataset.index import DatasetIndex, default_index_path
        from ..dataset.version import UnsupportedVersionError
        from ..registry.repository import get_project

        if not isinstance(params, dict) or not isinstance(params.get("project_id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "project_id is required")
        try:
            project = get_project(self._get_registry(), params["project_id"])
        except KeyError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

        offset = params.get("offset", 0)
        limit = params.get("limit", 20)
        if not isinstance(offset, int) or offset < 0:
            raise RpcDispatchError(INVALID_PARAMS, "offset must be a non-negative integer")
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise RpcDispatchError(INVALID_PARAMS, "limit must be 1-1000")

        source_path = project.get("source_path", "")
        try:
            idx = DatasetIndex(db_path=default_index_path())
            doc = idx.get_or_build(source_path)
            idx.close()
        except UnsupportedVersionError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc
        except Exception as exc:
            raise RpcDispatchError(INTERNAL_ERROR, f"Failed to browse dataset: {exc}") from exc

        all_episodes = doc.episodes
        page = all_episodes[offset:offset + limit]
        return {
            "total": len(all_episodes),
            "episodes": [
                {
                    "index": episode.index,
                    "length": episode.length,
                    "chunk_index": episode.chunk_index,
                    "file_index": episode.file_index,
                    "tasks": list(episode.tasks),
                    "frame_start": episode.frame_start,
                    "frame_end": episode.frame_end,
                }
                for episode in page
            ],
        }

    def _dataset_validate(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        from ..dataset.validation import validate_dataset
        from ..dataset.version import UnsupportedVersionError
        from ..registry.repository import get_project

        if not isinstance(params, dict) or not isinstance(params.get("project_id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "project_id is required")
        try:
            project = get_project(self._get_registry(), params["project_id"])
        except KeyError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc

        source_path = project.get("source_path", "")
        try:
            result = validate_dataset(source_path)
        except UnsupportedVersionError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc
        except Exception as exc:
            raise RpcDispatchError(INTERNAL_ERROR, f"Validation failed: {exc}") from exc

        return {
            "valid": result.valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }

    def _project_source(self, params: Any) -> tuple[dict[str, Any], "Path"]:
        from pathlib import Path
        from ..registry.repository import get_project

        if not isinstance(params, dict) or not isinstance(params.get("project_id"), str):
            raise RpcDispatchError(INVALID_PARAMS, "project_id is required")
        try:
            project = get_project(self._get_registry(), params["project_id"])
        except KeyError as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc
        source = Path(str(project.get("source_path", ""))).resolve()
        if not source.is_dir():
            raise RpcDispatchError(INVALID_PARAMS, "project source_path is not a directory")
        return project, source

    def _parity_dispatch(self, method: str, params: Any) -> dict[str, Any]:
        import mimetypes
        from pathlib import Path

        from ..parity import (
            dataset_analytics, dataset_episode, dataset_summary, export_annotations,
            list_annotations, map_replay, read_progress, save_annotations,
        )

        project, source = self._project_source(params)
        assert isinstance(params, dict)
        try:
            if method == "dataset.summary":
                return dataset_summary(source)
            if method == "dataset.episode":
                return dataset_episode(source, params.get("episode_index"))
            if method == "dataset.analytics":
                return dataset_analytics(source)
            if method == "progress.read":
                return read_progress(source, params.get("episode_index"), params.get("duration"))
            if method == "replay.map":
                values = params.get("values")
                if not isinstance(values, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    raise ValueError("values must be a numeric array")
                ranges = params.get("joint_ranges")
                if ranges is not None and (not isinstance(ranges, list) or not all(
                    isinstance(item, dict) and isinstance(item.get("min"), (int, float))
                    and isinstance(item.get("max"), (int, float)) for item in ranges
                )):
                    raise ValueError("joint_ranges must contain numeric min/max objects")
                return map_replay(source, values, params.get("joint_names"), params.get("urdf_joints"), ranges)
            if method == "annotations.list":
                return list_annotations(source, params["project_id"], params.get("episode_index"))
            if method == "annotations.save":
                return save_annotations(source, params["project_id"], params.get("episode_index"), params.get("atoms"))
            if method == "annotations.export":
                output = params.get("output_path") or project.get("output_path")
                if not isinstance(output, str) or not output:
                    raise ValueError("output_path is required")
                return export_annotations(source, params["project_id"], Path(output))
            if method == "media.resolve":
                relative = params.get("relative_path")
                if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                    raise ValueError("relative_path must be a non-empty relative path")
                candidate = (source / relative).resolve()
                if source not in candidate.parents or not candidate.is_file():
                    raise ValueError("media path is outside the project or missing")
                mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                if not (mime.startswith("image/") or mime.startswith("video/")):
                    raise ValueError("path is not a supported media file")
                return {"path": str(candidate), "mime_type": mime}
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc
        raise RpcDispatchError(METHOD_NOT_FOUND, "Method not found", {"method": method})

    def _hub_dispatch(self, method: str, params: Any) -> dict[str, Any]:
        from pathlib import Path
        from ..parity.hub import hub_import, hub_info, hub_search

        if not isinstance(params, dict):
            raise RpcDispatchError(INVALID_PARAMS, "params must be an object")
        try:
            if method == "hub.search":
                return hub_search(params.get("query"), params.get("token"))
            if method == "hub.info":
                return hub_info(params.get("repo_id"), params.get("token"), revision=params.get("revision"))
            destination = params.get("destination")
            if not isinstance(destination, str) or not destination:
                raise ValueError("destination is required")
            return hub_import(params.get("repo_id"), Path(destination), params.get("token"),
                              revision=params.get("revision"))
        except (TypeError, ValueError, OSError) as exc:
            # Hub helpers intentionally produce redacted messages.
            raise RpcDispatchError(INVALID_PARAMS, str(exc)) from exc


def _write(stream: BinaryIO, message: dict[str, Any]) -> None:
    stream.write(encode_frame(message))
    stream.flush()


def run_server(input_stream: BinaryIO, output_stream: BinaryIO) -> int:
    server = RpcServer()
    _write(
        output_stream,
        notification("server.ready", {"protocolVersion": PROTOCOL_VERSION}),
    )
    while True:
        try:
            message = read_frame(input_stream)
        except JsonPayloadError as exc:
            _write(output_stream, error_response(None, PARSE_ERROR, "Parse error", str(exc)))
            continue
        except FrameTooLargeError as exc:
            _write(output_stream, error_response(None, INVALID_REQUEST, "Invalid Request", str(exc)))
            return 2
        except FrameProtocolError as exc:
            _write(output_stream, error_response(None, INVALID_REQUEST, "Invalid Request", str(exc)))
            continue
        if message is None:
            return 0

        try:
            request = parse_request(message)
        except ValueError as exc:
            _write(output_stream, error_response(None, INVALID_REQUEST, "Invalid Request", str(exc)))
            continue

        try:
            result = server.dispatch(request)
            response = success_response(request.id, result)
        except RpcDispatchError as exc:
            response = error_response(request.id, exc.code, exc.message, exc.data)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response = error_response(request.id, INTERNAL_ERROR, "Internal error")
        _write(output_stream, response)

        if server.shutdown_requested:
            _write(output_stream, notification("server.exit", {"code": 0}))
            return 0


def main() -> None:
    raise SystemExit(run_server(sys.stdin.buffer, sys.stdout.buffer))

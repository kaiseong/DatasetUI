"""Validator policy identity and content manifests for the export gate."""

from __future__ import annotations

from datasetui.content_integrity import ContentIntegrityError, dataset_content_manifest

VALIDATOR_POLICY = "datasetui-semantic-relative-deferred-bound-v5"


validation_content_manifest = dataset_content_manifest


__all__ = ["ContentIntegrityError", "VALIDATOR_POLICY", "validation_content_manifest"]

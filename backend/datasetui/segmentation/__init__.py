"""Workspace segmentation (SAM 3.1): keep/remove objects, review, export.

Layers (each imports only from layers above it):
  errors, contract, workflow_contract       request models and errors
  paths, media, source, frames, catalog      I/O and source access
  selection, candidates, engine              mask rules and inference
  sample, preview, export                    job services
  workspace, workflows                       editing state and batches
  api, workflow_api                          HTTP routes
"""

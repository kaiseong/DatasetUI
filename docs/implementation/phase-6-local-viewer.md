# Phase 6 — Registered NAS datasets in Viewer

Phase 6 connects ready entries in the NAS registry to the existing LeRobot
Viewer without exposing or accepting host filesystem paths. The Viewer remains
read-only for registered NAS datasets; profile flags, curation, annotation, and
dataset mutation stay assigned to later phases.

## User flow

1. Open the NAS Library and expand a dataset marked `ready`.
2. Select **Viewer에서 열기**.
3. DatasetUI opens the existing episode Viewer at
   `/~nas/{opaque_dataset_id}/episode_0`.
4. Episodes, Statistics, Frames, Action Insights, and supported 3D Replay use
   the same loading and rendering code as Hugging Face datasets.

The browser route contains only the registry ID. The display uses the registry
name, and the Viewer provides a direct return link to the Library. Annotation,
Filtering, Hugging Face sign-in, and the external Doctor tab are hidden for
this read-only source until their Workbench phases provide safe server-side
contracts.

## File API

`GET|HEAD /api/v1/datasets/{dataset_id}/files/{dataset_relative_file}` serves
Viewer files from the dataset selected by the registry ID.

The API:

- resolves `raw` or `derived` and the registered relative path only on the
  server;
- requires the registry entry to be available and `ready`;
- opens every directory and the final regular file with `O_NOFOLLOW` relative
  to already-open directory descriptors;
- rejects absolute paths, traversal components, backslashes, empty components,
  oversized paths, directory files, and symbolic links;
- streams from the already-open descriptor so a later path swap cannot redirect
  the response;
- supports full, open-ended, suffix, and bounded single HTTP byte ranges;
- returns `416` with the complete file size for invalid or unsatisfiable ranges;
- uses `private, no-store` and never returns an absolute server path.

The common Viewer URL builder maps only the reserved `~nas/{dataset_id}` source
to this same-origin API. Hugging Face URLs keep their original behavior. Even
when the browser has a Hugging Face OAuth token, same-origin NAS requests omit
the `Authorization` header.

## Verification

Automated coverage includes:

- GET, HEAD, complete, bounded, open-ended, and suffix byte responses;
- invalid and multi-range rejection;
- unavailable, unready, and unknown registry entries;
- file and directory symlink rejection;
- traversal and backslash rejection without absolute-path disclosure;
- same-origin URL construction and Hugging Face token isolation;
- preservation of the existing Hugging Face URL contract.

Container and browser acceptance covers the complete Library → Viewer →
Statistics flow using a temporary LeRobot v2.1 dataset with a real MP4 and
Parquet file. It also checks Caddy byte-range forwarding, desktop and mobile
layouts, the missing-dataset error state, browser page errors, and credential
headers.

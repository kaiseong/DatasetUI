# Phase 10 — Relative Action

DatasetUI keeps the stored `action` column absolute. A curation recipe may freeze a
relative-action profile that names only the dimensions training should calculate as
`action[i] - observation.state[i]`.

The materializer validates that every selected name exists at the same index in both
features. It does not guess a mapping. Unselected dimensions, including a gripper,
remain absolute. The run manifest records the selected dimensions, absolute
dimensions, formula, and derived statistics; the dataset root remains standard
LeRobot content.

Acceptance evidence:

- schema migration 7 stores the recipe and immutable snapshot configuration;
- strict API models reject enabled profiles without dimensions and duplicates;
- materialization never rewrites the stored action values;
- dimension-order mismatch and non-finite relative values fail the job;
- the curation UI exposes the profile and marks recipes that use it.

# Segmentation implementation / approved scope

Goal: protected task environment segmentation with black non-task pixels, same geometry; optional image background, reusable masks, fixed+wrist camera templates, batch review and one derivative LeRobot output on RTX6000.

Constraints used: both fixed and moving wrist views require per-camera policy; fixed workspace ROI reusable only with explicit setup confirmation; wrist coordinates never transferred between episodes; RGB pixel masks do not imply calibrated geometry; no intrinsics/extrinsics/depth assumed. Source datasets/action/state/instruction/time axis unchanged. Training image statistics optional and explicitly deferred by existing policy.

Implementation lanes: core inference/render/multi-preview export; persisted templates and batch workflow; frontend editor/review; leader integrates task safety/scheduler, validates official SAM runtime and server deployment.

Cleanup/refactor plan: run existing segmentation tests first; decouple background from mask identity with compatibility tests; group same-object corrections into one tracking history; delete unsafe independent correction union; keep immutable mask manifests and source/approval binding; stage shared-shard rewrites once.

Tests: black default, image legacy, mask reuse without inference, wrong-owner/stale/corrupt rejection, correction propagation, fixed/wrist coordinate policies, idempotent batch creation and dispatch recovery, queued cancellation, all approvals required, multi-camera shared shard export, statistics deferred, frame/data invariants, frontend tests and fixture Playwright, real GPU proof only if licensed checkpoint available.

Deployment: preserve current RTX source differences, private backup and idle guard, scoped image builds; no interruptions of active upload/processing. No checkpoint license bypass/download without granted access. Do not claim GPU accuracy based on mocked tests. Representative 10-episode quality/training comparison are rollout gates, not silently fabricated evidence.

## Completion evidence (2026-10-01 KST)

Deployed to https://192.168.0.3/augment. Backend/GPU release segmentation-v1 and frontend segmentation-v2; existing application source differences preserved. Private source/config/SQLite backup taken after idle checks. Redis, Caddy, credential store unaffected; later coordinate fix restarted web only.

Main backend597passed1skipped, actualRTX-stage backend425passed, frontend275passed, scopedlint/typechecks/productionbuild passed. Non-root final GPU image completed four12-frame real640x480cases: mixedtext+point, pointonly, pointkeyframe, boxkeyframe; forward/backward correctionverified. Live realAPI/UI passed desktop/mobile with exactsource-image/pointer-overlay bounds andzeroJS/HTTPerrors/mutations. Temporarydevserver stopped. Detailed evidence: .omx/state/segmentation/implementation-result.json and ralph-progress.json.

Remaining rollout gates: representative fixed/wrist10-episode quality, occlusion/reappearance/contact review and learning benefit comparison. No masks automatically approved and no real derivative dataset or HF upload created during verification.

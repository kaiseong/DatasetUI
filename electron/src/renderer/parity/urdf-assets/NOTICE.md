# Vendored robot URDF provenance

These three kinematic descriptions are vendored so DatasetUI replay remains deterministic and offline-capable. Mesh files are intentionally not redistributed; DatasetUI uses only link/joint topology, origins, axes, and joint types.

Upstream bucket: `https://huggingface.co/buckets/lerobot/robot-urdfs/resolve`

| Local file | Upstream object | SHA-256 |
| --- | --- | --- |
| `so101_new_calib.urdf` | `so101/so101_new_calib.urdf` | `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c` |
| `openarm_bimanual.urdf` | `openarm/openarm_bimanual.urdf` | `9ce52e175b2781888e334c40ef6721ca6c4c6cfe8992e7d2e44b9ca16cbb71c5` |
| `g1_body29_hand14.urdf` | `g1/g1_body29_hand14.urdf` | `289493ba824a78c9a838cb310cc356d10d3acd4025d0a1f52c3a89700f123e85` |

The model selection and upstream paths match the pinned `huggingface/lerobot-visualize-dataset` implementation at revision `d724744111cae6feb9a2194e607e71749813a97a`. The upstream application is Apache-2.0 (see `third_party/visualize_dataset.NOTICE.md`). Robot model files retain their upstream notices and source comments; downstream users must also observe any original hardware-model license terms applicable to those descriptions.

# DatasetUI V1 implementation roadmap

The handoff is tracked as fourteen phases, numbered 0 through 13. Each phase
must preserve the immutable-originals boundary and finish with its own tests
before the next dataset mutation capability is enabled.

| Phase | Deliverable                                                      | State    |
| ----- | ---------------------------------------------------------------- | -------- |
| 0     | Freeze and document the existing Viewer baseline                 | Complete |
| 1     | Compose, HTTPS, NAS mount boundaries, and client CA setup        | Complete |
| 2     | Profiles, durable jobs, SQLite, Redis/RQ, and API boundary       | Complete |
| 3     | NAS dataset registry, safe discovery, and version classification | Complete |
| 4     | Library, profile selector, and Jobs user interface               | Complete |
| 5     | Hugging Face listing, revision selection, and managed import     | Complete |
| 6     | Open registered NAS datasets in the existing Viewer              | Complete |
| 7     | Per-profile flags and reusable curation recipes                  | Complete |
| 8     | Trim, edit, and train/eval split workflows                       | Complete |
| 9     | Task and VQA annotation workflows                                | Complete |
| 10    | Relative-action derivation                                       | Planned  |
| 11    | Merge, lineage, and collision handling                           | Planned  |
| 12    | Integrity checks, export gate, and v3.0 to v2.1 conversion       | Planned  |
| 13    | NAS/Hugging Face/Ubuntu-PC delivery and full V1 acceptance       | Planned  |

This repository document is the executable roadmap. If the external handoff
is recovered and differs in grouping, update this table without weakening the
accepted safety or verification contracts.

# 백엔드 코드 구조 (2026-10-02, `release-20261002-curation-v1`)

기능별 패키지로 정리했다. 각 패키지 안에서는 위 계층이 아래 계층만 가져다 쓴다.

| 패키지 | 모듈 | 역할 |
|---|---|---|
| `dataset_io/` | `files`, `tables`, `source`, `video`, `stats`, `publish` | 여러 기능이 같이 쓰는 데이터셋 읽기·쓰기·공개 도구 (심볼릭 링크 차단 읽기, parquet, `DatasetSource`, 영상 코덱·구간 자르기, stats.json, derived 공개와 원본 변경 감지) |
| `curation/` | `materialize` → `writer` → `language`, `trim` | 레시피를 새 데이터셋으로 만드는 작업 |
| `merge/` | `job`, `schema`, `normalization`, `writer` | 여러 데이터셋 합치기 |
| `delivery/` | `transfer`, `workflow` | NAS 내보내기, HF 업로드, PC 복사 |
| `validation/` | `run`, `structure`, `video`, `statistics`, `integrity`, `reporting` | 검증과 Export Gate |
| `segmentation/` | (기존) | SAM 3.1 세그멘테이션 |

- 진행률 보고는 `job_progress` (`JobProgressReporter`, `report_progress`).
- 옛 모듈(`transforms`, `merge_writer`, `delivery_workflow`, `validation_*`, `merge_progress`)은 없다. RQ 작업은 `datasetui.tasks.run_job`으로 들어오므로 대기 중 작업에 영향 없음.
- 동작 변경 없음: 로컬 598개, processor 이미지(torch 포함) 618개 통과. `test_materialize_v3_rebuilds_shards_and_episode_offsets`는 공식 LeRobot 런타임이 있는 이미지에서 개편 전부터 실패(테스트 데이터에 task 문자열 인덱스 없음).
- 백업 `~/kgs/datasetui-backup-20261002-curation-091419`(이전 HEAD `68391b1`), 이전 태그 `~/kgs/DatasetUI-release-20261002-curation/compose.override.before.yaml`.

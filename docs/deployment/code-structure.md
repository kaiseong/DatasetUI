# 백엔드 코드 구조 (2026-10-02, `release-20261002-core-v1`)

기능별 패키지로 정리했다. 각 패키지 안에서는 위 계층이 아래 계층만 가져다 쓴다.

| 패키지          | 모듈                                                                                                   | 역할                                                                                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_io/`   | `files`, `tables`, `source`, `video`, `stats`, `publish`                                               | 여러 기능이 같이 쓰는 데이터셋 읽기·쓰기·공개 도구 (심볼릭 링크 차단 읽기, parquet, `DatasetSource`, 영상 코덱·구간 자르기, stats.json, derived 공개와 원본 변경 감지) |
| `curation/`     | `materialize` → `writer` → `language`, `trim` → `stationary`                                           | 레시피를 새 데이터셋으로 만드는 작업 (정지 구간 자르기 포함)                                                                                                           |
| `merge/`        | `job`, `schema`, `normalization`, `writer`                                                             | 여러 데이터셋 합치기                                                                                                                                                   |
| `delivery/`     | `transfer`, `workflow`                                                                                 | NAS 내보내기, HF 업로드, PC 복사                                                                                                                                       |
| `validation/`   | `run`, `structure`, `video`, `statistics`, `integrity`, `reporting`                                    | 검증과 Export Gate                                                                                                                                                     |
| `statistics/`   | `exact`, `episode`, `visual`, `deferred`, `output`                                                     | 새 출력의 통계(stats.json): 정확 통계, 에피소드별 집계, RGB 영상 통계, 상속 통계, 출력 통계 재계산                                                                     |
| `official/`     | `runtime`, `sources`, `operations`                                                                     | 고정된 공식 LeRobot 런타임 어댑터 (런타임 신원 확인, 원본 사본 제공, 공식 v3 연산). `lerobot_source_lock.json`도 여기 있음                                             |
| `relative/`     | `actions`, `artifacts`                                                                                 | relative action 학습용 통계와 산출물                                                                                                                                   |
| `segmentation/` | (기존)                                                                                                 | SAM 3.1 세그멘테이션                                                                                                                                                   |
| `database/`     | `core` + `profiles`, `jobs`, `huggingface`, `datasets`, `curation`, `validation` / `errors`, `schema`  | SQLite 레지스트리. `Database`는 영역별 mixin을 합친 클래스라 `from datasetui.database import Database`는 그대로. 마이그레이션은 `schema.MIGRATIONS`, 실행은 `core`     |
| `api/`          | `system`, `profiles`, `jobs`, `huggingface`, `trash`, `datasets`, `curation`, `processing`, `delivery` | HTTP 라우트. 각 모듈이 `register(router, ctx)`로 등록, 공유 값은 `context.RouterContext`                                                                               |

- 진행률 보고는 `job_progress` (`JobProgressReporter`, `report_progress`).
- 옛 모듈(`transforms`, `merge_writer`, `delivery_workflow`, `validation_*`, `merge_progress`)은 없다. RQ 작업은 `datasetui.tasks.run_job`으로 들어오므로 대기 중 작업에 영향 없음.
- 동작 변경 없음: 로컬 598개, processor 이미지(torch 포함) 618개 통과. `test_materialize_v3_rebuilds_shards_and_episode_offsets`는 공식 LeRobot 런타임이 있는 이미지에서 개편 전부터 실패(테스트 데이터에 task 문자열 인덱스 없음).
- 백업 `~/kgs/datasetui-backup-20261002-curation-091419`(이전 HEAD `68391b1`), 이전 태그 `~/kgs/DatasetUI-release-20261002-curation/compose.override.before.yaml`.

## 2026-10-02 통계·공식 런타임 묶기 (`release-20261002-statistics-v1`)

- `exact_statistics`, `episode_statistics`, `visual_statistics`, `deferred_statistics`, `output_statistics` → `statistics/`; `official_operations`, `lerobot_runtime`, `processing_sources` → `official/`. 런타임 고정 파일 `lerobot_source_lock.json`을 `official/`로 이동(런타임이 자기 파일 옆에서 읽음).
- 공식 엔진 환경에서만 실패하던 테스트 2개 수정: v3 테스트 데이터를 공식 형식(task 문자열 인덱스, meta/episodes 인덱스, 2축 action)으로, 프로세스 정리 테스트의 시작 직후 SIGTERM 경합 제거. 운영 코드 변경 없음.
- processor 이미지(공식 엔진, torch 포함) 619개 전부 통과(반복 실행 모두 통과), 로컬 598개 통과.
- 배포: 백업 `~/kgs/datasetui-backup-20261002-statistics-115415`(이전 HEAD `4d24af0`), 이전 태그 `~/kgs/DatasetUI-release-20261002-statistics/compose.override.before.yaml`.
- 이후 배포 방식: `rtx6000`이 docker 그룹에 들어가 sudo 없이 배포 스크립트(`~/kgs/DatasetUI-release-*/rollout_*.sh`)를 실행한다.

## 2026-10-02 핵심 모듈 분리 (`release-20261002-core-v1`)

- `relative_actions`/`relative_artifacts` → `relative/`, `stationary_trim` → `curation/stationary`.
- `database.py`(3,255줄) → `database/` 패키지: 메서드 87개를 그대로 영역별 mixin으로 이동(AST 동일).
- `api.py`(1,744줄) → `api/` 패키지: 라우트 54개 정의를 그대로 이동(AST 동일). 라우트 42개의 경로·메서드·핸들러·상태코드·응답모델과 경로 매칭 결과, OpenAPI 스키마가 이전과 완전히 같음을 확인.
- 이 배포부터 sudo 없이 자동 배포(`rollout_core.sh`). 백업 `~/kgs/datasetui-backup-20261002-core-133357`(이전 HEAD `9f12bc6`).
- 운영 확인 시 주의: Caddy는 `DATASETUI_HOST` 이름으로만 응답한다. `https://127.0.0.1/...`는 빈 200을 주므로 `curl --resolve "$DATASETUI_HOST:443:127.0.0.1" https://$DATASETUI_HOST/...`로 확인.

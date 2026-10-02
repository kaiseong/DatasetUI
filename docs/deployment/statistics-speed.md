# 통계 재계산 속도 개선 (결과 동일)

출력 통계를 다시 계산하는 모든 작업이 대상이다(병합, 큐레이션, 트림, 관절 오프셋 증강 등. `statistics/output.write_output_statistics`와 `recompute_visual_statistics`).
**계산 결과는 바이트 단위로 같고**, 시간만 줄었다.

## 측정 (RTX6000, JIMTOF_TEST_0 관절 오프셋 증강 K=1 → 318 에피소드, 카메라 3개, 9.4 GB)

| 단계 | 이전 | 이후 |
|---|---|---|
| 전체 수치 통계 | 10 s | 10 s |
| RGB 표본 통계 | 795 s | 76 s |
| 에피소드별 수치 통계 | 334 s | 10 s |
| **합계** | **1139 s** | **97 s** |

이전 이미지와 새 이미지로 같은 출력의 `meta/stats.json`과 `meta/episodes/*.parquet`를 만들어 sha256을 비교했고, 모두 같았다.
작은 데이터셋에서도 lerobot-v3·legacy 엔진과 converter(Python 3.10) 이미지에서 전역, 에피소드별, 부분 선택 결과의 float 비트가 모두 같았다.

## 바뀐 것

1. **RGB 표본마다 버리던 계산을 없앴다** (`statistics/visual.py`).
   - 이전에는 표본 프레임마다 lerobot `get_feature_stats`를 불렀다. 이 함수는 5000-bin 분위수 히스토그램까지 만드는데, 그 결과(q01..q99)는 곧바로 버렸다.
   - 게다가 같은 프레임을 전역용과 에피소드용으로 두 번 계산했다.
   - 이제 `_image_moments`가 같은 reshape와 같은 numpy 축약으로 min/max/mean/std/count만 계산하고, 프레임마다 한 번만 측정해서 공유한다.
   - 카메라(파일)마다 첫 프레임은 원래 lerobot 호출과 비트 단위로 비교한다. 다르면 그 파일은 원래 호출로 되돌아간다(lerobot 버전이 바뀔 때의 안전장치).
2. **영상 파일을 병렬로 디코딩한다**.
   - 스레드 수는 `min(8, CPU/4)`이고 `DATASETUI_STATS_DECODE_WORKERS`로 바꿀 수 있다. PyAV는 디코딩과 변환 중에 GIL을 놓는다.
   - 누적은 원래 파일 순서대로만 한다. 그래서 순서에 민감한 lerobot float 집계도 순차 실행과 같다.
   - 진행 콜백(취소 확인)은 단일 처리일 때 64프레임마다, 병렬일 때 1초마다 계속 호출된다. 취소되면 남은 디코딩도 바로 멈춘다.
3. **에피소드별 수치 통계를 한 번에 읽는다** (`statistics/exact.recompute_numeric_statistics_by_episode`).
   - 이전에는 에피소드마다 전체 parquet를 다시 읽어서 시간이 에피소드 수 × 데이터 크기로 늘었다.
   - 이제 한 번 읽으면서 각 에피소드에 이전과 같은 record batch 조각을 같은 순서로 넣는다. 분위수는 그 에피소드의 행만으로 계산한다.

## 검증

- `backend/tests/test_exact_statistics.py`: 한 번에 계산한 결과가 에피소드별 개별 계산과 JSON 문자열까지 같다(batch 경계가 에피소드를 자르는 경우 포함). 행 수 검사.
- `backend/tests/test_visual_statistics_writer.py`: `_image_moments`가 lerobot `get_feature_stats`와 비트 단위로 같다(lerobot이 있는 이미지에서 실행). 다를 때 원래 호출로 되돌아가는지.
- `backend/tests/test_episode_statistics.py`: 병렬(2)과 순차(1) 결과가 같다. 병렬 디코딩 중에도 취소가 된다. 기존 취소·누락 프레임 테스트.

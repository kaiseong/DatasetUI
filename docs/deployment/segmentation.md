# 작업 영역 Segmentation

`분할·증강` 탭에서 작업에 필요한 픽셀을 남기고 별도 LeRobot 데이터셋을 만듭니다. 기본은 **보존 영역 밖을 검게 처리**하는 방식이며, 사진 배경은 개별 미리보기에서 선택할 수 있습니다. 자르기·확대·리사이즈 없이 원래 해상도와 위치를 유지합니다.

## 사용 순서

1. 원본 데이터셋과 대표 에피소드·카메라를 선택합니다. 보드, 그리퍼, 플러그, 케이블처럼 보존할 대상을 각각 다른 **추적 객체 번호**로 지정합니다.
2. `black plug`, `board`, `robot gripper` 같은 대상별 짧은 프롬프트와 점·박스로 초기화합니다. 같은 객체 번호의 다른 프레임 지시는 해당 객체의 보정입니다. 반드시 남겨야 하는 지지면·접촉 경계는 수동 보호 영역으로 보완합니다.
3. 카메라 템플릿을 저장하고 에피소드 × 카메라 범위를 선택합니다. 기본 선택은 앞쪽 최대 10개 에피소드이며 대표성은 직접 확인해야 합니다. 한 묶음은 최대 **256개 영상**입니다. 일괄 생성은 검은 배경으로 시작합니다.
4. 각 결과를 열어 **텍스트가 찾은 객체 후보를 직접 선택**하고 저장된 마스크로 다시 렌더링합니다. 후보를 자동 확정하거나 모든 결과를 자동 승인하지 않습니다. 원본·합성 영상·마스크를 확인한 뒤 각 영상을 승인합니다.
5. 선택한 모든 영상이 성공하고 승인되어야 **하나의 전체 데이터셋**으로 내보낼 수 있습니다. 실패·미승인 영상은 자동 제외하거나 원본으로 대체하지 않습니다. 미선택 에피소드·카메라는 원본 그대로 포함됩니다.

대기·실행 상태는 모든 탭 하단 작업 진행 상황에서 확인합니다. `대기열 등록 재시도`는 등록 실패를 복구하는 기능이며, 추론 자체가 실패한 영상은 열어 설정을 보정한 뒤 새 미리보기를 생성합니다. 원본 데이터셋은 수정하지 않고 새 이름으로 생성하며 기존 출력은 덮어쓰지 않습니다.

## 영역 지정과 검토 기준

| 구분           | 동작과 주의점                                                                                                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 고정 카메라    | 같은 설치·화각임을 확인한 촬영 묶음에서만 좌표를 재사용합니다. 프레임을 지정하지 않은 수동 보호 영역은 전 프레임에 적용합니다.                                                                                           |
| 손목 카메라    | 에피소드마다 텍스트로 다시 탐지합니다. 템플릿의 점·박스·브러시·수동 영역 좌표는 다른 에피소드에 복사하지 않습니다. 개별 검토에서 만든 수동 영역은 지정 프레임만 보호합니다.                                              |
| 브러시 보정    | 현재 프레임에는 전체 브러시와 반경을 반영합니다. 다른 프레임 추적에는 최대 64개 대표 점을 사용하므로 브러시 모양이 그대로 전파된다고 보장하지 않습니다. 후보가 모호하면 대상 안쪽의 양성 점·좁은 박스로 먼저 구분합니다. |
| 수동 보호 영역 | 선택 객체 마스크와 합쳐 반드시 남깁니다. 고정 카메라는 수동 영역만으로도 GPU 추론 없이 처리할 수 있습니다. 손목 카메라 일괄 템플릿에는 재탐지할 텍스트 대상이 필요합니다.                                                |
| 품질 신호      | 빈 마스크·화면 전체 선택·마스크 면적 급변 프레임을 표시합니다. 이는 검토 보조 휴리스틱이지 모델 신뢰도나 품질 보증이 아닙니다. 모든 프레임이 빈 결과는 승인을 차단합니다.                                                |

얇은 케이블, 플러그 끝, 포트 내부, 그리퍼 접촉부와 가림 후 재등장을 직접 확인하세요. 사진 배경은 2D 합성이므로 손목 이동·다중 카메라의 3D 시점 일치를 보장하지 않습니다. 실제 카메라 보정값·깊이 기반 재구성은 이번 기능에 포함하지 않습니다.

## 캐시·승인·통계

- 배경이나 후보 선택만 변경하면 저장된 객체 마스크로 다시 렌더링하며 SAM을 다시 실행하지 않습니다. 원본·객체 지시·보정·보호 영역이 바뀌면 이전 마스크를 재사용하지 않고 후보 선택도 다시 요구합니다.
- 원본 식별값과 미리보기·설정·결과 파일 해시에 승인을 연결합니다. 편집하면 해당 결과의 승인을 해제하며, 출력 등록 및 작업 시작 시 승인 상태를 다시 검증합니다.
- **검증을 마치고 이미 실행 중인 내보내기는 후속 편집만으로 자동 취소되지 않습니다.** 중단하려면 작업 진행 상황의 일반 취소 기능을 사용합니다.
- `분포 통계 재계산`은 기본 OFF입니다. 생략 시 통계를 미재계산으로 표시하며 기존 통계가 새 영상의 분포를 대표한다고 간주하지 않습니다. 학습 전 최종 데이터·변환 기준으로 `norm_stats`를 계산하세요. Relative 정규화 아티팩트가 있는 데이터는 필요한 통계를 재계산합니다.
- 프레임 수·FPS·타임스탬프·action·state·instruction은 유지합니다. 미재계산은 구조·영상 무결성 검사를 생략한다는 의미가 아닙니다.

## RTX6000 실행 구성

실행 대상은 `rtx6000@192.168.0.3`의 DatasetUI입니다. 별도 GPU 작업자 **1개**가 추론을 처리하며, 캐시 렌더링·수동 보호 영역만의 처리는 IO 대기열을 사용합니다. 체크포인트는 읽기 전용으로 제공하며 실제 추론에서 파일 해시와 CUDA 상태를 검사합니다. 설정 표시만으로 추론 성공이 보장되지는 않습니다.

| 항목                 | 고정 값                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 공식 SAM 3.1 소스    | [`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`](https://github.com/facebookresearch/sam3/tree/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7) |
| 체크포인트           | `sam3.1_multiplex.pt`                                                                                                                |
| 모델 저장소 revision | `daa63191845a41281374e725f4c9e51c7a824460`                                                                                           |
| SHA-256              | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`                                                                   |
| 호스트 파일          | `/data/datasetui/models/sam3.1.pt`                                                                                                   |
| Docker CDI 장치      | `nvidia.com/gpu=0`                                                                                                                   |
| CDI 명세             | `/etc/cdi/datasetui-nvidia.yaml`                                                                                                     |
| CDI hook 경로        | `/opt/datasetui-nvidia-cdi`                                                                                                          |

호스트의 dpkg 작업이 중단된 상태여서 필요한 CDI 도구·hook만 별도 경로에 추출한 구성입니다. 이 기능 배포를 위해 광범위한 패키지 복구나 Docker 데몬 재시작을 수행하지 않았습니다. **NVIDIA 드라이버를 변경하면 CDI 명세를 다시 생성하고 컨테이너 GPU 접근과 실제 추론을 재검증해야 합니다.** 일반 설치·라이선스·옵션은 아래 일반 배포 설정을 참고하세요.

### 고정된 SAM 소스의 호환성 보정

`backend/datasetui/segmentation/engine.py`의 제한된 어댑터가 위 고정 revision의 API 차이를 처리합니다. 공식 소스 파일이나 모델 가중치를 변경하지 않으며, 소스 pin을 갱신할 때 아래 회귀 테스트와 실제 GPU 검증을 다시 수행해야 합니다.

- 공식 builder가 결합 체크포인트를 개별 tracker에 먼저 읽는 경로를 피하고, 최종 detector+tracker 모델에는 검증된 로컬 체크포인트를 **`strict=True`**로 다시 읽습니다. 누락·예상 밖 키·형상 불일치를 정상 로드로 취급하지 않습니다.
- multiplex `init_state`가 받지 않는 `offload_state_to_cpu` 인수를 전달하지 않고, 종료 시 autocast 상태를 복원합니다.
- 텍스트/박스와 점 입력은 공식 API의 별도 단계로 전달합니다. 개념 추적을 영상 전체에 먼저 실행한 뒤 같은 세션·객체를 보정합니다. 박스 보정은 의미 탐지를 초기화하지 않는 객체별 모서리 점 입력을 사용합니다.
- 공식 객체 마스크 병합 함수는 프레임 캐시 키가 없으면 유효한 보정 결과까지 버립니다. 점 전용 입력에는 **빈 캐시 컨테이너만** 준비해 실제 추론 결과를 병합하게 합니다. 마스크를 생성·복사하거나 검출 누락을 가짜 결과로 채우는 조치가 아닙니다.
- 재전파는 마지막 보정 키프레임에서 시작합니다. 마지막 영상 프레임이 시작점이면 역방향부터 실행해 한 프레임 처리만으로 추적이 완료됐다고 판정하는 upstream 동작을 피합니다. 양방향 결과는 이전 마스크와 합집합하지 않고 해당 객체의 최종 결과로 교체합니다.

이 보정은 추론·추적 API가 제대로 실행되도록 하는 호환성 조치이지, 모든 객체·프레임의 정확도나 가림 후 복원을 보장하지 않습니다. 빈 마스크와 면적 급변 경고 및 사용자 검토는 계속 적용됩니다.

## 검증 범위와 학습 도입

프런트엔드 회귀 테스트와 `scripts/smoke_segmentation_workflow.py`는 템플릿 저장, 후보 선택, 캐시 재렌더링, 개별 승인·무효화, 누락 없는 내보내기, 데스크톱·모바일 UI를 검증합니다. 브라우저 테스트는 가짜 데이터/API를 사용하므로 **실제 분할 품질의 증거는 아닙니다**.

고정·손목 카메라, 가림·재등장·접촉이 포함된 대표 영상으로 실제 GPU 품질·처리 시간·메모리를 먼저 확인하세요. 현 기능 구현만으로 전체 파일럿의 분할 품질이나 학습 성능 향상은 입증되지 않았습니다. 원본만 학습한 경우와 원본+증강을 섞은 경우를 비교하고 평가 데이터는 원본으로 유지합니다. **같은 원본 에피소드와 모든 증강 파생본은 반드시 같은 train/validation/test 분할에 배치**해 증강본을 통한 평가 누출을 막습니다.

## 일반 배포 설정

아래는 RTX 호스트의 별도 CDI 구성이 아닌 기본 Compose 배포를 활성화하는 절차입니다.

DatasetUI keeps segmentation optional. The default Compose deployment does not
start a GPU worker, download model weights, or request Hugging Face credentials.
Enabling the `segmentation` profile adds one worker that consumes only the
`gpu` queue.

## Requirements

- Linux host with an NVIDIA GPU, a current driver, Docker, and NVIDIA Container
  Toolkit providing a valid native Docker CDI specification. The worker requests
  `${DATASETUI_GPU_DEVICE:-nvidia.com/gpu=0}`; it does not depend on a Docker
  `nvidia` runtime registration.
- A locally obtained [SAM 3.1 multiplex checkpoint](https://huggingface.co/facebook/sam3.1).
  Access is gated on Hugging Face and is subject to Meta's
  [SAM License](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE).
  Review and accept those terms before obtaining or using the weights.
- Sufficient local disk, GPU memory, and shared memory for the selected video.
  DatasetUI bounds submitted videos with `DATASETUI_SEGMENTATION_MAX_FRAMES`, but
  does not claim that every accepted video fits every GPU.

The container pins Meta's SAM 3 source to commit
[`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`](https://github.com/facebookresearch/sam3/tree/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7).
It uses the upstream `build_sam3_multiplex_video_predictor` entry point with an
explicitly supplied local checkpoint, so the worker never falls back to an
automatic weight download.

## Checkpoint setup

1. Download the licensed SAM 3.1 multiplex checkpoint yourself after access is
   granted. Do not place Hugging Face tokens in Compose environment files.
2. Create the model directory and save the file as `sam3.1.pt`:

   ```text
   ${DATASETUI_MODEL_ROOT:-./.runtime/models}/sam3.1.pt
   ```

3. Calculate the file's SHA-256 locally:

   ```bash
   sha256sum .runtime/models/sam3.1.pt
   ```

4. Set only the resulting hexadecimal digest in the deployment environment:

   ```dotenv
   DATASETUI_SAM3_SHA256=<64 lowercase hexadecimal characters>
   DATASETUI_MODEL_ROOT=./.runtime/models
   DATASETUI_SEGMENTATION_MAX_FRAMES=3600
   ```

The model directory is mounted read-only into the API and workers. The API uses
the mount only to report whether segmentation is configured. The GPU worker
recomputes and compares the complete checkpoint SHA-256 before importing SAM or
loading the model. A missing file, digest mismatch, unavailable CUDA runtime, or
missing GPU fails the job explicitly; there is no CPU or synthetic-mask
fallback.

## Start and verify

Build and start the optional worker together with the normal stack:

```bash
docker compose --profile segmentation build worker-gpu
docker compose --profile segmentation up -d
docker compose ps
docker compose logs worker-gpu
```

Open the DatasetUI segmentation capability endpoint after startup. A configured
response confirms only that a checkpoint path and digest are present; actual
CUDA availability and checkpoint validity are checked by the GPU job itself.
Segmentation output is model inference and must be reviewed before approval and
export. Deployment does not imply an accuracy or safety guarantee.

To disable segmentation, stop the profiled worker. The rest of DatasetUI remains
available without it:

```bash
docker compose stop worker-gpu
```

## 2026-10-01 — 검토 수정: 배치 연결·객체 단위 Instruction·재등장·원본 보존 내보내기

### 동작 변경

- **배치 검토 수정본 연결**: AUGMENT는 선택 토큰으로 미리보기를 요청하므로 요청에는 지문이 없습니다. 배치 연결·승인·내보내기 검증은 작업자가 확인한 지문(`effective_preview_job`)으로 비교합니다. 대기·실행 중인 토큰 미리보기도 연결할 수 있으며 승인 시 다시 검증합니다.
- **객체 단위 Instruction**: 객체마다 Instruction 하나가 모든 프레임에 보입니다. `후보 찾기`를 누른 프레임이 탐지 프레임이 되며, 이전 프레임의 점·Box는 그 프레임에 남습니다. 후보 그룹 보정 대상 선택은 모든 프레임에서 가능합니다.
- **추적 샘플**: 탐지 프레임이 아닌 프레임의 샘플은 탐지 프레임~현재 프레임 사이 최대 약 24프레임 클립으로 SAM이 추적한 뒤 현재 프레임만 보여줍니다. 추적 샘플의 후보는 다른 객체의 후보 참조로 쓸 수 없습니다.
- **늦은 등장·재등장**: 텍스트 객체가 프롬프트 프레임에 없으면 실패하지 않고, 이후 SAM이 기준 점수 이상으로 새로 잡은 트랙을 후보로 받습니다. 후보를 선택한 객체는 선택 수만큼의 트랙과 **동시에 보이지 않는** 새 트랙만 같은 객체의 재등장으로 이어 붙이고 `재등장 추정` 검토 신호를 남깁니다. 이는 휴리스틱이므로 해당 프레임을 직접 확인해야 합니다.
- **자동 선택**: 객체의 후보가 하나뿐이거나 후보들이 시간상 겹치지 않으면(같은 객체가 나갔다 들어온 경우) 자동 선택하고 `자동 선택` 검토 신호를 남깁니다. 명시적 선택이 항상 우선합니다. `후보 찾기` 결과가 하나뿐이어도 자동으로 체크합니다.
- **배치 재탐지**: 템플릿을 그린 대표 에피소드는 템플릿을 그대로 사용합니다. 다른 에피소드에서 텍스트 없는 객체가 빠지면 배치에 경고를 표시합니다.
- **검토 모드 객체 저장**: 저장된 객체 목록은 현재 확정 초안에서 만들어지므로 배치 항목의 다른 템플릿 객체가 저장 시 사라지지 않습니다. 텍스트 객체는 후보 미선택 상태로도 저장할 수 있으며 미리보기에서 선택(또는 자동 선택)합니다.
- **메타정보 변경**: 원본 `info.json`이 바뀌어 저장 객체가 stale이면 `저장된 객체 비우고 새로 시작`으로 복구합니다.
- **오류 안내**: 사용자가 고칠 수 있는 거부 사유(`SegmentationGuidanceError`)는 `segmentation_guidance` 코드와 실제 문구로 표시합니다.

### 내보내기 영상 정책 (`source-codec-new-files-v1`)

- 분할하지 않은 에피소드·카메라의 영상은 **원본 파일 바이트 그대로** 복사됩니다. 공유 shard를 다시 인코딩하지 않습니다.
- v3.0: 승인된 에피소드·카메라마다 새 `videos/{key}/chunk-XXX/file-YYY.mp4`를 만들고 그 에피소드의 `videos/{key}/chunk_index·file_index·from_timestamp·to_timestamp`만 바꿉니다. 기존 shard에는 해당 구간의 원본 프레임이 참조되지 않은 채 남습니다.
- v2.x: 해당 에피소드 영상 파일만 교체합니다.
- 코덱은 원본 카메라 코덱(AV1은 `libsvtav1`), 원본 GOP(`video.g`, 기본 2), 높은 품질(SVT-AV1 crf 18, preset 8)입니다. 실제 JIMTOF 30초 기준 원본 9.2 MB → 13.1 MB, 원본 디코드 대비 PSNR 44.2 dB. 이전 무손실 RGB H.264는 135.7 MB(약 15배)였고 브라우저에서 재생되지 않았습니다.
- `info.json`의 코덱 정보는 바뀌지 않습니다(같은 코덱).

### 성능

- 원본·미리보기 전체 해시는 파일 식별값(장치·inode·크기·mtime·ctime)이 같으면 프로세스 내 결과를 재사용합니다. 쓰기·교체·메타데이터 변경이 있으면 ctime이 바뀌어 다시 해시합니다.
- 미리보기 조회(GET)와 승인 요청은 원본 전체 해시를 하지 않습니다. 내보내기 등록은 미리보기 수와 관계없이 원본을 한 번만 검증하고, 내보내기 작업이 복사 전후로 다시 검증합니다.
- GPU 작업자는 `rq.worker.SimpleWorker`와 `DATASETUI_SAM3_REUSE_PREDICTOR=1`로 실행해 검증된 모델을 작업 간에 유지합니다. 실패·시간 초과 작업은 모델을 버리고 다음 작업에서 다시 로드합니다. 체크포인트 식별값이 같으면 SHA-256 재계산을 생략합니다.
- GPU 큐도 IO 작업 시간 제한(`DATASETUI_IO_JOB_TIMEOUT_SECONDS`)을 사용합니다(기존 900초 제한으로 긴 에피소드가 실패할 수 있었음).

### 검증 범위

- 백엔드: 593 passed(수정 전 583), 수정 전과 동일한 환경 의존 수집 오류 8건(`torch`·`git` 미설치) 외 실패 없음. 새 테스트: 토큰 미리보기 연결, 늦은 등장·재등장, 자동 선택, 추적 샘플, 배치 재탐지 경고, v2.1/v3.0 내보내기(원본 바이트 보존·메타데이터·코덱·검증).
- 프런트엔드: bun test 285 pass, TypeScript app/test 통과, ESLint 오류 없음.
- **RTX6000 배포·실검증 (2026-10-02)**: 이미지 `release-20261001-review-v1`(web/api/processor/converter/segmentation, 현재 이미지 위 overlay), 백업 `/home/rtx6000/kgs/datasetui-backup-20261001-review-225333`, release `/home/rtx6000/kgs/DatasetUI-release-20261001-review`(`review-manifest.json`, 이전 override `compose.override.before.yaml`). 유휴 확인 → RQ suspend → API 중지·재확인 → API/web → **RQ resume 후** 작업자 재시작, 재시작 횟수 0.
- 검증 프로필 `Segmentation 검토 수정 검증 20261001`, `JIMTOF_TEST_0` 에피소드 0 front(1,198프레임), 원본 읽기 전용·내보내기 없음: 텍스트 `board` 샘플 10.0초(모델 로드 포함, 후보 1개 0.61) → 동일 샘플 1.0초(모델 재사용) → 프레임 0 후보를 선택한 프레임 90 추적 샘플 3.0초(24프레임 클립, 마스크 13,819 px) 모두 성공.
- GPU 작업자는 유휴 상태에서도 모델을 유지해 약 20 GB(98 GB 중)를 점유합니다. GPU를 다른 작업과 나눠야 하면 `DATASETUI_SAM3_REUSE_PREDICTOR`를 비워 작업마다 로드하도록 되돌릴 수 있습니다.
- 아직 실데이터로 확인하지 않은 것: 카메라 밖으로 나갔다 들어오는 객체의 재등장 이어붙이기, 전체 에피소드 미리보기·배치 연결·새 방식 내보내기(분할 에피소드 새 파일 + 메타데이터 재지정).
- 배포 시 RTX `compose.yaml`의 `worker-gpu` 명령·환경과 `Dockerfile.segmentation`을 함께 반영해야 SimpleWorker/모델 재사용이 적용됩니다.

### 2026-10-02 후속 배포

- `release-20261002-reid-v1`: 이미 선택한 후보를 다시 찾을 때(추적 샘플·전체 미리보기) 탐지 기준을 절반(최소 0.05)으로 완화하고 IoU로 식별합니다. 기준 점수 경계의 후보가 재인코딩 점수 흔들림으로 "일치하는 후보 없음"이 되던 문제를 고쳤습니다. 이미 후보를 고른 객체를 다른 프레임에서 다시 탐지하면 대체 확인을 받고, 다른 부위는 새 객체로 추가하도록 안내합니다. 기존 객체 저장 후 새 객체로 자동 이동하지 않습니다.
- `release-20261002-coverage-v1`: 미리보기마다 객체별 감지 프레임(`object_coverage`)을 기록하고 누락 구간을 `object_gap` 검토 신호로 표시합니다. 배치 목록은 실패·후보 선택 필요·객체 누락 항목을 "다시 작업할 항목만 보기"로 모아 보여줍니다. 이 기능 이전에 만든 미리보기는 누락 정보가 없습니다.
- `release-20261002-progress-v1`: 샘플·후보 찾기·에피소드 미리보기는 단계별 예상 작업량으로 **전체 진행률**(`progress.overall`)을 기록하고, SAM 단계는 실제 추적 프레임 수로 진행합니다. 화면은 전체 진행률과 경과 시간으로 남은 시간을 추정합니다. 배치는 끝난 항목 평균 소요 시간으로 전체 남은 시간을 보여줍니다(GPU 작업자 1개 순차 처리 가정). 실측: JIMTOF 1,198프레임 미리보기 140초, 50초 시점 예상 145초 남음 → 실제 90초(초반 과대 추정, 진행될수록 수렴). 원본 전체 해시 확인 구간(첫 실행 약 35초)은 계획 수립 전이라 퍼센트 없이 표시됩니다.

### 2026-10-02 구조 개편 배포 (`release-20261002-refactor-v1`)

- 브랜치 `refactor/segmentation`(GitHub): 세그멘테이션 백엔드를 계층형 `backend/datasetui/segmentation/` 패키지로, 편집기를 `src/components/workbench/segmentation/` 컴포넌트·훅으로 분리. 동작 변경 없음(테스트 기준선 동일, 스크린샷 픽셀 동일).
- 의존성은 그대로라 기존 이미지에서 코드 폴더만 삭제 후 교체(`/app/datasetui`, `/app/src`)해 옛 모듈이 남지 않게 빌드. 이미지 내 옛 모듈(`segmentation_api` 등) 없음 확인.
- 운영 소스(`~/kgs/DatasetUI`)는 이 브랜치를 체크아웃한 상태. 백업 `~/kgs/datasetui-backup-20261002-refactor-074630`(이전 HEAD `3b9f601`), 이전 이미지 태그 `~/kgs/DatasetUI-release-20261002-refactor/compose.override.before.yaml`.
- 실 GPU 검증(개편 전과 동일): 샘플 10.0초 → 재사용 1.0초 → 프레임 90 추적 샘플 3.0초(13,819 px), JIMTOF 1,198프레임 미리보기 140초, 객체 감지 1198/1198.

### 2026-10-02 local-dev 선별 반영 (`release-20261002-port-v1`)

- 브랜치 `port/local-dev-20261002`: 배포되지 않았던 local-dev 작업 중 아직 유효한 것만 선별 반영.
  - curation·merge·v2.1 변환이 작업 시작 시 원본 트리의 파일 크기·수정 시각(stat만, 내용 해시 없음)을 기록하고 공개 직전에 다시 비교해 작업 중 원본이 바뀌면 중단. 목록 지문은 `meta/info.json` 해시 그대로 유지해 플래그·레시피 키와 스캔 속도에 영향 없음(NAS 2,152개 파일 기준 0.5초).
  - 라이브러리 스캔에서 심볼릭 링크·특수 파일이 있는 데이터셋은 invalid 처리.
  - HF 이름 검사를 가져오기·업로드 공통 규칙으로 통일(끝 `_` 허용).
  - 주석 임시저장을 프로필·데이터셋 버전별로 분리하고 로드·저장 중 수정 유실 방지.
- 반영하지 않은 것(운영에 이미 더 새로운 방식이 있음): HF 저장소 자동 삭제, 끊긴 외부 전송 자동 재시도 금지, 작업 행 결과 링크, 전체 파일 해시 지문.
- 백업 `~/kgs/datasetui-backup-20261002-port-081204`(이전 HEAD `fd0b492`), 이전 이미지 태그 `~/kgs/DatasetUI-release-20261002-port/compose.override.before.yaml`.

### 2026-10-02 보정 대상 후보 선택을 시각화 (`release-20261002-member-v1`)

- 문제: Instruction에서 후보를 고른 객체는 Labeling(점·박스·브러시)이 어느 후보를 보정할지 "보정할 그룹 내 후보"를 골라야 하는데, 고르지 않으면 클릭이 조용히 무시되고 안내는 화면 맨 아래에만 나왔다. 후보가 하나여도 직접 골라야 했다.
- 변경: 드롭다운 대신 후보 썸네일 버튼(`segmentation/member-picker.tsx`).
  - 후보가 하나면 자동으로 그 후보로 표시하고 바로 찍을 수 있다(저장 형식은 이전과 같이 후보 미지정, 추론 시 그 후보에 연결).
  - 여러 개인데 고르지 않으면 선택 영역이 강조되고, 프레임 위에 "보정할 후보를 먼저 고르세요" 안내와 금지 커서가 나온다. 이때 클릭하면 선택 영역으로 스크롤된다.
- 검증: `scripts/smoke_segmentation_member.py`(단일 후보 자동, 그룹 차단→선택 후 찍힘, 저장 형식, 다시 열기), 기존 `smoke_segmentation_confirm.py` 통과.

### 2026-10-02 경계 여유(px) 설정 (`release-20261002-margin-v1`)

- 출력 배경 옆에 "경계 여유 (px)"(0–64, 기본 0)를 추가했다. 남길 객체는 그만큼 더 넓게 남기고, 제거할 객체는 더 넓게 지운다.
  - 여유는 어느 객체에도 속하지 않은 픽셀만 차지한다. 두 여유가 겹치면 제거가 우선이고, 객체 자체는 깎지 않는다(`selection.retained_mask`, `grow_mask`: 원형 팽창).
  - 현재 프레임 샘플, 미리보기 영상, 검토 신호, 내보내기에 똑같이 적용된다. 카메라 템플릿에 저장되어 일괄 작업에도 전달된다.
- 렌더링 설정이라 SAM 마스크 입력(`mask_inputs`)에서 빠진다. 값만 바꾸면 저장된 마스크로 다시 렌더링한다(재승인 필요).
- 0이면 spec·템플릿 직렬화에서 생략되어 기존 recipe hash가 바뀌지 않는다.
- 검증: 백엔드 626 통과(운영 processor 이미지), 프론트 293 통과, `smoke_segmentation_confirm.py`(입력→미리보기 요청에 값 포함), `smoke_segmentation_member.py` 통과.

### 2026-10-02 카메라 한 화면 설정 (`release-20261002-cameras-v1`)

- "설정할 카메라" 선택을 없앴다. 01 단계에서 모든 카메라의 편집기(원본과 범위·전체 적용)를 세로로 쌓아 한 번에 설정한다.
  - 순서는 front, right, left, 그 밖의 카메라는 원래 순서(`orderWorkflowCameras`). 02 일괄 처리의 카메라 목록도 같은 순서이고 기본 선택은 첫 카메라(front)다.
  - 카메라마다 객체·작업 영역은 따로 저장되고, 템플릿에는 지시가 있는 카메라만 들어간다. 검토 중에는 해당 항목 하나만 보인다.
- 화면만 바뀐 배포라 web만 다시 띄웠다(백엔드 이미지는 같은 이미지에 태그만 추가).
- 검증: 프론트 294 통과, `scripts/smoke_segmentation_cameras.py`(세 카메라 순서, 카메라별 독립 저장, 템플릿 구성, 모바일 가로 스크롤 없음), `smoke_segmentation_confirm.py`, `smoke_segmentation_member.py` 통과.

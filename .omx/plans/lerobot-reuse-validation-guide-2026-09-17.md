# DatasetUI 공식 LeRobot 재사용·동등성 검증 계획

작성: 2026-09-17 / 상태: 계획 초안 — 구현·설치·배포·실데이터 변경 없음

## 1. 요구사항

- 같은 목적의 공식 기능이 있고 지원 버전·보존 계약을 만족하며 독립 검증을 통과하면 **공식 함수를 직접 호출하는 방식이 1순위**다. 공식이라는 이유만으로 채택하지 않는다.
- 공식 함수 주위의 UI·진행도·안전 경계·랜덤 선택 같은 확장은 허용한다. 기능이 없거나 필요한 버전을 지원하지 않으면 이유를 기록하고 자체 구현을 유지할 수 있다.
- 코드 참고/재작성과 실제 공식 함수 재사용을 구별한다. upstream module/function/commit/옵션과 실행 증거를 남긴다.
- 기존 자체 결과와 공식 결과를 비교한다. 다만 기존 버그까지 유지하도록 강제하지 않고, 공식 결과도 독립 정답과 요구사항에 비추어 검사한다.
- Trim·Merge·Subset·Train/Eval·형식 변환·증강 Export의 기본은 원본 코덱 보존이다. 지원 불가 시 H.264 자동 폴백은 금지한다.
- 원본과 기존 파생 데이터셋은 수정/삭제하지 않는다. 수정이 필요하면 사용자 지정 이름의 별도 파생 데이터셋으로 만든다.
- 이번 요청은 계획만 수립한다. 아래 단계의 실행·배포 완료를 주장하지 않는다.

공식 우선은 최신 main 자동 추적, 형식 자동 업그레이드, 공식 기본값 무조건 적용을 뜻하지 않는다. 사용자 데이터 보존과 확정 기능 계약이 우선이다.

### 채택 결정 — 사용자 확정 원칙

- 공식 구현이 요구사항·독립 정답·실제 소비자 검증을 통과하면 그대로 재사용한다.
- 결과가 다르면 먼저 입력 오류, 버전/설정 차이, 의도적 알고리즘 차이, adapter 결함, 공식 구현 결함을 분리한다. 불일치만으로 공식 구현을 틀렸다고 판단하지 않는다.
- 공식 구현 결함이 최소 재현 입력과 독립 기대값으로 확인되면 해당 부분을 자체 구현/최소 확장으로 대체한다. 정상인 공식 부분까지 모두 다시 만들지 않는다.
- 공식 기능이 없거나 요구사항을 지원하지 않는 경우도 자체 구현을 허용한다. 기존 자체 구현이 이미 검증 기준을 만족한다면 재사용하고, 틀린 부분만 수정한다.
- 대체 구현에도 동일한 독립 정답·보존 계약·실제 소비자 테스트를 적용한다. 자체 구현이라는 이유로 검증을 완화하지 않는다.
- 판단 근거, 공식 버전·재현 사례, 대체 범위, 통과 테스트를 출처표와 차이 보고서에 기록한다. 원인/정답이 미확정이면 미검증으로 남기며 성공 또는 배포 대상으로 처리하지 않는다.

## 2. 기준 상태와 알려진 차이

작업 트리: /home/kgs/workspace/DatasetUI  
배포 호환 스테이지: /tmp/datasetui-trim-progress-20260915  
서버: rtx6000@192.168.0.3:/home/rtx6000/kgs/DatasetUI

두 로컬 트리 HEAD는 ef0fa9b81c8c5418a06383732fb7ac3b86b2e4dd지만 수정·미추적 파일이 많다. HEAD만으로 실행 코드를 재현할 수 없다. V0에서 파일 manifest와 컨테이너 digest를 고정한다. 이번 계획 턴에서는 서버를 새로 조회하지 않았다.

확인된 구현 차이:

- Curation은 source codec 정책을 전달하지만, Trim 외 Subset/Train/Eval도 같은 재작성 경로를 사용한다. Merge는 별도의 원본 영상 복사 경로다. [L1][L2]
- v3.0→v2.1은 현재 writer의 H.264 기본값을 사용한다. 증강 Export에도 별도 libx264rgb 경로가 있다. 이번 보존 정책의 점검 대상에 포함한다. [L3][L4]
- 수치 통계는 자체 NumPy 계산이다. 영상 통계 생성과 검사가 같은 VideoValidator를 사용하므로 자체 검사 통과만으로 독립 동등성을 입증할 수 없다. [L5]
- 검증 결과의 loader_probe는 datasetui-parquet-video-reader이며 실제 LeRobot loader와 다르다. [L6]
- 랜덤 분할은 episode 단위, floor(N×eval_percent/100+0.5), seed와 명시적 선택 ID를 사용한다. 이 사용자 기능을 보존한다. [L7]
- Relative Action은 저장 action을 절대값으로 유지하는 별도 프로필이다. 학습에 자동 적용됐다고 표시하지 않는다. [L8]

공식 비교 소스 후보는 이전 감사와 같은 huggingface/lerobot@30074f7f1358b3c015ae1750017200e86e9c4eb6다. 즉시 배포에 채택한다는 뜻은 아니다. 이 소스는 dataset version v3.0이며 v2.1 직접 로딩에 제한이 있다. [U1]

## 3. 확정이 필요한 사항

### Q1. 정확한 Trim과 재인코딩 — 제안 방향 수용

앞선 제안에 대한 사용자 응답 “그래”에 따라, 가능한 전체 파일/스트림은 복사하고 정확한 frame 경계에 필요하면 **영향받는 구간 또는 shared shard만 원본과 같은 코덱으로 재인코딩**하는 방향을 적용한다. 구체적인 화질 수준은 아래 quality gate에서 별도 검증한다.

같은 코덱이어도 화질·용량·압축 바이트는 달라질 수 있다. 재인코딩을 금지하면 지원 가능한 보존 경로만 허용하고, 불가능한 요청은 명시적으로 중단한다. 키프레임에 맞춰 절단점을 몰래 이동하지 않는다. 공식의 임의 시각 절단 primitive도 재인코딩 함수다. [U4]

Q1의 허용은 모든 작업의 재인코딩을 허용한다는 뜻이 아니다. 보존 가능한 전체 파일은 복사한다.

### Q2. 실제 학습·평가 소비자 — V0에서 조사, 미확인 사항만 질문

확인표: 소비자 저장소/커밋, LeRobot 또는 포크 버전, 허용 데이터 형식, decoder, policy/chunk_size, normalization 및 relative-action 설정.

pi05/OpenPI/GR00T 이름만으로 소비자 설정을 추정하지 않는다. 실제 환경을 읽기 전용으로 확인하고 불명확한 경로·설정만 사용자에게 묻는다. 확인 전에는 v2.1/v3.0 출력 형식을 보존한다. v2.0/v3.1은 별도 지원 증거가 필요하며, 기존 지원을 몰래 삭제하거나 v2.1을 자동 v3로 바꾸지 않는다. [L9][U1]

Q1의 코덱 보존 방향은 수용됐으며 구체적인 재인코딩 화질은 별도 gate를 적용한다. Q2는 실제 학습 호환성 완료 판정을 제한한다. 다른 검증 설계는 진행할 수 있다.

## 4. 기능별 재사용 방향

| 파트 | 우선 방향 | 비교 기준·유지할 확장 |
| --- | --- | --- |
| Merge | 지원되는 v3에 공식 merge_datasets 사용. concatenate_videos=False, concatenate_data=False 명시. [U2] | 원본 MP4 hash·episode 순서·숫자 데이터·task 의미·offset. lineage/진행도/원본 보호 유지. robot_type 불일치를 단순 이름 변경으로 우회하지 않음. |
| Subset/삭제 표시 반영 | 공식 delete_episodes 또는 명시적 선택 API. [U3] | 동일 episode 목록·행·영상 범위. 원본 삭제 없이 새 출력. |
| Train/Eval | 기존 Flag/랜덤의 명시적 ID를 공식 split_dataset에 전달. [L7][U3] | 공식 fraction 기본값과 랜덤을 직접 비교하지 않음. membership/순서/배타성/완전성 검증. |
| 수동 Trim | DatasetUI 구간 계약 + 공식 reencode_video/encoder config 우선. [L9][U4] | [start,end), 정확한 frame/행 대응, 시간 원점. 공식 end-time 포함/반올림 규칙을 adapter에서 검증. |
| Auto Trim | 직접 해당 workflow는 감사 범위에서 확인되지 않아 구간 탐지 확장 유지. [L9] | 구간 탐지 정답과 추출 정확도를 분리. 앞/뒤 시간 개별 설정·내부 정지 유지·수동 범위 우선. |
| 수치 통계 | 공식 get_feature_stats/compute_episode_stats/aggregate_stats 우선. [U5] | mean/std/min/max/count/q01/q10/q50/q90/q99, dtype·shape·축·샘플·집계 방식. |
| 영상 통계 | 공식 샘플링/통계 primitive + 필요한 frame 추출 확장. [U5][U6] | RGB 정규화·depth 단위·sample IDs/count. recompute_stats 하나로 영상까지 재검사됐다고 주장하지 않음. |
| Task 수정 | 공식 modify_tasks를 격리 staging 복사본에서 호출. [U7] | 문자열 의미/행별 연결, task ID 역매핑. 원본 in-place 변경 금지. |
| VQA/언어 | 지원되는 공식 schema/API 재사용, UI/수동 편집은 확장. | v3.1 fixture/소비자 테스트 전 형식 승격·언어 컬럼 삭제 금지. |
| Relative Action | 공식 processor/통계 API 후보. 절대 action 저장 계약 유지. [L8][U8] | 동일 mask·현재 state·action horizon·padding. action[t]-state[t]와 action[t+k]-state[t] 통계를 구별. |
| v3.0→v2.1 | 감사 소스에 역변환 API 미확인: 자체 확장 허용. [L3][U9] | 숫자·task·frame 대응·codec 보존 + 실제 v2.1 loader. 언어 정보 손실은 명시적 거부. |
| 검사/Export Gate | 공식 loader/schema 검사 추가 + 자체 정밀 검사·안전 경계 유지. [L6][U10] | 로딩/전수 무결성/통계/학습 입력 검증 상태 분리. |
| Viewer/Library/Flag/Jobs/NAS·PC 전달 | 목적이 다른 기존 제품 기능 유지. | 사용자 격리, Range 재생, 이름, 진행도, 취소/재시도 회귀. HF SDK 재사용과 LeRobot 재사용을 구분. |
| SAM3 배경 증강 | SAM3 모델 재사용 + 수정/합성/Export 확장. [L4] | 지정 영역 밖 보존/허용 압축오차, 숫자 보존, codec. 미배포 코드와 운영 코드를 섞어 완료 판정하지 않음. |

공식 일부 함수는 in-place 쓰기 또는 디렉터리 이동/정리를 한다. 원본은 read-only, 쓰기는 독립 staging으로 제한한다. 원본과 inode를 공유하는 writable hardlink 복사는 금지한다. [U7][U9]

v2.1은 호환 공식 버전/기능부터 확인한다. 없으면 자체 adapter를 유지한다. 공식 v2.1→v3.0을 쓰더라도 복사본에만 적용하며, 출력 형식 변경은 별도 결정이다. [U1][U9]

## 5. 영상 보존·화질 계약

| 유형 | 방식 | 합격 기준 |
| --- | --- | --- |
| 전체 파일 보존 | 파일 복사 우선 | 대응하는 모든 MP4 SHA-256 일치, 실제 codec/pix_fmt·영상 범위 일치. |
| container 재구성 | 가능한 stream copy/remux | 파일 SHA는 달라도 됨. 고정 decoder에서 모든 frame identity, frame 수·PTS 관계·codec/extradata의 의미 일치. packet 비교는 demux 표현이 고정된 경우만. |
| 정확한 부분 Trim/mixed shard | Q1 허용 시 필요한 범위만 같은 codec 재인코딩 | 선택 frame identity/순서, 행·영상 동기, 시간·개수·codec/pix_fmt·해상도·FPS 보존 + 별도 화질 gate. |
| 같은 codec encoder 없음/metadata 불일치 | 명시적 실패 | H.264 자동 폴백 없음. 실제/선언 차이와 필요한 조치 표시. |

- 실제 stream에서 codec을 검사한다. decoder 이름(libdav1d 등)과 codec을 구분한다. metadata를 실제 stream과 대조한 후 공식 config로 전달한다. 알 수 없는 source CRF/preset은 unknown으로 남기며 새 설정을 원본 설정이라고 하지 않는다. [L1][U4]
- Split의 shared shard는 일부 episode만 선택되면 공식 경로도 해당 shard를 재인코딩한다. 전체 shard를 복사하고 metadata만 숨기면 제외 영상 payload가 남는다. 이를 삭제/독립 데이터 분리라고 주장하지 않는다. 독립 Train/Eval 출력의 기본은 제외 payload를 내보내지 않는 방식이다. [U3]
- 같은 codec은 동일 encoder/품질/용량/바이트를 뜻하지 않는다. 손실 재인코딩 결과에 원본 decoded hash 일치를 요구하지 않는다.
- no-op Trim 또는 보존 가능한 whole-file/shard 작업에서 불필요한 encode는 0이어야 한다.
- 재인코딩 화질은 고정된 quality_policy_id로 관리한다. 실제 영상의 고정 frame 집합에서 PSNR/SSIM 평균·최저치, 작은 물체/그리퍼 ROI, 색 변화, 크기·시간을 기록한다. 객관 지표만으로 합격하지 않는다.
- V0에서 정상/저조도/빠른 움직임/작은 물체를 포함하는 episode·camera·frame·ROI 목록을 동결한다. 구현 담당 agent가 원본-출력 side-by-side montage(1:1 및 ROI 200% 확대)와 같은 구간의 1배속 비교 영상을 생성한다. 원본 기준은 고정 decoder로 읽은 영상이며, 압축 전 원본이 있다고 가정하지 않는다.
- 시각 체크리스트는 추가 번짐, 색 변화, 물체/그리퍼 경계 소실, frame 순서 오류, 시간적 flicker다. 각 항목을 PASS/FAIL/NOT_VERIFIED와 근거 frame 번호로 기록한다. 의도한 Trim 범위 변화는 열화와 구분한다.
- 결과는 검증 artifact 아래 reports/<comparison_run>/quality/의 metrics.json, review.md, montage와 비교 영상으로 보관한다. 구현 담당 agent가 기술 검토를 수행하고, 최초 실데이터 quality policy 및 codec/encoder 정책 변경의 최종 품질 수용은 사용자가 증거를 보고 결정한다. agent 판정만으로 사용자 수용을 대신하지 않는다.
- 시각 검토 또는 최초 사용자 수용이 없으면 화질 gate는 NOT_VERIFIED이며 V4 완료/배포 승인으로 계산하지 않는다. 승인 뒤 같은 고정 정책의 회귀는 자동 gate로 수행하되, 새 codec·설정·표본 범위는 재검토한다.
- 화질 표본/임계값은 배포 전 확정하며, 실패 후 통과를 위해 낮추지 않는다. 미확정이면 quality gate는 NOT_VERIFIED다. Q1 허용이 특정 화질 수준의 승인까지 뜻하지 않는다.
- mixed codec/pix_fmt/FPS를 카메라 전체 하나의 encoder로 통일하지 않는다. 공식 API로 보존할 수 없는 조합은 지원 불가를 표시한다.
- browser preview는 별도 artifact다. 별도 codec이 필요하면 명시하고 원본/Export 파일을 대체하지 않는다.

## 6. A/B/C + 독립 정답 비교

같은 변경 불가능한 입력과 같은 작업 명세를 사용한다.

- A: 동결된 기존 자체 구현. 운영/개발 snapshot이 다르면 별도 표기.
- B: 고정 upstream의 공식 함수와 명시적 옵션만 사용하는 최소 runner.
- C: 새 adapter/확장 경로. 실제 공식 함수 호출을 trace로 확인.
- O: 손계산 가능한 숫자·frame ID fixture, 별도 PyArrow/고정 decoder 검사. A/B/C의 writer/validator helper를 정답 생성에 재사용하지 않음.

필수 비교: A↔B(기존 평가), C↔B(연결 검증), A/B/C↔O(공통 결함 방지), 입력↔출력(보존 계약), 출력↔실제 소비자(사용 가능성).

C=A는 필수 합격 조건이 아니다. A에 오류가 있으면 수정되어야 한다. 공식 전체 workflow가 없으면 B를 꾸미지 않고 NO_OFFICIAL_EQUIVALENT로 기록하고 primitive 비교와 O를 사용한다.

C=B 역시 공식 결과 B의 정확성이 확인된 범위에서만 합격 기준이다. B의 결함이 확인된 항목에서는 C가 그 결함을 재현하면 실패다. 해당 B 결과는 결함 재현 자료로 남기고, C의 독립 정답 O·사용자 계약·실제 소비자 통과를 채택 기준으로 삼는다. C는 공식 adapter 또는 검증된 자체 대체 구현일 수 있으며 실제 실행 경로를 명시한다.

비교 규칙:

1. 변경 대상이 아닌 action/state/기타 숫자 feature는 물리 dtype·shape·값을 정확히 보존한다. 선언 dtype 불일치/NaN/Inf는 보고하며 검사 통과용 cast/제거 금지.
2. index/episode_index/task_index/timestamp/경로는 작업별 변경 허용 목록과 기대 변환식으로 검사한다. 비교에서 일괄 제외하지 않는다. Merge/whole-episode 선택은 불필요한 시간 재생성을 하지 않는다.
3. Parquet 압축/row group/파일명/JSON 순서/생성 시각과 데이터 의미를 구분한다. comparator가 필드 삭제/dtype 통일로 차이를 숨기지 못하게 한다. 알 수 없는 차이는 실패.
4. 정수/count/집합/shape/task 의미/복사 SHA는 exact. float 통계의 시작 제안값은 현재 rtol=1e-5, atol=1e-7이며 dtype·수치 규모별 독립 오차 분석으로 최종 정책을 고정한다. 실패를 숨기기 위한 tolerance 확대 금지. [L10]
5. 통계 primitive에는 동일 배열/축/episode·chunk 경계를 넣는다. 최종 artifact 비교는 별도다. histogram 근사 분위수 대 exact NumPy, episode-envelope 집계 대 global 분위수는 의미 차이이며 큰 tolerance로 숨기지 않는다. [L5][U5]
6. 상수/큰 offset+작은 variance/극값/작은 dtype/단일 관측값으로 수학적 oracle도 검사한다. 공식 결과가 어긋나면 upstream 후보 결함으로 기록하고 원인을 재현한다. 공식 구현 결함으로 확인되면 해당 범위를 자체 구현으로 대체하고 같은 oracle·회귀·소비자 테스트를 다시 통과시킨다.
7. RGB/depth, sampled-frame count/pixel count, 압축 전 이미지/압축 후 decode를 구분한다. sample IDs와 전처리 배열 hash를 기록한다. 다른 입력 도메인 차이를 통계 함수 결함으로 단정하지 않는다. [U5][U6]

차이는 EXACT / WITHIN_TOLERANCE / LAYOUT_ONLY / INTENTIONAL_SEMANTIC_DIFFERENCE / LEGACY_DEFECT / ADAPTER_DEFECT / UPSTREAM_CANDIDATE_DEFECT / UNSUPPORTED / NOT_VERIFIED로 분류한다.

모든 차이에 feature·episode·frame·경로, 기대/실제, 절대/상대 차이, 적용 규칙, 영향/조치를 남긴다. 의도적 차이는 이유와 영향이 승인된 allowlist에 있을 때만 gate를 통과한다.

## 7. 테스트 묶음·합격 조건

| ID | 범위/fixture | 합격 조건 |
| --- | --- | --- |
| T01 | A 운영/개발, B revision, dependency/환경/source manifest | commit+dirty diff+미추적 해시+image digest로 재현 가능, secret 없음, source 전후 hash 동일. |
| T02 | v2.1/v3.0, multi-episode/task/chunk, scalar/vector/추가 feature | 기대 행·전 feature 정확 일치, task/index/time 변환식 충족, 예상 밖 필드 손실 0. |
| T03 | Merge AV1/H.264 multi-camera/shared shard, 충돌/호환 불일치 | 모든 원본 MP4 SHA 일치, 행/영상 순서·offset 정확. schema/FPS/robot_type 불일치 명확히 거부. |
| T04 | Flag/random, 0/1/50/99/100%·소수 비율, N=1/2/3/7, seed | 동결 ID와 출력 일치, 중복/누락 0, train∩eval=∅, 합집합=선택 집합. 0/100% 빈 dataset 없음. 제외 영상 payload 누출 없음. |
| T05 | manual [s,e), 앞/뒤 시간 독립, 내부 pause/no motion, shared offset, 최소 길이 | 탐지와 추출 각각 정답 일치, 정확한 frame/행 대응. no-op 불필요 encode 0, 경계 오류 묵살 없음. |
| T06 | 실제 AV1/H.264 필수, 지원 선언할 HEVC/VP9, tiny/10-bit/444, 30000/1001 FPS, 긴 GOP/clip, encoder 부재 | codec 변경/drop/duplication/잘못된 seek 0. 지원한 pix_fmt/FPS 보존. Q1/화질 gate 충족, unsupported 명시. |
| T07 | 손계산 stats, 크기 다른 episode, constant/extreme/low variance, RGB 패턴/샘플 경계, 지원 depth/image | 모든 필수 stats key/shape/count 검사. 동일 정의끼리 tolerance 이내; 다른 정의 별도 판정. 생성과 독립 검사. |
| T08 | dtype 불일치, NaN/Inf, 누락 key, 틀린 stats/task/offset, truncated 영상/Parquet | 주입 오류 전부 감지, 기대/실제·위치 제시. 단순 loader 성공을 PASS로 대체하지 않음. |
| T09 | task 교체/중복, 언어 event Trim 경계, relative mask/horizon/padding | 의미 보존/변환식 일치, 원본 action 변경 0, 실제 processor/역변환/chunk 통계 검증. |
| T10 | v3 shared shard→v2.1, AV1/tasks/추가 feature | 숫자·영상 대응/codec 보존, 실제 v2.1 loader 읽기. 지원 불가 언어 명시적 거부. forward roundtrip은 보조 증거. |
| T11 | 소비자별 loader, episode/shard 경계, delta_timestamps/action chunk, normalize/preprocessor, DataLoader batch | shape/dtype/값/영상 대응·finite·split membership 확인. 단순 import/open smoke 불충분. 학습 수렴/실기 안전 보장과 구분. |
| T12 | UI/progress, 취소/재시도/lease loss, disk full/NAS 단절, source 변경/symlink/이름 충돌 | 원본 손상/부분·중복 publish 0. 미검사 표시, count 없으면 indeterminate. engine/policy 변경 시 stale cache 재사용 금지. |
| T13 | flowers_sorting/mirrored 등 대표 실데이터 발췌 | 격리 scratch 비교, production 등록/업로드/기존 출력 교체 없음. 시간·메모리·디스크 측정 및 전수 실행 비용 산정. |

모든 codec을 새로 지원한다는 뜻은 아니다. AV1/H.264는 필수 검증, 나머지는 지원 선언에 맞춰 supported/unsupported를 표시한다. 기존 테스트는 회귀 기반으로 유지하며 공식 동등성 입증으로 오인하지 않는다. [L11][L12][L13]

Train/Eval의 물리적 분리와 통계 정책을 구분한다. 각 출력 stats는 해당 데이터 분포를 설명할 수 있지만, 실제 평가 normalization은 소비자가 요구하는 train 기준을 사용해야 한다. 평가셋 stats를 몰래 train stats로 덮어쓰지 않는다.

원본/mirrored/augmented의 같은 원본 episode가 train/eval 양쪽에 들어갈 위험은 lineage로 보고한다. 기존 episode-random을 group-random으로 자동 변경하지 않으며, 그룹 분할은 별도 확장 결정이다.

## 8. 실행 순서 — 이전 Phase 0~13과 별도인 V0~V7

| 단계 | 작업 | 종료 gate |
| --- | --- | --- |
| V0 기준 동결 | 운영/개발/공식/소비자 버전과 입력 해시, 기능 출처·지원 형식 표, Q1/Q2 반영 | T01, 미확정 항목 가시화. |
| V1 독립 비교 기반 | 손계산/공식 생성/실데이터 fixture, comparator·고정 tolerance·차이 report. comparator 자체 결함 주입 테스트 | T02/T08. 아직 production 교체 없음. |
| V2 통계/loader 기준 | 공식 stats primitive·집계·loader 격리 실행, A/B 차이 분류, normalization 영향 보고 | T07/T11 지원 범위 통과. 정의 차이와 결함 구분. |
| V3 Merge→Subset/Split | 공식 adapter 연결, 명시적 episode 선택, stream preflight/shared shard | T03/T04/T06/T12. 보존·안전·진행도 회귀 없음. |
| V4 Trim/영상 writer | 구간 결정·추출 분리, Q1/codec 정책, converter·증강 Export 별도 경로 점검 | T05/T06/T10, 화질/정확도/codec 각각 판정. 미배포 SAM 자동 배포 금지. |
| V5 Task/Relative/제품 회귀 | 지원 공식 부분 thin adapter, 언어·Viewer·delivery·Export Gate·UI 회귀 | T09/T11/T12. 미확인 소비자를 지원 완료로 표시하지 않음. |
| V6 실데이터/과거 영향 | 발췌부터 shadow 검증, 전수 비용 산정, lineage 기반 과거 출력 영향 목록 | T13, 미해결 보존 위반/adapter 결함 0. 실제 검증과 추정 구분. 과거 데이터/checkpoint/norm stats 자동 수정 없음. |
| V7 별도 요청 범위의 배포 | 검증 diff만 통합, 작업 drain/DB·소스·이미지 백업, 단계 반영·rollback | 배포 이미지 내 fixture/loader/UI smoke, health/worker. main 미배포 변경 전체 복사 금지. |

V2~V5의 각 공식 채택 단계에는 동일한 분기를 적용한다: 검증 통과 → 직접 재사용, 결함 확인 → 필요한 부분 자체 대체 후 재검증, 원인 미확정 → 해당 기능 채택/배포 보류. 공식 채택 자체를 단계의 성공 조건으로 삼지 않는다.

이번에는 V0~V7 계획을 수립했을 뿐 실행하지 않았다. 후속 구현·배포 요청 범위에서 진행한다. rollback은 코드/서비스 대상이며 사용자 데이터 삭제/롤백과 구분한다.

## 9. 구현 가이드

- 검증용 upstream 환경부터 고정하고 기존 API Python/PyAV 환경을 바로 교체하지 않는다. dependency 충돌을 조사하고 필요하면 processing worker 격리라는 최소 구조를 선택한다. 설치는 후속 구현 단계다.
- thin adapter 책임: 입력/옵션 검증 → 공식 함수 호출 → 결과 검사 → 원자적 publish. 공식 엔진을 다시 만드는 generic framework를 추가하지 않는다.
- 공식 결함의 자체 대체는 별도 명시된 실행 경로로 구현한다. 잘못된 공식 결과를 흉내 내거나 성공으로 덮지 않고, 검증된 대체 구현임을 함수 출처·manifest·보고서에 표시한다.
- 기존 merge.py/merge_writer.py/transforms.py/conversion.py/validation*.py/database.py의 기능 경계에서 변경한다. 정확한 새 파일 분리는 V1/V2 결과 뒤 결정한다. jobs/progress/immutable publish 유지. [L1–L13]
- engine/upstream/codec/stats/comparator 정책, selection/seed/source hashes를 provenance와 cache key에 기록한다.
- legacy는 비교 runner·rollback 근거로 보존한다. 검증 완료 후 production에서 조용히 fallback하는 두 번째 엔진으로 남기지 않는다.
- 원본 stats/metadata 오류를 보고하는 검사와 새 파생셋을 만드는 복구 작업을 분리한다. 자동 cast/NaN 제거/stats 덮어쓰기 금지.
- 공식 단계가 count를 제공하지 않으면 단계/경과/heartbeat를 표시한다. 임의 percent를 만들지 않는다. publish 후에만 완료.
- 과거 파생셋은 영향 목록과 재생성 필요 여부를 보고한다. 기존 출력/checkpoint/정규화 파일은 자동 수정하지 않는다.

## 10. 산출물·최종 종료 조건

산출물:

1. 기능 출처표: 기능/버전/공식 함수/확장·자체 이유/검증 상태.
2. A/B/C/O JSON + 읽기 쉬운 차이 보고서: 위치·기대·실제·분류·영향·조치.
3. 영상 보존표: codec/pix_fmt/FPS/SHA 또는 frame 비교/재인코딩 사유/quality policy.
4. stats 정책표: 샘플·축·집계·오차·정의 차이·학습 normalization 영향.
5. 실제 소비자별 smoke 결과, 지원/미지원/미검증 형식 표.
6. 변경 파일·테스트·이미지·과거 출력 영향·rollback/재생성 가이드.

최종 종료: **지원한다고 표시한 기능 전부의 gate 통과, 설명되지 않은 의미 차이 0, 원본 변경 0, 자동 H.264 전환 0**. NOT_VERIFIED/UNSUPPORTED는 성공으로 계산하지 않는다. 버전/codec 변경 시 해당 행을 재검증한다.

## 로컬 근거

- [L1] [Curation 정책](/home/kgs/workspace/DatasetUI/backend/datasetui/transforms.py:186), [writer 기본값](/home/kgs/workspace/DatasetUI/backend/datasetui/transforms.py:456), [codec 검사](/home/kgs/workspace/DatasetUI/backend/datasetui/transforms.py:1390), [배포 동작 문서](/home/kgs/workspace/DatasetUI/docs/deployment/trim-source-codec.md:1).
- [L2] [원본 보존 Merge](/home/kgs/workspace/DatasetUI/backend/datasetui/merge_writer.py:150), [shared shard](/home/kgs/workspace/DatasetUI/backend/datasetui/merge_writer.py:317).
- [L3] [v3→v2.1 호출](/home/kgs/workspace/DatasetUI/backend/datasetui/conversion.py:138).
- [L4] [증강 encoder](/home/kgs/workspace/DatasetUI/backend/datasetui/segmentation.py:1237), [SAM3 모델 호출](/home/kgs/workspace/DatasetUI/backend/datasetui/sam3_engine.py:134).
- [L5] [자체 stats](/home/kgs/workspace/DatasetUI/backend/datasetui/transforms.py:1617), [영상 validator 공유](/home/kgs/workspace/DatasetUI/backend/datasetui/transforms.py:1703).
- [L6] [loader_probe](/home/kgs/workspace/DatasetUI/backend/datasetui/validation.py:90).
- [L7] [랜덤 분할](/home/kgs/workspace/DatasetUI/backend/datasetui/database.py:424), [분할 정책](/home/kgs/workspace/DatasetUI/docs/deployment/semantic-validation-random-split.md:18).
- [L8] [Relative 계약](/home/kgs/workspace/DatasetUI/docs/implementation/phase-10-relative-action.md:1), [현재 계산](/home/kgs/workspace/DatasetUI/backend/datasetui/transforms.py:602).
- [L9] [형식·Trim 계약](/home/kgs/workspace/DatasetUI/docs/implementation/phase-8-transforms.md:3), [구간 결정](/home/kgs/workspace/DatasetUI/backend/datasetui/transforms.py:962).
- [L10] [현재 허용오차](/home/kgs/workspace/DatasetUI/backend/datasetui/validation_statistics.py:364).
- [L11] [Trim codec 테스트](/home/kgs/workspace/DatasetUI/backend/tests/test_trim_source_codec.py:120).
- [L12] [Merge 보존 테스트](/home/kgs/workspace/DatasetUI/backend/tests/test_merge_preserve_video.py:115), [실영상 smoke](/home/kgs/workspace/DatasetUI/scripts/smoke_preserved_merge.py).
- [L13] [stats 테스트](/home/kgs/workspace/DatasetUI/backend/tests/test_validation_statistics.py:55), [분할 테스트](/home/kgs/workspace/DatasetUI/backend/tests/test_curation_api.py:349).

## 공식 근거

아래는 고정 소스의 동작 확인이며 실행 동등성을 입증했다는 의미가 아니다.

- [U1] [dataset version](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_metadata.py#L61), [version gate](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/utils.py#L353).
- [U2] [Merge API](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L271), [호환성](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/aggregate.py#L123), [복사](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/aggregate.py#L502).
- [U3] [Delete](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L107), [Split](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L175), [fraction 선택](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L487), [shared-shard](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L741).
- [U4] [reencode_video](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/video_utils.py#L547), [source config](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/configs/video.py#L340).
- [U5] [quantile 구현](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/compute_stats.py#L31), [feature/episode stats](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/compute_stats.py#L435), [집계 의미](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/compute_stats.py#L581).
- [U6] [recompute_stats 지원 범위](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L1609).
- [U7] [modify_tasks](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L1459).
- [U8] [Relative processor](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/processor/relative_action_processor.py#L40), [chunk stats](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/datasets/dataset_tools.py#L1623).
- [U9] [공식 forward 변환](https://github.com/huggingface/lerobot/blob/30074f7f1358b3c015ae1750017200e86e9c4eb6/src/lerobot/scripts/convert_dataset_v21_to_v30.py#L464).
- [U10] [공식 loader 문서](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3#load-a-dataset-for-training). 문서는 mutable main이므로 실행 API는 고정 소스로 재확인한다.

# 자동 검사·전달 대기열과 공유 Library 작업

## 사용 흐름

- 전달 탭에서 NAS/Hugging Face/PC(키·일회용 비밀번호) 요청을 바로 접수한다.
  연결된 `datasets.delivery_preflight` CPU 작업이 현재 내용의 `export_gate`를
  확인한다. 같은 revision·파일 내용·검사 정책의 통과 기록은 재사용하고,
  기록이 없거나 오래됐으면 전체 검사와 파일 무결성을 다시 검사한다.
- 검사를 통과한 경우에만 원래 전달 작업을 IO 큐에 넣는다. 검사 실패·취소·
  중단은 전달 실패로 표시되며 파일을 전달하지 않는다. 전송 직전과 준비본도
  기존 무결성 검사를 거친다. 대기 중 원본이 변경되면 조용히 다른 내용을
  전송하지 않고 실패한다. 브라우저를 닫아도 API의 5초 복구 루프와 워커의
  완료 처리로 다음 단계가 이어진다.
- Workbench 모든 탭의 하단 **팀 작업 진행 상황**에서 팀 전체 실행·대기 작업을
  확인한다. 펼치면 단계별 실측 진행률, CPU/IO/GPU별 실제 RQ 대기 순번,
  선행 검사 링크, 작업자를 표시한다. 전체 작업의 완료 시간을 추정하지 않는다.
  다른 사용자의 작업은 볼 수 있지만 선택한 사용자 자신의 작업만 취소한다.
- CPU가 3개 실행 중이어도 요청 버튼을 잠그지 않는다. 작업 접수 중 중복 클릭과
  동일한 출력·저장소 충돌만 막는다. CPU 최대 3개 및 리소스 admission 정책은
  기존 adaptive worker 설정을 유지하며 IO/GPU/변환 큐는 별도다.
- NAS Library 정렬은 **최근 등록순 / 오래된 등록순 / 이름 오름·내림차순**이다.
  등록 시각은 `first_seen_at`이며 파일 수정 시각이나 다운로드 시각과 다르다.
  선택을 `datasetui.library.sort.v1`에 저장한다. 검색·필터·정렬은 SQL에서 먼저
  적용하고 100개씩 불러온다. **데이터셋 더 보기**로 500개 초과 목록도 볼 수 있다.

## 영구 삭제 경계

- 팀 허브의 **Hugging Face에서 삭제**는 원격 dataset repository 전체를 삭제한다.
  조직은 `rainbowrobotics`로 고정하고 서버의 쓰기 토큰만 사용한다. 확인창에 나온
  repo ID와 commit SHA를 작업에 고정한다. 워커에서 SHA가 바뀌었거나 권한이
  없으면 삭제하지 않는다. NAS에 내려받은 원본·작업본은 지우지 않는다.
- 동일 원격 저장소의 가져오기·업로드·삭제 충돌은 서버에서 원자적으로 막는다.
  삭제 요청의 idempotency key 재전송은 새 작업을 만들지 않는다. 원격 삭제가
  시작된 후 응답을 잃으면 자동 재시도하지 않으며 사용자가 원격 결과를 확인한다.
  접근 불가/404를 삭제 성공으로 오인하지 않는다. HF 삭제 API에는 commit SHA를
  조건으로 하는 원자적 삭제가 없어, 직전 확인과 삭제 사이 외부 push까지 막을 수는 없다.
- **휴지통 비우기**는 개인 휴지통이 아니라 팀 공유 휴지통이다. 확인창에 표시된
  최대 500개 ID와 fingerprint만 대상이며 이후 들어온 데이터셋은 포함하지 않는다.
  이동·복구 중, revision 변경, 관련 작업 진행, 다른 삭제 예약은 제외한다.
  대상 폴더의 device/inode를 다시 확인하고 descriptor 기반으로 삭제하며 symlink나
  하위 mount를 따라가지 않는다. 원본 영역 전체를 재귀 삭제하지 않는다.
- 대기 취소는 예약을 해제해 복구할 수 있게 한다. 첫 영구 삭제/원격 publish 직전
  기존 finalization guard를 기록하며 이후에는 취소·자동 재실행을 막는다.
  부분 실패는 삭제/실패/제외 개수를 각각 표시한다. 삭제 중 워커가 중단되면
  남은 대상은 `recovery_required`로 남기고 관리자 확인 전 자동 복구/재삭제하지 않는다.

## PC 일회용 비밀번호

- 일반 RQ Redis와 분리된 `credential-store` Redis를 사용한다. host port나 disk
  volume 없이 tmpfs, `--save "" --appendonly no`, 32 MiB/noeviction으로 설정한다.
  API도 저장 전에 persistence 설정을 확인한다.
- SQLite payload, 작업 이벤트, API 작업 응답, RQ 인자에는 비밀번호를 넣지 않는다.
  opaque job ID로 참조하고 워커가 GETDEL로 한 번 소비한다. password 작업의 원시
  예외는 로그와 RQ 실패 traceback에 기록하지 않는다.
- `DATASETUI_CREDENTIAL_TTL_SECONDS` 기본 86400초, 최대 86400초이다. 완료·취소·
  실패·만료 시 정리한다. 메모리 저장소 재시작/TTL 만료로 비밀번호를 잃으면 작업을
  실패 처리하고 다시 입력한 새 요청이 필요하다. 재시작 후 자동 복구를 약속하지 않는다.
- standalone 실행은 `DATASETUI_CREDENTIAL_REDIS_URL`이 필요하다. 일반 작업 Redis와
  같은 URL을 지정하면 거부한다. HF 쓰기 토큰과 SSH known-hosts는 기존 서버 설정이다.

## 업데이트·검증

DB migration 16은 전달의 검사 의존성/준비 상태와 휴지통 삭제 예약만 추가한다.
기존 migration 1–15의 데이터·프로필·작업·recipe 기록을 보존한다.

1. 진행·대기 작업과 최종 게시 상태를 확인한다. 실행 중인 워커를 강제 종료하지 않는다.
2. 기존 이미지/설정과 SQLite online backup을 보관한다. `.env`/토큰을 공개 로그나
   일반 백업 산출물에 복사하지 않는다.
3. `docker compose config --quiet` 검증 후 credential-store를 먼저 기동한다.
4. web/API 및 CPU/IO/converter 워커 코드를 같은 버전으로 맞춘다. CPU admission
   설정/큐 이름을 유지하고 Redis/NAS 볼륨을 재생성하지 않는다. adaptive 설정은
   [기존 운영 절차](adaptive-jobs-and-cancellation.md)를 따른다.
5. health의 schema 16, 목록/정렬, 전체 작업 패널을 확인한다. 실제 HF 저장소나
   NAS 자료를 삭제하는 배포 smoke test는 하지 않는다.

검증: backend pytest, Bun tests, TypeScript, ESLint, Ruff, Next build, Compose config.
브라우저 회귀 스크립트 `scripts/smoke_workflow_features.py`는 모든 `/api/v1` 요청을
일회용 fixture로 가로채며 정렬 저장·더 보기·팀 작업·소유자 취소·삭제 확인창·
고정 삭제 대상·CPU 3/3에서 추가 요청·실측 진행률·모든 탭 및 모바일 패널을 확인한다.
HF/SSH 호출은 모의 클라이언트로 검증한다. 실제 권한·네트워크 전송·원격 삭제는 검증 범위 밖이다.

## 이 컴퓨터의 로컬 반영 (2026-09-30)

- `docker compose --env-file .runtime/workflow-local.env --profile adaptive ...`로
  기동했다. 기존 고정 CPU 워커는 작업·큐가 비었음을 확인한 뒤 정상 종료했다.
- Docker에서 사용 가능한 CPU는 10개라 `DATASETUI_ADAPTIVE_CPU_LIMIT=10.0`을
  적용했다. 동시 실행 상한은 3개이고 기존 4 CPU·8 GiB 예약 및 30% 여유 정책은
  완화하지 않았다. 실제 여유가 부족하면 3개 미만이어도 대기한다.
- 기존 이미지 태그 및 SQLite 백업: `.runtime/data/backups/workflow-20260930/`.
  Redis/NAS 및 Caddy 데이터 볼륨은 유지했다. 로컬 접속 Host는 실제 LAN 주소
  `192.168.0.45`로 맞췄으며, 운영 서버용 `.env.example` 기본값은 바꾸지 않았다. schema 16 및 API/web health를 확인했다.
- 로컬에는 HF 쓰기 토큰이 없어 업로드·원격 삭제 버튼은 설정 전까지 비활성이다.
  PC 일회용 비밀번호 저장소는 기동됐으며 디스크 snapshot/AOF가 꺼져 있다.
- 컨테이너 이름을 수동 변경하지 않고 Compose 배포로 서비스 구성을 맞췄다.

최종 검증: backend **538 passed / 1 skipped**, frontend **265 passed**, 타입 검사·
Ruff·변경 파일 포맷·이미지/Next 빌드 통과. ESLint 오류 0개이며 이번 변경과 무관한
기존 Viewer hook 경고 3개는 남아 있다. 실제 로컬 HTTPS
`https://192.168.0.45/library`에서 fixture 기반 브라우저 시나리오도 통과했다.
브라우저 증거는 `/tmp/datasetui-workflow-browser-final/`에 있다.

## RTX6000 운영 반영 (2026-09-30)

- 실제 운영 접속 주소는 **`https://192.168.0.3/library`**이다. 위 `.45` 주소는
  로컬 검증 환경이며 RTX6000 운영 서비스의 주소가 아니다.
- `/home/rtx6000/kgs/DatasetUI`의 기존 수정 사항에 이번 기능만 3-way merge했다.
  서버에 없던 main 전용 SAM/segmentation 및 augment 화면은 추가하지 않았다.
  기존 NAS 경로, Redis/Caddy 컨테이너, 큐 이름과 CPU/메모리 예약 정책을 유지했다.
- 운영 설정과 코드 및 SQLite 백업은 서버 내부
  `/home/rtx6000/kgs/datasetui-backup-20260930-workflow/`에 비공개로 보관한다.
  비밀번호는 소스·설정·로그 파일에 저장하지 않았다.
- 레지스트리·RQ 모두 활성/대기 작업이 없음을 확인하고 잠시 RQ 배정을 중지한 뒤,
  미리 빌드한 `release-20260930-workflow-v1` 이미지로 API/web 및 모든 워커를 교체했다.
  schema 16을 확인하고 배정을 재개했다. 워커는 유지보수 중 일시적으로 종료·재기동하며,
  배정 재개 후 정상 대기 상태를 확인했다.
- 서버용 검증: backend **345 passed**, frontend **246 passed**, 앱/테스트 타입 검사,
  Ruff 및 RTX6000 API/processor/converter/web 이미지 빌드 통과. 기존 서버에 남아 있던
  취소 동작 기대값·통계 fixture 3개는 원본에서도 실패함을 확인한 뒤 테스트만 갱신했다.
- HF 업로드·삭제 및 PC 비밀번호 전달의 설정 플래그는 모두 활성이다. HF 토큰의 실제
  삭제 권한이나 실제 원격 삭제는 실행해 검증하지 않았다. 비밀번호 Redis는 snapshot/AOF가
  꺼진 별도 메모리 저장소다.
- 실제 RTX6000 HTTPS 프런트에서 6개 운영 탭·모바일 패널 및 모든 fixture 기반 기능
  시나리오가 통과했고 브라우저 오류는 0개였다. 증거는 로컬 검증 산출물
  `/tmp/datasetui-rtx-browser-final/`에 있다. NAS/HF의 실제 자료는 삭제하지 않았다.

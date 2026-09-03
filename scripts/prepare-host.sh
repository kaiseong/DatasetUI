#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
env_file="${1:-${repo_root}/.env}"
production_mode=false

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
  production_mode=true
fi

data_root="${DATASETUI_DATA_ROOT:-${repo_root}/.runtime/data}"
nas_root="${DATASETUI_NAS_ROOT:-${repo_root}/.runtime/nas}"
service_uid="${DATASETUI_UID:-$(id -u)}"
service_gid="${DATASETUI_GID:-$(id -g)}"

if [[ ! "${service_uid}" =~ ^[0-9]+$ || ! "${service_gid}" =~ ^[0-9]+$ ]]; then
  echo "DATASETUI_UID and DATASETUI_GID must be numeric." >&2
  exit 1
fi

if [[ "${data_root}" != /* ]]; then
  data_root="${repo_root}/${data_root}"
fi
if [[ "${nas_root}" != /* ]]; then
  nas_root="${repo_root}/${nas_root}"
fi

if [[ "${production_mode}" == true ]]; then
  if [[ ! -d "${nas_root}" ]]; then
    echo "NAS root does not exist: ${nas_root}" >&2
    exit 1
  fi

  nas_fstype="$(findmnt -n -o FSTYPE -T "${nas_root}" 2>/dev/null || true)"
  case "${nas_fstype}" in
    nfs | nfs4 | cifs) ;;
    *)
      echo "NAS root is not on NFS/CIFS: ${nas_root} (${nas_fstype:-unknown})" >&2
      echo "Mount the NAS on the host before starting DatasetUI." >&2
      exit 1
      ;;
  esac
fi

install -d -m 0755 -o "${service_uid}" -g "${service_gid}" \
  "${data_root}" \
  "${data_root}/cache" \
  "${data_root}/staging" \
  "${data_root}/jobs" \
  "${data_root}/registry" \
  "${data_root}/logs" \
  "${data_root}/redis" \
  "${data_root}/caddy-config" \
  "${data_root}/ssh-known-hosts"

install -d -m 0700 -o "${service_uid}" -g "${service_gid}" \
  "${data_root}/ssh-private"

install -d -m 0755 \
  "${nas_root}/raw" \
  "${nas_root}/derived" \
  "${nas_root}/exports" \
  "${nas_root}/manifests" \
  "${nas_root}/client-tools"

overcommit="$(sysctl -n vm.overcommit_memory 2>/dev/null || true)"
if [[ "${overcommit}" != "1" ]]; then
  echo "WARNING: Redis recommends vm.overcommit_memory=1 (current: ${overcommit:-unknown})." >&2
  echo "Set it persistently on the deployment host before production use." >&2
fi

echo "DatasetUI host directories are ready."
echo "Data root: ${data_root}"
echo "NAS root:  ${nas_root}"
echo "Service identity: ${service_uid}:${service_gid}"

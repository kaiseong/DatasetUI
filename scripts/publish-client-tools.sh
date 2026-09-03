#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
env_file="${1:-${repo_root}/.env}"

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

data_root="${DATASETUI_DATA_ROOT:-${repo_root}/.runtime/data}"
nas_root="${DATASETUI_NAS_ROOT:-${repo_root}/.runtime/nas}"

if [[ "${data_root}" != /* ]]; then
  data_root="${repo_root}/${data_root}"
fi
if [[ "${nas_root}" != /* ]]; then
  nas_root="${repo_root}/${nas_root}"
fi

source_cert="${data_root}/caddy/pki/authorities/local/root.crt"
target_dir="${nas_root}/client-tools"
temporary_cert=""

if [[ ! -r "${source_cert}" ]]; then
  temporary_cert="$(mktemp)"
  trap 'rm -f "${temporary_cert}"' EXIT
  if ! docker compose exec -T caddy \
    cat /data/caddy/pki/authorities/local/root.crt >"${temporary_cert}"; then
    echo "Caddy Root CA is not available from the host or container." >&2
    echo "Start DatasetUI once with 'docker compose up -d' and retry." >&2
    exit 1
  fi
  source_cert="${temporary_cert}"
fi

install -d -m 0755 "${target_dir}"
install -m 0644 "${source_cert}" "${target_dir}/datasetui-root-ca.crt"
install -m 0755 \
  "${repo_root}/deploy/client-tools/install-datasetui-ca.sh" \
  "${target_dir}/install-datasetui-ca.sh"
install -m 0644 \
  "${repo_root}/deploy/client-tools/README.txt" \
  "${target_dir}/README.txt"

echo "DatasetUI client tools published to ${target_dir}"

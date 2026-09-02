#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
source_cert="${1:-${script_dir}/datasetui-root-ca.crt}"
target_cert="/usr/local/share/ca-certificates/datasetui-root-ca.crt"

if [[ ! -f "${source_cert}" ]]; then
  echo "DatasetUI Root CA not found: ${source_cert}" >&2
  exit 1
fi

if ! command -v update-ca-certificates >/dev/null 2>&1; then
  echo "This installer requires Ubuntu/Debian update-ca-certificates." >&2
  exit 1
fi

echo "Installing the DatasetUI Root CA on this computer..."
sudo install -m 0644 "${source_cert}" "${target_cert}"
sudo update-ca-certificates

if [[ ! -f "${target_cert}" ]]; then
  echo "Certificate installation failed." >&2
  exit 1
fi

echo "DatasetUI Root CA installed. Fully restart the browser before connecting."

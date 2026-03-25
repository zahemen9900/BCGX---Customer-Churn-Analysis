#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="${CONDA_ENV_NAME:-bcgx-churn}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: ${ENV_FILE} not found." >&2
  exit 1
fi

echo "Creating/updating conda environment '${ENV_NAME}' from ${ENV_FILE}..."
conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune

echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"

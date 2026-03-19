#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-bcgx-churn}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

python3 "${SCRIPT_DIR}/feature_engineering.py" "$@"

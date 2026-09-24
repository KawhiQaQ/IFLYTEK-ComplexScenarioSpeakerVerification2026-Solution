#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${MIWU_ENV_PREFIX:-$ROOT/.conda/envs/miwu}"
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  CONDA="$(command -v conda)"
else
  printf '%s\n' 'Conda was not found. Install Miniconda/Miniforge or set CONDA_EXE.' >&2
  exit 2
fi

"$CONDA" create -y -p "$PREFIX" python=3.8 pip
"$PREFIX/bin/python" -m pip install --upgrade pip==24.3.1
"$PREFIX/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1+cu121 torchaudio==2.4.1+cu121
"$PREFIX/bin/python" -m pip install -r "$ROOT/requirements/training.txt"
"$PREFIX/bin/python" -m pip install --no-deps -e "$ROOT"
"$PREFIX/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; fix the host GPU before training.")
x = torch.ones(1, device="cuda", requires_grad=True)
x.square().sum().backward()
print("cuda_backward_ok", torch.cuda.get_device_name(0))
PY

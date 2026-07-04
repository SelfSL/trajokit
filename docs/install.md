# Installation notes

Core install: `uv sync --extra dev --extra hf --extra swebench`. Below are the
sharp edges we hit on RTX Pro 6000 (Blackwell, sm_120) + CUDA 13 — most apply to
any sm_120 setup.

## CUDA toolkit
torch cu130 wheels need a CUDA 13.x nvcc for any source-built kernels. Ubuntu 24.04:
install `cuda-toolkit-13-0` (NVIDIA apt repo), then export
`CUDA_HOME=/usr/local/cuda-13.0` and prepend `$CUDA_HOME/bin` to PATH — otherwise
builds pick up an old `/usr/local/cuda` and fail with "CUDA version mismatch (12.0 vs 13.0)".

## flash-attn
No prebuilt wheel for cu13/torch2.11 → source build:
`CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH MAX_JOBS=32 uv pip install flash-attn --no-build-isolation`

## flashinfer (sm_120)
pip wheels predate sm_120: sampler crashes with "requires sm75 or higher", MoE path
with "No supported CUDA architectures [12]". Fix: build from source
(`git clone --recursive`, `TORCH_CUDA_ARCH_LIST="12.0"`, same CUDA_HOME env,
`uv pip install --no-build-isolation <clone>`). Kernels JIT-compile on first use
(minutes, all cores). If you see "flashinfer-cubin version mismatch": uninstall
`flashinfer-cubin` (stale companion of the old pip wheel).

## verl (upstream)
- `TransferQueue==0.1.8` is required but easy to miss.
- Launch with `.venv/bin/python`, NOT `uv run`: Ray's uv runtime-env hook crashes
  on `working_dir=None` (`path_or_uri must be a string`).

## uv quirks
- Corrupted dist-info ("failed to read METADATA"): `uv cache clean <pkg>`,
  `rm -rf site-packages/<pkg>*`, reinstall. Build hooks that install their own
  CUDA deps mid-command can re-trigger this once; just rerun.
- Keep uv cache off the OS drive: `UV_CACHE_DIR=/mnt/raid5/uv-cache`.

## Docker
Move data-root to big storage (`/etc/docker/daemon.json`: `"data-root": ...`) —
SWE-bench images are hundreds of GB.

#!/bin/sh
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_DISABLED_KERNELS=B12xTensorFP8ScaledMMLinearKernel CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
SNAP=$(ls -td /c/Users/njbak/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*/ | head -1)
exec "${VLLM_PY:-/c/vllm-rtx-spark/release/.venv/Scripts/python.exe}" -m vllm.entrypoints.cli.main serve "$(cygpath -w "$SNAP")" --config "$(cygpath -w "$(dirname "$0")/inkling2-mini-test.yaml")"

# Maintaining this fork

This is bambash's long-term fork of [devjeonghwan/vllm-for-rtx-spark](https://github.com/devjeonghwan/vllm-for-rtx-spark).
It is the only vLLM install on the RTX Spark machine: `C:\vllm-rtx-spark\release\.venv`.

| Remote | URL |
|---|---|
| `origin` | `https://github.com/bambash/vllm-for-rtx-spark` |
| `upstream` | `https://github.com/devjeonghwan/vllm-for-rtx-spark` |

Fork-specific changes:

- `patches/vllm/0019-inkling2-mini.patch`: Inkling 2 / Inkling 2 Mini from vLLM main (see `inkling2-backport/README.md`).
- `scripts/repack_wheel.py`: builds this checkout's wheel without the CUDA / Visual Studio toolchain.

## Build and install (Python-only changes)

Patches that only touch Python (like 0019) don't need a native build: `repack_wheel.py` takes
the native extensions, CMake-installed source components and vendored dependencies from the
release wheel whose native inputs match, and everything else from this checkout. It refuses if
the manifest's `vllm`/`platform`/`build` sections, `0001-native-build.patch` or native kernel
sources differ from the base tag.

```powershell
# 1. Prepared source (pinned vLLM + all patches in patches/vllm/manifest.json)
python scripts/build_vllm.py prepare --source-dir C:\src\rtx-spark-src
# 2. Base wheel = latest release whose native inputs match (currently v0.2.0)
gh release download v0.2.0 -R devjeonghwan/vllm-for-rtx-spark -p '*.whl' -D C:\src\releases\v0.2.0
# 3. Repack
python scripts/repack_wheel.py --base-tag v0.2.0 `
    --base-wheel C:\src\releases\v0.2.0\vllm-0.32.0+rtx_spark.0.2.0-cp313-cp313-win_arm64.whl `
    --source-dir C:\src\rtx-spark-src --output C:\src\releases\fork-dist
# 4. Install into a NEW environment, verify, then swap folders
.\scripts\install.ps1 -Python "$env:LOCALAPPDATA\Programs\Python\Python313-arm64\python.exe" `
    -Environment C:\vllm-rtx-spark\release-new\.venv -WheelDirectory C:\src\releases\fork-dist
```

`prepare` refuses to reuse an existing `--source-dir`; delete it (or use a new one) after changing patches.
Bump `PROJECT_VERSION` in `src/version.py` for each install so wheels don't collide.

Native changes (`csrc`, `0001-native-build.patch`, `src/kernels/**/*.cu`, a new vLLM pin) need the
full build: ARM64 Python 3.13, Visual Studio ARM64 C++ tools, CUDA Toolkit 13.4, then
`.\scripts\build-vllm.ps1`.

## Syncing upstream

```powershell
git fetch upstream
git merge upstream/main          # resolve conflicts in patches/vllm/manifest.json by keeping 0019 last
```

If upstream changed files that 0019 touches, regenerate it: prepare a fresh source **without**
0019 (remove it from the manifest temporarily), commit that tree as a base, run
`inkling2-backport/apply_inkling2.py <source>\vllm <vllm-main checkout>`, and write
`git diff --cached --binary` back to `0019-inkling2-mini.patch`. `apply_glue.py` fails loudly on any
anchor that moved.

If upstream ships a new release with changed native code, use that release's wheel as the base.

## Verifying an install

1. `.\.venv\Scripts\python.exe -m vllm._rtx_spark.runtime.check doctor` (the installer runs this).
2. Unit tests: `$env:CUDA_VISIBLE_DEVICES='-1'; $env:VLLM_RTX_SPARK_SOURCE_DIR='<prepared source>'; python -I -X utf8 -B -m unittest discover -s tests -p '*_test.py'`.
   Ten spawn/runtime tests fail on this machine with or without the fork's patches.
3. Serve each model you use and send a real request (Qwen: `C:\vllm-rtx-spark\start-qwen.cmd`; Inkling: `inkling2-backport\serve-test.sh`).

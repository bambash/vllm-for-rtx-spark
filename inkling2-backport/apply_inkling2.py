"""Backport Inkling 2 / Inkling 2 Mini from vLLM main to vLLM 0.32.0+spark.

usage: apply_inkling2.py <vllm package dir> <vllm main checkout>

<vllm package dir> is either an installed site-packages/vllm or the `vllm/` dir of
a prepared source tree. <vllm main checkout> is a vLLM source tree at the commit in
UPSTREAM below. Safe to re-run.
"""
import pathlib, shutil, subprocess, sys

UPSTREAM = "8c17e4692a7cf393e3d1dacf6404fa4d48d993a1"  # vllm-project/vllm main, 2026-11-22
HERE = pathlib.Path(__file__).resolve().parent
dst, main = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve() / "vllm"

# 1. Files taken from upstream as-is.
shutil.rmtree(dst / "models/inkling2", ignore_errors=True)
shutil.copytree(main / "models/inkling2", dst / "models/inkling2",
                ignore=shutil.ignore_patterns("__pycache__"))
for rel in ("renderers/inkling.py", "renderers/inkling_encoding.py", "parser/inkling.py"):
    shutil.copyfile(main / rel, dst / rel)
print("copied upstream inkling2 package, renderers and parser")

# 2. 0.32 adaptations of those files are in apply_glue.py (they are re-copied above,
#    so their edits always apply to the pristine upstream text).

# 3. Glue edits in existing 0.32 files.
sys.argv = [sys.argv[0], str(dst)]
exec(compile((HERE / "apply_glue.py").read_text(encoding="utf-8"), "apply_glue.py", "exec"))

"""Build this checkout's wheel from a released wheel without recompiling native code.

    python scripts/repack_wheel.py --base-wheel <released .whl> --source-dir <prepared source> [--output dist]

The base wheel supplies everything this checkout cannot rebuild without the CUDA /
Visual Studio toolchain: the two core extensions, the CMake-installed source
components (`vllm/third_party`, `vllm_flash_attn`, ...) and the vendored Python
dependencies. This checkout supplies the patched vLLM Python sources (from
`build_vllm.py prepare`), the `src/` and `assets/` payloads and the manifest.

Only valid when the native build inputs are unchanged since the base wheel: the
pinned vLLM commit, platform and build sections of the manifest must match, and
`0001-native-build.patch` must be byte-identical to the one the base was built from
(checked with --base-native-patch, or against the git tag given by --base-tag).
Changes to native code need a full `build-vllm.ps1` build instead.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_vllm as B  # noqa: E402

SKIP_SUFFIXES = (".pyc", ".pyd", ".so", ".pdb", ".lib", ".exp")
STAMP = (2026, 10, 27, 0, 0, 0)


def check_native_inputs(base: zipfile.ZipFile, base_tag: str | None, base_patch: Path | None):
    base_manifest = json.loads(base.read("vllm/_rtx_spark/manifest.json"))
    for key in ("vllm", "platform", "build"):
        if base_manifest[key] != B.MANIFEST[key]:
            raise SystemExit(f"Base wheel was built for a different `{key}` manifest section; do a full build")
    native_patch = B.PATCH_ROOT / "0001-native-build.patch"
    if base_patch is not None:
        old = base_patch.read_bytes()
    elif base_tag is not None:
        old = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{base_tag}:patches/vllm/0001-native-build.patch"])
    else:
        raise SystemExit("Pass --base-tag or --base-native-patch to prove the native build inputs match")
    if old != native_patch.read_bytes():
        raise SystemExit("0001-native-build.patch changed since the base wheel; do a full build")
    if base_tag is not None:
        changed = subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--name-only", base_tag, "HEAD", "--",
             "src/kernels", "assets/python"], text=True).split()
        native = [p for p in changed if p.endswith((".cu", ".cuh", ".cpp", ".h", ".pyd", ".dll"))]
        if native:
            raise SystemExit("Native kernel sources changed since the base wheel: " + ", ".join(native))


def repack(base_path: Path, source: Path, output: Path, base_tag: str | None, base_patch: Path | None) -> dict:
    B.verify_source(source)
    version = B.VERSION
    output.mkdir(parents=True, exist_ok=True)
    base = zipfile.ZipFile(base_path)
    check_native_inputs(base, base_tag, base_patch)
    base_dist = next(n.split("/")[0] for n in base.namelist()
                     if re.fullmatch(r"vllm-[^/]+\.dist-info/WHEEL", n))
    tag = "-".join(B.MANIFEST["build"][k] for k in ("python_tag", "abi_tag", "platform_tag"))
    dist = f"vllm-{version}.dist-info"
    destination = output / f"vllm-{version}-{tag}.whl"
    if destination.exists():
        raise FileExistsError(f"Wheel already exists: {destination}")

    members: dict[str, bytes] = {}
    kept = replaced = added = 0
    # 1. Base wheel, minus its metadata and the payloads this checkout owns.
    owned = ("vllm/_rtx_spark/",)
    vendor = "vllm/_rtx_spark/vendor/"
    for name in base.namelist():
        if name.endswith("/") or name.startswith(base_dist + "/"):
            continue
        if name.startswith(owned) and not name.startswith(vendor):
            continue
        members[name] = base.read(name)
        kept += 1
    # 2. Patched vLLM Python sources.
    for path in sorted((source / "vllm").rglob("*")):
        rel = path.relative_to(source).as_posix()
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in SKIP_SUFFIXES:
            continue
        raw = B.render_vllm_version(B.COMMIT) if rel == "vllm/_version.py" else path.read_bytes()
        if rel in members:
            replaced += members[rel] != raw
        else:
            added += 1
        members[rel] = raw
    # 3. This checkout's runtime payloads (as build_vllm.py wheel lays them out).
    payload = B._SOURCE_CHANGES["payload_files"]
    for folder, prefix in (("src", "vllm/_rtx_spark"), ("assets", "vllm/_rtx_spark/assets")):
        for path, rel in payload(ROOT / folder):
            if folder == "assets" and rel.startswith("python/"):
                continue
            members[f"{prefix}/{rel}"] = path.read_bytes()
    members["vllm/_rtx_spark/manifest.json"] = B._VERSION["manifest_path"]().read_bytes()
    # 4. Metadata: the base's, renamed to this version.
    meta = base.read(f"{base_dist}/METADATA").decode()
    meta = re.sub(r"(?m)^Version: .*$", f"Version: {version}", meta, count=1)
    members[f"{dist}/METADATA"] = meta.encode()
    for name in base.namelist():
        if name.startswith(base_dist + "/") and not name.endswith(("/METADATA", "/RECORD", "/")):
            members[dist + name[len(base_dist):]] = base.read(name)
    members[f"{dist}/licenses/THIRD_PARTY_NOTICES.md"] = (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes()

    records = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            raw = members[name]
            archive.writestr(info, raw)
            digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
            records.append((name, "sha256=" + digest, str(len(raw))))
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(records + [(f"{dist}/RECORD", "", "")])
        info = zipfile.ZipInfo(f"{dist}/RECORD", STAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, buffer.getvalue().encode())
    return {"wheel": str(destination), "version": version, "base_wheel": base_path.name,
            "files": len(records) + 1, "kept_from_base": kept, "python_files_replaced": replaced,
            "python_files_added": added, "native_extensions_recompiled": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-wheel", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True, help="Output of build_vllm.py prepare")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--base-tag", help="Git tag the base wheel was built from (e.g. v0.2.0)")
    parser.add_argument("--base-native-patch", type=Path, help="0001-native-build.patch the base was built from")
    args = parser.parse_args()
    print(json.dumps(repack(args.base_wheel, args.source_dir, args.output, args.base_tag,
                            args.base_native_patch), indent=2))


if __name__ == "__main__":
    main()

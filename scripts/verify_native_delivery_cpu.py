"""Verify local wheel contents and RECORD hashes without importing model code."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


def verify(path, native_sha256=None, source_root=None):
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)) or wheel.testzip() is not None:
            raise RuntimeError("Invalid or duplicate wheel entries")
        record = next(name for name in names if name.endswith(".dist-info/RECORD"))
        for name, digest, size in csv.reader(io.StringIO(wheel.read(record).decode())):
            data = wheel.read(name)
            if digest:
                algorithm, expected = digest.split("=", 1)
                actual = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest()).decode().rstrip("=")
                if actual != expected or len(data) != int(size):
                    raise RuntimeError(f"Invalid RECORD entry: {name}")
        if not any(name.endswith("/LICENSE") for name in names):
            raise RuntimeError("License not included")
        tag = wheel.read(next(name for name in names if name.endswith(".dist-info/WHEEL"))).decode()
        binaries = [name for name in names if name.endswith((".dll", ".so", ".pyd"))]
        if native_sha256:
            binary = "ddo_vmm_allocator/native/win32/ddo_vmm.dll"
            if binaries != [binary] or "Tag: py3-none-win_amd64" not in tag or "Root-Is-Purelib: false" not in tag:
                raise RuntimeError("Native wheel has incorrect binary contents or platform tag")
            digest = hashlib.sha256(wheel.read(binary)).hexdigest()
            manifest = json.loads(wheel.read("ddo_vmm_allocator/native/win32/artifact.json"))
            if digest != native_sha256.lower() or manifest["sha256"] != digest:
                raise RuntimeError("Native artifact hash mismatch")
        elif binaries or "Tag: py3-none-any" not in tag:
            raise RuntimeError("Ordinary wheel unexpectedly includes native binaries")
        if any(name.endswith((".safetensors", ".pyc", ".cpp", ".cu")) for name in names):
            raise RuntimeError("Unwanted model, cache or development source in wheel")
        if source_root:
            expected = {item.relative_to(source_root).as_posix(): item for item in source_root.rglob("*.py")}
            actual = {name for name in names if name.endswith(".py")}
            if actual != set(expected):
                raise RuntimeError("Wheel Python file set differs from current source tree")
            for name, source in expected.items():
                if wheel.read(name) != source.read_bytes():
                    raise RuntimeError(f"Wheel contains stale Python source: {name}")
    return {"wheel": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "entries": len(names), "native_binaries": binaries, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--native-sha256")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.wheel, args.native_sha256, args.source_root), indent=2))


if __name__ == "__main__":
    main()

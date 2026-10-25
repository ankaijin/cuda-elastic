#!/usr/bin/env python3
"""Fetch and verify loom_private assets into the cache outside tracked source.

  fetch_assets.py [--cache DIR] [--verify-only] [NAME ...]

With no names, every asset in assets.json is handled. A cached file is reused
only when its size and SHA-256 match; anything else is refetched and verified.
Exit status: 0 when every requested asset verifies, 3 otherwise.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REGISTRY = Path(__file__).resolve().parents[1] / "assets.json"


def default_cache():
    return Path(os.environ.get("LOOM_PRIVATE_ASSET_CACHE") or
                Path.home() / ".cache/cuda-elastic/loom-private-assets")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, spec):
    if not path.is_file():
        return f"missing {path}"
    if path.stat().st_size != spec["bytes"]:
        return f"{path} has {path.stat().st_size} bytes, expected {spec['bytes']}"
    actual = sha256(path)
    if actual != spec["sha256"]:
        return f"{path} has sha256 {actual}, expected {spec['sha256']}"
    return None


def has_commit(clone, commit):
    """True when the clone holds the pinned commit (a partial clone does not)."""
    return (clone / ".git").is_dir() and subprocess.run(
        ["git", "-C", str(clone), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True).returncode == 0


def fetch_bytes(source, cache):
    if source["kind"] == "git":
        clone = cache / "git" / f"{Path(source['url']).stem}-{source['commit'][:12]}"
        if not has_commit(clone, source["commit"]):
            clone.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q", str(clone)], check=True)
            subprocess.run(["git", "-C", str(clone), "fetch", "-q", "--depth", "1",
                            source["url"], source["commit"]], check=True)
        return subprocess.run(["git", "-C", str(clone), "show",
                               f"{source['commit']}:{source['path']}"],
                              check=True, capture_output=True).stdout
    if source["kind"] == "github":
        return subprocess.run(
            ["gh", "api", "-H", "Accept: application/vnd.github.raw",
             f"repos/{source['repository']}/contents/{source['path']}"
             f"?ref={source['commit']}"],
            check=True, capture_output=True).stdout
    raise ValueError(f"unknown source kind {source['kind']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    assets = json.loads(REGISTRY.read_text())["assets"]
    names = args.names or sorted(assets)
    status = 0
    for name in names:
        if name not in assets:
            print(f"unknown asset {name}", file=sys.stderr)
            status = 3
            continue
        spec = assets[name]
        path = args.cache / name
        problem = verify(path, spec)
        if problem and not args.verify_only:
            try:
                data = fetch_bytes(spec["source"], args.cache)
            except (subprocess.CalledProcessError, OSError) as error:
                problem = f"cannot fetch {name}: {error}"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
                    out.write(data)
                os.replace(out.name, path)
                problem = verify(path, spec)
        if problem:
            print(f"FAIL {name}: {problem}", file=sys.stderr)
            status = 3
        else:
            print(f"ok   {name} {spec['sha256']} {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

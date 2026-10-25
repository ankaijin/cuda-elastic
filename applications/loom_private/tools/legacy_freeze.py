#!/usr/bin/env python3
"""Record or verify the hashes of the legacy CUDA portfolio.

  legacy_freeze.py record   write legacy-freeze.json from the baseline commit
  legacy_freeze.py verify   check the worktree still matches it

The frozen set is every file tracked at the baseline commit under the legacy
applications, their scripts, the selected inputs, the raw results, and the
generated reports. The loom_private suite adds files beside them and never
changes one of these.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
FREEZE = Path(__file__).resolve().parents[1] / "legacy-freeze.json"
BASELINE = "4ce57c7c0a333b74e9e6e2bd78142604c4c2cf39"
FROZEN = [
    "llama2c", "loom_msa", "pagerank", "mlperf-anomaly", "mlperf-classification",
    "vec_add.cu", "gpu.sh", "time_all.sh", "profile_ncu_all.sh",
    "results/README.md", "results/cuda_performance.xlsx", "results/inputs",
    "rtx3090", "rtx4090",
]


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record():
    listing = git("ls-tree", "-r", "-z", BASELINE, "--", *FROZEN).split(b"\0")
    files = {}
    for entry in filter(None, listing):
        meta, path = entry.split(b"\t", 1)
        mode, kind, blob = meta.decode().split()
        if kind != "blob":
            continue
        data = git("cat-file", "blob", blob)
        files[path.decode()] = {"mode": mode, "git_blob": blob, "sha256": sha256(data),
                                "bytes": len(data)}
    FREEZE.write_text(json.dumps({
        "schema": "loom_private.legacy_freeze/1",
        "baseline_commit": BASELINE,
        "paths": FROZEN,
        "files": dict(sorted(files.items())),
    }, indent=1) + "\n")
    print(f"recorded {len(files)} files from {BASELINE}")
    return 0


def verify():
    freeze = json.loads(FREEZE.read_text())
    problems = []
    for path, expected in freeze["files"].items():
        local = REPO / path
        if not local.is_file():
            problems.append(f"missing {path}")
            continue
        if sha256(local.read_bytes()) != expected["sha256"]:
            problems.append(f"changed {path}")
    tracked = set(git("ls-files", "-z", "--", *freeze["paths"]).decode().split("\0"))
    untracked_loss = [p for p in freeze["files"] if p not in tracked]
    problems += [f"no longer tracked {p}" for p in untracked_loss]
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(freeze['files']) - len(problems)}/{len(freeze['files'])} legacy files "
          f"match {freeze['baseline_commit']}")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["record", "verify"])
    return record() if parser.parse_args().action == "record" else verify()


if __name__ == "__main__":
    raise SystemExit(main())

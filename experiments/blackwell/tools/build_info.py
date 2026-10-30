#!/usr/bin/env python3
"""Compile one experiment binary and record how it was built.

  build_info.py --output BIN --case NAME --experiment EXP --arch ARCH
                --spec SPEC [--source PATH]... -- COMPILER ARGS...

Runs the compile command, then writes BIN.build.json: the command, the
compiler version, the architecture, the specification hash, the SHA-256 of
every listed source and of the binary, the repository revision and whether
the worktree had uncommitted changes, and the git revision of every include
directory outside the repository (CUTLASS). A failed compile writes nothing.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def git(*args, cwd=REPO):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def external_includes(command):
    """Include directories outside the repository, with their git revision."""
    found = {}
    for i, arg in enumerate(command):
        path = arg[2:] if arg.startswith("-I") and len(arg) > 2 else (
            command[i + 1] if arg == "-I" and i + 1 < len(command) else None)
        if not path:
            continue
        resolved = Path(path).resolve()
        if resolved.is_relative_to(REPO) or not resolved.is_dir():
            continue
        top = git("rev-parse", "--show-toplevel", cwd=resolved)
        found[str(resolved)] = {
            "git_toplevel": top,
            "git_commit": git("rev-parse", "HEAD", cwd=resolved) if top else None,
            "git_describe": git("describe", "--tags", "--always", cwd=resolved) if top else None,
            "dirty": bool(git("status", "--porcelain", cwd=resolved)) if top else None,
        }
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a compile command is required after --")
    print(" ".join(command), flush=True)
    code = subprocess.run(command).returncode
    if code:
        return code
    binary = Path(args.output)
    try:
        compiler = subprocess.check_output([command[0], "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        compiler = None
    info = {
        "schema": "blackwell_experiments.build/1",
        "experiment": args.experiment,
        "case": args.case,
        "binary": relative(binary),
        "binary_sha256": sha256(binary),
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arch": args.arch,
        "command": command,
        "compiler_version": compiler,
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "spec": relative(args.spec),
        "spec_sha256": sha256(args.spec),
        "sources": {relative(path): sha256(path) for path in args.source},
        "external_includes": external_includes(command),
    }
    Path(str(binary) + ".build.json").write_text(json.dumps(info, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

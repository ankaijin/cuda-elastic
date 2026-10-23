#!/usr/bin/env python3
"""Validate, run, and record an experiment without changing published reports."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import uuid


def utc_now():
    return datetime.now(timezone.utc)


def output(command, repo):
    return subprocess.check_output(command, cwd=repo, text=True).strip()


def tool_version(command, repo):
    try:
        return output(command, repo)
    except (OSError, subprocess.CalledProcessError):
        return None


def run_logged(command, path, repo, env):
    with path.open("w") as log:
        try:
            return subprocess.run(command, cwd=repo, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        except OSError as error:
            log.write(str(error) + "\n")
            return 127


def source_changes(repo, commit):
    paths = [".", ":(exclude)rtx3090", ":(exclude)rtx4090", ":(exclude)rtx5090"]
    changed = output(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths], repo)
    if output(["git", "rev-parse", "HEAD"], repo) != commit:
        changed += "\nHEAD changed"
    return changed.strip()


def binary_hashes(repo, folder):
    hashes = {}
    for path in (repo / folder).rglob("*"):
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path.relative_to(repo))] = digest.hexdigest()
    return hashes


def record_run(repo, device, check, command):
    repo = Path(repo).resolve()
    if output(["git", "status", "--porcelain", "--untracked-files=all"], repo):
        raise ValueError("Start from a clean worktree: commit source and previous results first")
    commit = output(["git", "rev-parse", "HEAD"], repo)
    query = output(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                    "--format=csv,noheader", "-i", device], repo)
    rows = list(csv.reader(query.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("Select exactly one GPU with --device")
    name, gpu_uuid, driver = rows[0]
    folder = name.removeprefix("NVIDIA ").removeprefix("GeForce ").replace(" ", "").lower()
    arches = {"rtx3090": "sm_86", "rtx4090": "sm_89", "rtx5090": "sm_120"}
    if folder not in arches:
        raise ValueError(f"Unsupported GPU: {name}")
    host = re.sub(r"[^A-Za-z0-9_]", "_", socket.gethostname().split(".")[0])
    run_id = f"{utc_now():%Y%m%d-%H%M%S}-{host}-{uuid.uuid4().hex[:8]}"
    directory = repo / folder / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, GPU=folder,
               ARCH=arches[folder], TIMESTAMP=run_id)
    env.setdefault("GPU_LOCK", "/tmp/cuda-elastic-gpu.lock")
    metadata = {
        "schema_version": 1, "run_id": run_id, "source_commit": commit,
        "source_branch": output(["git", "branch", "--show-current"], repo),
        "host": socket.gethostname(), "started_utc": utc_now().isoformat(),
        "gpu": {"name": name, "uuid": gpu_uuid, "driver": driver, "folder": folder},
        "installed_tools": {name: tool_version([name, "--version"], repo)
                  for name in ("nvcc", "ncu", "nsys")},
        "environment": {key: value for key, value in env.items() if key in (
            "CUDA_VISIBLE_DEVICES", "GPU", "ARCH", "TIMESTAMP", "GPU_LOCK",
            "RUNS", "BATCH", "SMALL_BATCH", "SHAPES", "GRAPH_REPLAYS", "LAUNCH",
            "APP", "VARIANT", "INPUTS", "SEEDS", "REPLAYS", "NVCC", "NVCCFLAGS",
            "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "PATH",
            "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH") or key.startswith("PAGERANK_")},
        "binary_sha256": {},
        "validation": {"command": check, "exit_code": None, "log": "validation.log"},
        "measurement": {"command": command, "exit_code": None, "log": "measurement.log"},
        "status": "validating", "artifacts": [],
    }
    path = directory / "metadata.json"

    def save():
        path.write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    print(f"Experiment {run_id}; logs: {directory}", flush=True)
    try:
        code = run_logged(["bash", "-c", check], directory / "validation.log", repo, env)
        metadata["validation"]["exit_code"] = code
        if code:
            metadata["status"] = "validation_failed"
            return code if code > 0 else 128 - code
        metadata["source_changes_after_validation"] = source_changes(repo, commit)
        if metadata["source_changes_after_validation"]:
            metadata["status"] = "source_changed"
            return 2
        metadata["binary_sha256"] = binary_hashes(repo, folder)
        metadata["status"] = "measuring"
        save()
        code = run_logged(command, directory / "measurement.log", repo, env)
        metadata["measurement"]["exit_code"] = code
        metadata["status"] = "completed" if code == 0 else "measurement_failed"
        metadata["source_changes_after_measurement"] = source_changes(repo, commit)
        if metadata["source_changes_after_measurement"]:
            metadata["status"] = "source_changed"
            return 2
        return code if code >= 0 else 128 - code
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        return 130
    finally:
        metadata["finished_utc"] = utc_now().isoformat()
        metadata["artifacts"] = sorted(str(p.relative_to(repo))
            for p in (repo / folder).rglob(f"*{run_id}*") if p.is_file())
        save()
        print(f"{metadata['status']}: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="one nvidia-smi index or UUID")
    parser.add_argument("--check", required=True, help="validation shell command; measurement requires exit 0")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="measurement command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a measurement command is required after --")
    try:
        return record_run(Path(__file__).resolve().parent.parent, args.device, args.check, command)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

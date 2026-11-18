"""Auditable subprocess execution: argv lists, no shell interpolation."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import time
from pathlib import Path

from .errors import DependencyError, ValidationError
from .io import utc_now, write_json


def require_version(package: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DependencyError(f"Missing optional backend: {package}=={expected}") from exc
    if actual != expected:
        raise DependencyError(
            f"Adapter targets {package}=={expected}; found {actual}. Review the adapter before changing version."
        )


def inspect_gpu() -> dict:
    command = shutil.which("nvidia-smi")
    if not command:
        raise DependencyError(
            "No nvidia-smi; GPU availability must be verified before this backend runs"
        )
    result = subprocess.run(
        [command, "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        raise DependencyError("No usable NVIDIA GPU inventory returned")
    processes = subprocess.run(
        [command, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return {
        "checked_at": utc_now(),
        "gpu_inventory": result.stdout.strip(),
        "process_inventory": processes.stdout.strip(),
        "process_query_returncode": processes.returncode,
    }


class CommandRunner:
    """Commands run in the caller's selected environment; no implicit installs."""

    def run(self, argv: list[str], cwd: Path, log: Path) -> dict:
        if not argv or not shutil.which(argv[0]):
            raise DependencyError(f"Executable unavailable: {argv[0] if argv else '<empty>'}")
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        record = {
            "argv": argv,
            "cwd": str(cwd.resolve()),
            "started_at": utc_now(),
            "status": "RUNNING",
        }
        write_json(log.with_suffix(".command.json"), record)
        try:
            with log.open("x") as stream:
                result = subprocess.run(
                    argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            record.update(
                status="SUCCEEDED" if result.returncode == 0 else "FAILED",
                returncode=result.returncode,
            )
            if result.returncode:
                raise ValidationError(f"Backend exited {result.returncode}; see {log}")
            return record
        except BaseException as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record.update(elapsed_seconds=time.monotonic() - start, finished_at=utc_now())
            write_json(log.with_suffix(".command.json"), record)

"""Download a pinned, explicitly selected Objaverse catalog via the official HF CLI."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .io import file_hash, read_json, write_json


class ObjaverseDownloader:
    def __init__(self, hf_executable: str = "hf", endpoint: str = "https://hf-mirror.com"):
        self.hf_executable = shutil.which(hf_executable)
        if not self.hf_executable:
            raise FileNotFoundError(hf_executable)
        self.endpoint = endpoint

    def download(self, catalog: Path, output: Path) -> Path:
        items = read_json(catalog)
        env = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
            env.pop(key.upper(), None)
        env["HF_ENDPOINT"] = self.endpoint
        env["HF_HUB_DISABLE_XET"] = "1"

        def fetch(item):
            # One exact filename per CLI call: do not recursively list the 8.9-TB repo.
            subprocess.run(
                [
                    self.hf_executable,
                    "download",
                    item["repository"],
                    item["repo_path"],
                    "--repo-type",
                    "dataset",
                    "--revision",
                    item["revision"],
                    "--local-dir",
                    str(output),
                ],
                env=env,
                check=True,
            )
            path = output / item["repo_path"]
            return {**item, "local_path": str(path.resolve()), "sha256": file_hash(path)}

        with ThreadPoolExecutor(max_workers=3) as executor:
            items["objects"] = list(executor.map(fetch, items["objects"]))
        target = output / "downloaded_catalog.json"
        write_json(target, items)
        return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf", default="hf")
    args = parser.parse_args()
    print(ObjaverseDownloader(args.hf).download(args.catalog, args.output))


if __name__ == "__main__":
    main()

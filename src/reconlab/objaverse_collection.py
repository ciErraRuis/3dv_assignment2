"""Build and download a reproducible, category-diverse Objaverse candidate pool.

This is raw asset acquisition, not Assignment 1 cleaning or a rendering evaluation.
The original pilot assets are reused through links and are never modified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .io import file_hash, read_json, utc_now, write_json


class ObjaverseCatalogBuilder:
    """Select from pinned local metadata; persist the exact selection and provenance."""

    preferred_categories = (
        "fire_extinguisher chair vase armchair bicycle airplane car_(automobile) "
        "backpack banana barrel book bookcase camera cup mug kettle teakettle "
        "sofa table flowerpot cabinet crate piano violin lamp table_lamp "
        "alarm_clock shoe boot helmet sculpture figurine toy teddy_bear "
        "water_bottle beer_bottle coffeepot computer_keyboard laptop_computer "
        "headset radio_receiver microphone skateboard basketball baseball "
        "trophy_cup watering_can suitcase refrigerator"
    ).split()

    def __init__(self, source: Path, repository: str, revision: str, seed: int = 42):
        self.source = source
        self.repository, self.revision, self.seed = repository, revision, seed

    def _order(self, value: str) -> str:
        return hashlib.sha256(f"{self.seed}:{value}".encode()).hexdigest()

    @staticmethod
    def _load(path: Path) -> dict:
        with gzip.open(path, "rt") as stream:
            return json.load(stream)

    @staticmethod
    def _eligible(item: dict) -> bool:
        glb = item.get("archives", {}).get("glb", {})
        return (
            item.get("license") in {"by", "cc0", "by-sa"}
            and not item.get("isAgeRestricted", False)
            and item.get("animationCount") == 0
            and 1024 <= glb.get("size", 0) <= 25 * 1024**2
            and 100 <= glb.get("faceCount", 0) <= 200_000
        )

    def build(self, target: int, reserves: int = 40) -> dict:
        paths_file = self.source / "object-paths.json.gz"
        lvis_file = self.source / "lvis-annotations.json.gz"
        metadata_files = sorted((self.source / "metadata").glob("*.json.gz"))
        paths, lvis = self._load(paths_file), self._load(lvis_file)
        metadata, shards, labels = {}, {}, defaultdict(list)
        for path in metadata_files:
            for uid, item in self._load(path).items():
                metadata[uid], shards[uid] = item, str(path)
        for category, uids in lvis.items():
            for uid in uids:
                labels[uid].append(category)

        def record(uid, category, reuse=None):
            item = metadata[uid]
            relative = paths[uid]
            if not re.fullmatch(r"glbs/\d{3}-\d{3}/[a-f0-9]{32}\.glb", relative):
                raise ValueError(f"Invalid asset path for {uid}")
            return {
                "id": uid,
                "uid": uid,
                "name": item.get("name", ""),
                "category": category,
                "lvis_categories": sorted(labels[uid]),
                "license": item["license"],
                "author": {
                    key: item.get("user", {}).get(key)
                    for key in ("username", "displayName", "profileUrl")
                },
                "url": item.get("viewerUrl"),
                "repository": self.repository,
                "revision": self.revision,
                "repo_path": relative,
                "metadata_file": shards[uid],
                "source_metadata": item,
                "reuse_path": str(reuse.resolve()) if reuse else None,
            }

        selected, used, counts = [], set(), Counter()
        # Keep all four previously downloaded models without copying their bytes.
        for path in sorted((self.source / "glbs").glob("*/*.glb")):
            uid = path.stem
            category = next(
                (c for c in self.preferred_categories if c in labels[uid]),
                next(iter(labels[uid]), "unlabeled"),
            )
            selected.append(record(uid, category, path))
            used.add(uid)
            counts[category] += 1
        pools = {
            category: sorted(
                {u for u in uids if u in metadata and u in paths and self._eligible(metadata[u])},
                key=self._order,
            )
            for category, uids in lvis.items()
        }
        categories = [c for c in self.preferred_categories if pools.get(c)]
        categories += sorted((c for c in pools if c not in categories), key=self._order)
        for _ in range(3):
            for category in categories:
                if len(selected) >= target + reserves:
                    break
                available = [u for u in pools[category] if u not in used]
                if not available or counts[category] >= 3:
                    continue
                uid = available[0]
                selected.append(record(uid, category))
                used.add(uid)
                counts[category] += 1
        if len(selected) < target:
            raise ValueError("Insufficient eligible distinct objects in metadata")
        return {
            "created_at": utc_now(),
            "target": target,
            "repository": self.repository,
            "revision": self.revision,
            "seed": self.seed,
            "selection": "Category round-robin, preferred everyday objects first, seeded UID order; "
            "up to 3 per assigned category. Metadata is limited to the saved shards, "
            "not a random sample of the full dataset. Prior assets retained.",
            "new_asset_filters": {
                "licenses": ["by", "cc0", "by-sa"],
                "animation_count": 0,
                "max_metadata_bytes": 25 * 1024**2,
                "faces": [100, 200000],
            },
            "metadata_provenance": [
                {"path": str(p), "sha256": file_hash(p)}
                for p in [paths_file, lvis_file, *metadata_files]
            ],
            "objects": selected,
        }


class GlbInspector:
    """Check GLB framing, embedded resources and nonempty mesh primitives on CPU."""

    def inspect(self, path: Path) -> dict:
        size = path.stat().st_size
        with path.open("rb") as stream:
            magic, version, declared = struct.unpack("<4sII", stream.read(12))
            if magic != b"glTF" or version != 2 or declared != size:
                raise ValueError("Invalid GLB header or truncated file")
            document, binary_bytes, chunks = None, 0, 0
            while stream.tell() < size:
                length, kind = struct.unpack("<II", stream.read(8))
                if length % 4 or stream.tell() + length > size:
                    raise ValueError("Invalid GLB chunk extent")
                if chunks == 0 and kind != 0x4E4F534A:
                    raise ValueError("GLB must start with JSON")
                if kind == 0x4E4F534A:
                    if document is not None:
                        raise ValueError("Multiple JSON chunks")
                    document = json.loads(stream.read(length))
                else:
                    if kind == 0x004E4942:
                        binary_bytes += length
                    stream.seek(length, 1)
                chunks += 1
        if not document or document.get("asset", {}).get("version") != "2.0":
            raise ValueError("Missing glTF 2.0 document")
        resources = document.get("buffers", []) + document.get("images", [])
        if any(x.get("uri") and not x["uri"].startswith("data:") for x in resources):
            raise ValueError("External resources require a separate download")
        embedded_bytes = sum(
            x.get("byteLength", 0) for x in document.get("buffers", []) if not x.get("uri")
        )
        if embedded_bytes > binary_bytes:
            raise ValueError("Truncated embedded binary buffer")
        primitives = [p for mesh in document.get("meshes", []) for p in mesh.get("primitives", [])]
        accessors = document.get("accessors", [])
        vertices = 0
        for primitive in primitives:
            index = primitive.get("attributes", {}).get("POSITION")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                raise ValueError("Mesh primitive has no valid position accessor")
            vertices += accessors[index].get("count", 0)
        if not primitives or vertices <= 0:
            raise ValueError("Empty mesh")
        return {
            "bytes": size,
            "sha256": file_hash(path),
            "mesh_count": len(document["meshes"]),
            "primitive_count": len(primitives),
            "position_accessor_count_sum": vertices,
            "material_count": len(document.get("materials", [])),
            "image_count": len(document.get("images", [])),
            "animation_count": len(document.get("animations", [])),
            "check": "GLB structure and embedded resources; not a rendering-quality check",
        }


class ObjaverseCollectionDownloader:
    def __init__(self, hf: str, endpoint: str = "https://hf-mirror.com", workers: int = 4):
        self.hf = shutil.which(hf)
        if not self.hf:
            raise FileNotFoundError(hf)
        if workers < 1:
            raise ValueError("workers must be positive")
        self.endpoint, self.workers = endpoint, workers
        self.inspector = GlbInspector()

    def _fetch(self, item: dict, output: Path) -> dict:
        path = output / "source" / item["repo_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if item.get("reuse_path") and not path.exists():
            path.symlink_to(item["reuse_path"])
        env = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
            env.pop(key.upper(), None)
        env.update(
            HF_ENDPOINT=self.endpoint,
            HF_HUB_DISABLE_XET="1",
            HF_HUB_DISABLE_PROGRESS_BARS="1",
            HF_HUB_ETAG_TIMEOUT="30",
            HF_HUB_DOWNLOAD_TIMEOUT="60",
        )
        log = output / "logs" / f"{item['uid']}.log"
        for attempt in range(1, 3):
            try:
                if not path.exists():
                    with log.open("a") as stream:
                        subprocess.run(
                            [
                                self.hf,
                                "download",
                                item["repository"],
                                item["repo_path"],
                                "--repo-type",
                                "dataset",
                                "--revision",
                                item["revision"],
                                "--local-dir",
                                str(output / "source"),
                            ],
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=240,
                        )
                audit = self.inspector.inspect(path)
                return {
                    **item,
                    "local_path": str(path),
                    "resolved_path": str(path.resolve()),
                    "audit": audit,
                    "status": "verified",
                    "checked_at": utc_now(),
                }
            except (OSError, ValueError, struct.error, subprocess.SubprocessError) as exc:
                if attempt == 2 or path.exists():
                    return {
                        "uid": item["uid"],
                        "status": "failed",
                        "error": str(exc),
                        "log": str(log),
                        "checked_at": utc_now(),
                    }
        raise AssertionError("Unreachable")

    def run(self, catalog: Path, output: Path) -> dict:
        selection = read_json(catalog)
        output.mkdir(parents=True, exist_ok=True)
        (output / "logs").mkdir(exist_ok=True)
        valid, rejected, hashes = [], [], set()
        candidates, target = selection["objects"], selection["target"]
        cursor = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while len(valid) < target and cursor < len(candidates):
                batch = candidates[cursor : cursor + min(self.workers, target - len(valid))]
                cursor += len(batch)
                for result in pool.map(lambda x: self._fetch(x, output), batch):
                    if result["status"] != "verified":
                        rejected.append(result)
                    elif result["audit"]["sha256"] in hashes:
                        rejected.append(
                            {
                                "uid": result["uid"],
                                "status": "duplicate_bytes",
                                "local_path": result["local_path"],
                            }
                        )
                    else:
                        valid.append(result)
                        hashes.add(result["audit"]["sha256"])
                    summary = {
                        "state": "DOWNLOADING",
                        "checked_at": utc_now(),
                        "target": target,
                        "verified_objects": len(valid),
                        "rejected_objects": len(rejected),
                        "verified_bytes": sum(x["audit"]["bytes"] for x in valid),
                    }
                    write_json(output / "status.json", summary)
                    write_json(output / "downloaded_catalog.json", {"objects": valid})
                    write_json(output / "rejections.json", rejected)
                    print(json.dumps(summary), flush=True)
        summary.update(
            state="COMPLETE" if len(valid) == target else "INCOMPLETE",
            category_counts=dict(Counter(x["category"] for x in valid)),
            license_counts=dict(Counter(x["license"] for x in valid)),
            reused_objects=sum(bool(x["reuse_path"]) for x in valid),
            catalog_sha256=file_hash(catalog),
            endpoint=self.endpoint,
            note="Raw model pool; not cleaned Assignment 1 data, not rendered or trained.",
        )
        write_json(output / "status.json", summary)
        if len(valid) < target:
            raise RuntimeError(f"Only {len(valid)}/{target} objects verified; see rejections.json")
        write_json(output / "COMPLETE.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "download"])
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--repository", default="allenai/objaverse")
    parser.add_argument("--revision", default="556637099bf4fa79ea7b239d0d9c328b8a2e9ac8")
    parser.add_argument("--hf", default="hf")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.target < 1:
        parser.error("target must be positive")
    if args.action == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        if args.catalog.exists():
            parser.error("Selection already exists; use a new catalog for a different selection")
        result = ObjaverseCatalogBuilder(args.source, args.repository, args.revision).build(
            args.target
        )
        write_json(args.catalog, result)
        print(json.dumps({"catalog": str(args.catalog), "candidates": len(result["objects"])}))
    else:
        if args.output is None:
            parser.error("download requires --output")
        print(
            json.dumps(
                ObjaverseCollectionDownloader(args.hf, workers=args.workers).run(
                    args.catalog, args.output
                ),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

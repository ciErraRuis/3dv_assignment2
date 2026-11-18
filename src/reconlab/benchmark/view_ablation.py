"""View-count ablations using pair-local caches, never full-scene camera estimates."""

from __future__ import annotations

import argparse
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .core import PairMatches, TrackGraph, draw_matches, save_json
from .sfm import IncrementalMapper
from .visuals import export_geometry


class ViewSubset:
    @staticmethod
    def select(names, count, sampling):
        names = sorted(names)
        if not 3 <= count <= len(names):
            raise ValueError("Invalid subset size")
        if sampling == "uniform_orbit":
            indices = np.floor(np.arange(count) * len(names) / count).astype(int)
        elif sampling == "contiguous_arc":
            indices = np.arange(count)
        else:
            raise ValueError(sampling)
        return [names[i] for i in indices]


class ViewCountBenchmark:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["output"])
        self.mapper = IncrementalMapper(num_threads=2)

    def run(self):
        for method in self.config["view_ablation_methods"]:
            for obj in self.config["objects"]:
                source = self.root / "results" / method / obj["uid"]
                if not (source / "matching.json").exists():
                    continue  # producer not finished; controller calls again later
                images = self.root / "inputs" / obj["uid"] / "images"
                available = [p.name for p in images.glob("*.png")]
                if len(available) != 36:
                    raise ValueError("Unexpected frozen train split")
                for count in [3, 5, 10, 20, 36]:
                    for sampling in ["uniform_orbit", "contiguous_arc"]:
                        if count == 36 and sampling == "contiguous_arc":
                            continue
                        dest = (
                            self.root
                            / "view_ablation"
                            / method
                            / obj["uid"]
                            / f"{sampling}_{count:02d}"
                        )
                        if (dest / "status.json").exists():
                            continue
                        dest.mkdir(parents=True, exist_ok=True)
                        names = ViewSubset.select(available, count, sampling)
                        isolated = dest / "images"
                        isolated.mkdir(exist_ok=True)
                        for name in names:
                            (isolated / name).symlink_to(images / name)
                        save_json(
                            dest / "input_manifest.json",
                            {
                                "names": names,
                                "count": count,
                                "sampling": sampling,
                                "source_manifest": str(images.parent / "input_manifest.json"),
                                "GT_cameras_supplied": False,
                                "cache_protocol": "pairwise correspondences depend on these two images only; SfM rerun from scratch on each subset",
                            },
                        )
                        start = time.monotonic()
                        try:
                            graph, rows = TrackGraph(names), []
                            for index, (a, b) in enumerate(itertools.combinations(names, 2)):
                                path = source / "matches" / f"{Path(a).stem}__{Path(b).stem}.npz"
                                with np.load(path) as data:
                                    matches = PairMatches(
                                        data["first"], data["second"], data["scores"]
                                    )
                                    mask = data["inliers"]
                                    graph.add(a, b, matches, mask)
                                rows.append(
                                    {
                                        "first": a,
                                        "second": b,
                                        "matches": len(mask),
                                        "inliers": int(mask.sum()),
                                    }
                                )
                                if index < 3:
                                    # Preserve actual grayscale/resize display evidence from producer.
                                    evidence = source / "model_inputs"
                                    candidates = sorted(evidence.glob(f"*/{a}"))
                                    display_a = candidates[-1] if candidates else images / a
                                    display_b = display_a.parent / b
                                    from PIL import Image

                                    width, height = Image.open(display_a).size
                                    shown = PairMatches(
                                        matches.first * [width / 512, height / 512],
                                        matches.second * [width / 512, height / 512],
                                        matches.scores,
                                    )
                                    draw_matches(
                                        display_a,
                                        display_b,
                                        shown,
                                        mask,
                                        dest / "visuals" / f"pair_{index:03d}.png",
                                    )
                            save_json(
                                dest / "matching.json", {"pairs": rows, "matching_reused": True}
                            )
                            geometry = self.mapper.reconstruct(graph, isolated, dest / "sfm")
                            geometry.metadata.update(
                                method=method,
                                input_views=count,
                                registration_rate=len(geometry.names) / count,
                            )
                            geometry.save(dest)
                            export_geometry(
                                geometry, Path(self.config["source_inputs"]) / obj["uid"], dest
                            )
                            state = {"state": "COMPLETE", **geometry.metadata}
                        except Exception as error:
                            (dest / "error.log").write_text(traceback.format_exc())
                            state = {"state": "FAILED", "error": str(error), "input_views": count}
                        state.update(
                            seconds=time.monotonic() - start,
                            checked_at=time.time(),
                            matching_runtime_included=False,
                        )
                        save_json(dest / "status.json", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    ViewCountBenchmark(json.loads(args.config.read_text())).run()


if __name__ == "__main__":
    main()

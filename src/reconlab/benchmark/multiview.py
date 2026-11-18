"""Official MV-RoMa joint matching, followed by the shared track/SfM backend."""

from __future__ import annotations

import itertools
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

from .core import PairMatches, ReconstructionMethod, TrackGraph, draw_matches, save_json, verify
from .matchers import record_pixels
from .sfm import IncrementalMapper


class MultiViewRoma(ReconstructionMethod):
    def __init__(self, sources: Path, weights: Path):
        import torch

        # The official MV-RoMa distribution vendors only part of UFM; the full
        # upstream package supplies its utilities, as its install guide requires.
        for path in [sources / "mvroma", sources / "ufm", sources / "ufm/UniCeption"]:
            sys.path.insert(0, str(path))
        from src.build_model import build_our_model
        from src.matchers.uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
        from src.mvroma import ModelConfig
        from src.run_model import run_model_test

        self.torch, self.run = torch, run_model_test
        cfg = ModelConfig()
        cfg.num_cluster = 512
        args = Namespace(
            use_dinov2=True,
            train_until_16x=False,
            train_refiner=False,
            train_all_model=False,
            num_cluster=512,
        )
        self.model, _ = build_our_model(args, cfg, use_dinov2=True)
        state = torch.load(weights / "mvroma_outdoor.pth", map_location="cpu", weights_only=False)
        loaded = self.model.load_state_dict(state, strict=False)
        # The official demo loads this checkpoint with strict=False. It contains
        # unused legacy conv backbones and persisted rotary-frequency buffers;
        # the inference model has every required parameter. Reject other drift.
        unexpected = [
            key
            for key in loaded.unexpected_keys
            if not (key.startswith("conv_backbones.") or key.endswith(".attn.rope.periods"))
        ]
        if loaded.missing_keys or unexpected:
            raise RuntimeError(f"MV-RoMa checkpoint mismatch: {loaded}")
        self.checkpoint_load = {
            "missing_keys": loaded.missing_keys,
            "unused_checkpoint_keys": loaded.unexpected_keys,
            "policy": "official demo strict=False; reject missing or unrecognized extra keys",
        }
        self.model.eval().cuda()
        self.prematcher = (
            UniFlowMatchClassificationRefinement.from_pretrained(
                str(weights / "ufm"), inference_resolution=(560, 560)
            )
            .eval()
            .cuda()
        )
        self.prematcher.register_forward_pre_hook(self._check_ufm_input)
        original_match = self.model.match

        def captured_match(**kwargs):
            for path, coarse, fine in zip(
                self.group_paths, kwargs["multi_view_frames"][0], kwargs["multi_view_frames_org"][0]
            ):
                self.capture(path, coarse, "mvroma_coarse_560")
                self.capture(path, fine, "mvroma_fine_560")
            return original_match(**kwargs)

        self.model.match = captured_match
        original_predict = self.prematcher.predict_correspondences_batched

        def captured_predict(*args, **kwargs):
            self.capture(self.group_paths[0], kwargs["source_image"], "ufm_rgb_560")
            self.capture(self.group_paths[self.ref_index], kwargs["target_image"], "ufm_rgb_560")
            self.ref_index += 1
            return original_predict(*args, **kwargs)

        self.prematcher.predict_correspondences_batched = captured_predict

    @staticmethod
    def _check_ufm_input(module, args):
        for view in args[:2]:
            if tuple(view["img"].shape[-2:]) != (560, 560):
                raise ValueError("UFM internal resizing violated square input policy")

    def capture(self, path, tensor, stage):
        if (path, stage) not in self.seen:
            record_pixels(self, path, tensor, stage)
            self.seen.add((path, stage))

    def reconstruct(self, images: Path, output: Path):
        paths = sorted(images.glob("*.png"))
        if len(paths) < 4:
            raise ValueError(
                "This configured MV-RoMa adapter requires one query and three distinct references"
            )
        total_pairs = len(paths) * (len(paths) - 1) // 2
        self.evidence_root, self.seen = output / "model_inputs", set()
        save_json(output / "checkpoint_load.json", self.checkpoint_load)
        graph = TrackGraph([p.name for p in paths])
        pairs = {pair: index for index, pair in enumerate(itertools.combinations(paths, 2))}
        (output / "matches").mkdir(exist_ok=True)
        rows, groups = [], []
        started = time.monotonic()
        for anchor, query in enumerate(paths[:-1]):
            references = paths[anchor + 1 :]
            for offset in range(0, len(references), 3):
                targets = references[offset : offset + 3]
                context = list(targets)
                for path in paths:
                    if len(context) == 3:
                        break
                    if path != query and path not in context:
                        context.append(path)
                self.group_paths, self.ref_index = [query, *context], 1
                group_start = time.monotonic()
                with self.torch.inference_mode():
                    predictions = self.run(
                        self.model,
                        {"query_img_path": str(query), "ref_img_paths": [str(p) for p in context]},
                        coarse_res_hw=(560, 560),
                        target_res_hw=(560, 560),
                        prematch_model=[None, self.prematcher],
                        prematch_model_name="ufm",
                        num_cluster=512,
                        upsample_preds=True,
                        device="cuda",
                    )
                finest = predictions[min(predictions)]
                for index, target in enumerate(targets):
                    flow = finest["flow"][0, index].permute(1, 2, 0)
                    score = finest["certainty"][0, index, 0].sigmoid()
                    height, width = score.shape
                    yy, xx = self.torch.meshgrid(
                        self.torch.arange(height, device=flow.device),
                        self.torch.arange(width, device=flow.device),
                        indexing="ij",
                    )
                    first = self.torch.stack(
                        ((xx + 0.5) * 512 / width - 0.5, (yy + 0.5) * 512 / height - 0.5), dim=-1
                    )
                    second = (flow + 1) * 256 - 0.5
                    valid = (score > 0.5) & (second >= 0).all(-1) & (second < 512).all(-1)
                    ids = valid.flatten().nonzero().flatten()
                    # Deterministic confidence-weighted selection, no GT mask.
                    if len(ids) > 5000:
                        ids = ids[
                            self.torch.multinomial(score.flatten()[ids], 5000, replacement=False)
                        ]
                    result = PairMatches(
                        first.reshape(-1, 2)[ids].cpu().numpy(),
                        second.reshape(-1, 2)[ids].cpu().numpy(),
                        score.flatten()[ids].cpu().numpy(),
                    )
                    mask = verify(result)
                    graph.add(query.name, target.name, result, mask)
                    np.savez_compressed(
                        output / "matches" / f"{query.stem}__{target.stem}.npz",
                        first=result.first,
                        second=result.second,
                        scores=result.scores,
                        inliers=mask,
                    )
                    rows.append(
                        {
                            "first": query.name,
                            "second": target.name,
                            "matches": len(mask),
                            "inliers": int(mask.sum()),
                        }
                    )
                    pair_id = pairs[(query, target)]
                    if pair_id in [0, 1, 4, 8, 17, 34]:
                        shown = PairMatches(
                            result.first * 560 / 512, result.second * 560 / 512, result.scores
                        )
                        draw_matches(
                            self.display_paths[query],
                            self.display_paths[target],
                            shown,
                            mask,
                            output / "visuals" / f"pair_{pair_id:03d}.png",
                        )
                groups.append(
                    {
                        "query": query.name,
                        "references": [p.name for p in context],
                        "exported": [p.name for p in targets],
                        "seconds": time.monotonic() - group_start,
                    }
                )
                save_json(
                    output / "progress.json",
                    {
                        "pairs": len(rows),
                        "total_pairs": total_pairs,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                print(f"mvroma: {len(rows)}/{total_pairs} pairs", flush=True)
                del predictions, finest
        save_json(
            output / "matching.json",
            {
                "method": "mvroma",
                "pairs": rows,
                "groups": groups,
                "total_seconds": time.monotonic() - started,
                "group_policy": f"all {total_pairs} pairs; 3 references jointly, deterministic filename-order context; no camera metadata",
                "sizes": {"ufm": [560, 560], "coarse": [560, 560], "fine": [560, 560]},
                "upstream_deviation": "replace demo 420x560 UFM and 560x840 target with full-frame square resize",
                "track_consolidation": "nearest original-image pixel; one-to-one per pair",
            },
        )
        geometry = IncrementalMapper().reconstruct(graph, images, output / "sfm")
        geometry.metadata.update(
            method="mvroma",
            input_views=len(paths),
            registration_rate=len(geometry.names) / len(paths),
        )
        return geometry

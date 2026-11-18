"""Read-only, same-page comparison of every object and method; lazy media loading."""

from __future__ import annotations

import argparse
import io
import itertools
import json
import mimetypes
import re
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw

from .core import sha256

LABELS = {
    "sift": "SIFT",
    "surf": "SURF",
    "orb": "ORB",
    "akaze": "AKAZE",
    "superpoint_lightglue": "SuperPoint + LightGlue",
    "disk_lightglue": "DISK + LightGlue",
    "aliked_lightglue": "ALIKED + LightGlue",
    "efficientloftr": "EfficientLoFTR",
    "loma": "LoMa",
    "romav2": "RoMa v2",
    "mvroma": "MV-RoMa",
    "mast3r_sfm": "MASt3R-SfM",
    "mast3r_slam": "MASt3R-SLAM",
    "gluemap": "GLUEMAP",
    "3dgs": "3DGS",
    "3dgs_mcmc": "3DGS-MCMC",
    "2dgs": "2DGS",
    "3dgut_mcmc": "3DGUT + MCMC",
}
OBJECT_LABELS = [
    "红色灭火器",
    "木椅",
    "瓶子",
    "彩绘陶罐",
    "第二个灭火器",
    "第二把椅子",
    "扶手椅",
    "自行车",
    "飞机",
    "汽车",
]


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class ResultCatalog:
    def __init__(self, config, gt_root):
        self.config = read(config)
        self.root, self.gt = Path(self.config["output"]), gt_root
        self.methods = self.config["methods"]
        self.gaussians = ["3dgs", "3dgs_mcmc", "2dgs", "3dgut_mcmc"]
        self.objects = [
            {**o, "label": OBJECT_LABELS[i]} for i, o in enumerate(self.config["objects"])
        ]
        self.uids = {o["uid"] for o in self.objects}
        self.roots = {"m": self.root, "g": self.gt}
        self.allowed = [
            self.root.resolve(),
            self.gt.resolve(),
            Path(self.config["original_source_inputs"]).resolve(),
            Path(self.config["pair_cache_root"]).resolve(),
        ]

    def url(self, path):
        for alias, root in self.roots.items():
            if path.is_relative_to(root):
                return "/media/" + alias + "/" + quote(str(path.relative_to(root)), safe="/")
        raise ValueError("Artifact outside result roots")

    def resolve(self, route):
        alias, relative = route.split("/", 1)
        if alias not in self.roots:
            raise ValueError("Unknown artifact root")
        path = (self.roots[alias] / unquote(relative)).resolve()
        if not any(path.is_relative_to(root) for root in self.allowed):
            raise ValueError("Artifact outside allowed roots")
        if path.suffix.lower() not in {".png", ".gif", ".html", ".json", ".npz", ".ply", ".csv"}:
            raise ValueError("Unsupported artifact format")
        return path

    def directory(self, uid, method, branch):
        if uid not in self.uids:
            raise ValueError("Unknown object")
        if branch == "frontend" and method in self.methods:
            return self.root / "results" / method / uid
        if method not in self.gaussians:
            raise ValueError("Unknown Gaussian method")
        if branch == "gt":
            return self.gt / "results" / method / uid
        if branch in {"final_gt", "final_estimated"}:
            return self.gt / "final" / branch.removeprefix("final_") / method / uid
        raise ValueError("Unknown branch")

    def case(self, uid, method, branch="frontend"):
        directory = self.directory(uid, method, branch)
        status = read(directory / "status.json")
        geometry, matching = read(directory / "geometry.json"), read(directory / "matching.json")
        render = directory / "gaussian" if branch == "frontend" else directory
        validation = read(render / "val_metrics.json")
        if branch == "frontend":
            values = [
                r.get("foreground_psnr")
                for r in read(render / "render_metrics.json").get("views", [])
                if r["split"] == "val"
            ]
            validation = (
                {"foreground_psnr": float(np.mean(values))}
                if values and all(v is not None for v in values)
                else {}
            )
        pose = read(directory / "pose_metrics.json") or read(render / "pose_metrics.json")
        stages = sorted({p.parent.name for p in (directory / "model_inputs").glob("*/*.png")})
        state = status.get("state", "PENDING")
        gif = render / "reconstruction.gif"
        points = directory / "geometry.html"
        return {
            "uid": uid,
            "method": method,
            "label": LABELS[method],
            "branch": branch,
            "state": state,
            "stage": status.get("stage", ""),
            "error": status.get("error", ""),
            "registered": geometry.get("registered", status.get("registered")),
            "val_fg": validation.get("foreground_psnr"),
            "rotation": pose.get("rotation_median_deg"),
            "gif": self.url(gif) if gif.exists() and state == "COMPLETE" else None,
            "geometry": self.url(points) if points.exists() else None,
            "progress": read(render / "progress.json"),
            "training": read(render / "training.json"),
            "pair_count": len(matching.get("pairs", [])),
            "stages": stages,
            "correspondence_source": matching.get(
                "correspondence_source",
                "Native correspondence output"
                if method == "mast3r_slam"
                else "Saved pairwise correspondence output",
            ),
            "verification": matching.get(
                "verification",
                "F-MAGSAC geometric-consistency filter; not ground-truth correspondence labels",
            ),
        }

    def index(self):
        objects = []
        for obj in self.objects:
            inputs = read(self.root / "inputs" / obj["uid"] / "input_manifest.json")
            names = sorted(r["name"] for r in inputs.get("frames", []))
            objects.append(
                {
                    **obj,
                    "input_names": names,
                    "inputs": [
                        self.url(self.root / "inputs" / obj["uid"] / "images" / n) for n in names
                    ],
                    "pairs": [
                        {"key": Path(a).stem + "__" + Path(b).stem, "label": a + " ↔ " + b}
                        for a, b in itertools.combinations(names, 2)
                    ],
                }
            )
        return {
            "checked_at": time.time(),
            "objects": objects,
            "methods": [{"id": m, "label": LABELS[m]} for m in self.methods],
            "gaussians": [{"id": m, "label": LABELS[m]} for m in self.gaussians],
            "rows": [self.case(o["uid"], m) for o in self.objects for m in self.methods],
            "selection": read(self.root / "selection.json"),
            "execution": read(self.root / "execution_status.json"),
        }

    def pair(self, uid, method, key, stage="auto", limit=24):
        directory = self.directory(uid, method, "frontend")
        rows = read(directory / "matching.json").get("pairs", [])
        row = next(
            (r for r in rows if Path(r["first"]).stem + "__" + Path(r["second"]).stem == key), None
        )
        if row is None:
            return {
                "available": False,
                "message": "该方法尚未导出这对图的对应关系。保留缺失状态，不用其他图对替代。",
            }
        pairfile = directory / "matches" / row.get("file", key + ".npz")
        if not pairfile.exists():
            return {"available": False, "message": "图对记录已存在，匹配数组尚不可读。"}
        with np.load(pairfile) as arrays:
            first, second, mask = arrays["first"], arrays["second"], arrays["inliers"].astype(bool)
        selected = np.flatnonzero(mask)
        if len(selected) > limit:
            selected = selected[np.linspace(0, len(selected) - 1, limit).astype(int)]
        evidence = directory / "model_inputs"
        stages = (
            sorted(
                p.name
                for p in evidence.iterdir()
                if p.is_dir() and (p / row["first"]).exists() and (p / row["second"]).exists()
            )
            if evidence.exists()
            else []
        )
        if stage == "auto":
            stage = stages[-1] if stages else "reconstruction_rgb_512"
        if stage == "reconstruction_rgb_512":
            folder = self.root / "inputs" / uid / "images"
            note = "512×512重建入口RGB；此方法未导出该图对的逐组件输入证据。"
        elif stage in stages:
            folder = evidence / stage
            note = "实际模型输入的已记录像素；归一化浮点张量显示为RGB8，精确哈希见原始证据。"
        else:
            raise ValueError("Unknown model input stage")
        paths = [folder / row["first"], folder / row["second"]]
        sizes = [list(Image.open(p).size) for p in paths]
        return {
            "available": True,
            "first": row["first"],
            "second": row["second"],
            "images": [self.url(p) for p in paths],
            "sizes": sizes,
            "points": [
                [float(a[0]), float(a[1]), float(b[0]), float(b[1])]
                for a, b in zip(first[selected], second[selected])
            ],
            "coordinate_size": [512, 512],
            "candidate_count": len(mask),
            "inlier_count": int(mask.sum()),
            "shown_count": len(selected),
            "indices": selected.tolist(),
            "stage": stage,
            "stages": stages,
            "input_note": note,
            "pair_sha256": sha256(pairfile),
        }

    def pair_png(self, uid, method, key, limit=24):
        """Scientific plot from saved correspondences, for a static preview/download."""
        data = self.pair(uid, method, key, limit=limit)
        canvas = Image.new("RGB", (1024, 550), "white")
        draw = ImageDraw.Draw(canvas)
        if not data["available"]:
            draw.text((12, 12), LABELS[method] + ": pair output unavailable", fill="black")
        else:
            for i, url in enumerate(data["images"]):
                im = Image.open(self.resolve(url.removeprefix("/media/"))).convert("RGB")
                canvas.paste(im.resize((512, 512), Image.Resampling.LANCZOS), (i * 512, 38))
            colors = ["#0f8f83", "#c44983", "#4f6bc2", "#b46b10"]
            for i, (x, y, u, v) in enumerate(data["points"]):
                color = colors[i % len(colors)]
                draw.line((x, y + 38, u + 512, v + 38), fill=color, width=1)
                for xx, yy in [(x, y + 38), (u + 512, v + 38)]:
                    draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), outline=color, width=2)
            text = f"{LABELS[method]} | {data['first']} <-> {data['second']} | {data['inlier_count']}/{data['candidate_count']} inliers; shown {data['shown_count']}"
            draw.text((12, 10), text, fill="black")
        output = io.BytesIO()
        canvas.save(output, format="PNG")
        return output.getvalue()


class ViewerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, catalog, **kwargs):
        self.catalog = catalog
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass

    def respond(self, data, content_type="application/json; charset=utf-8", code=200):
        content = (
            data
            if isinstance(data, bytes)
            else json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        )
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def artifact(self, path):
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        if header := self.headers.get("Range"):
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
            if not match:
                return self.send_error(416)
            start = int(match[1])
            end = min(int(match[2]) if match[2] else size - 1, size - 1)
            if start > end or start >= size:
                return self.send_error(416)
            code = 206
        self.send_response(code)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_GET(self):
        route = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(route.query).items()}
        try:
            if route.path == "/health":
                return self.respond({"app": "reconlab-all-methods-v1", "time": time.time()})
            if route.path in {"/", "/dashboard.html", "/all_methods.html"}:
                return self.respond(
                    Path(__file__).with_name("live_viewer.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            if route.path == "/api/index":
                return self.respond(self.catalog.index())
            if route.path == "/api/case":
                return self.respond(
                    self.catalog.case(
                        query["uid"], query["method"], query.get("branch", "frontend")
                    )
                )
            if route.path in {"/api/pair", "/api/pair.png"}:
                limit = int(query.get("limit", "24"))
                if limit not in {12, 24, 50, 100}:
                    raise ValueError("Unsupported display count")
                if route.path.endswith(".png"):
                    return self.respond(
                        self.catalog.pair_png(query["uid"], query["method"], query["pair"], limit),
                        "image/png",
                    )
                return self.respond(
                    self.catalog.pair(
                        query["uid"],
                        query["method"],
                        query["pair"],
                        query.get("stage", "auto"),
                        limit,
                    )
                )
            if route.path.startswith("/media/"):
                return self.artifact(self.catalog.resolve(route.path.removeprefix("/media/")))
            self.send_error(404)
        except (ValueError, KeyError) as error:
            self.respond({"error": str(error)}, code=400)
        except FileNotFoundError:
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    catalog = ResultCatalog(args.config, args.gt_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(ViewerHandler, catalog=catalog))
    print(f"Listening on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

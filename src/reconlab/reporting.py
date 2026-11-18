"""Local HTML report based on actual run records, including failures."""

from __future__ import annotations

import html
import os
from pathlib import Path
from urllib.parse import quote

from .io import read_json


class HtmlReport:
    def write(self, run_dir: Path) -> Path:
        run = read_json(run_dir / "run.json")

        def link(path, label):
            relative = os.path.relpath(path, run_dir)
            return f'<a href="{quote(relative)}">{html.escape(str(label))}</a>'

        sections = [
            "<!doctype html><html lang='en'><meta charset='utf-8'><title>ReconLab run</title>",
            "<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}img{max-width:100%;max-height:320px}article{border-top:1px solid #ccc;padding:20px 0}code{white-space:pre-wrap}.gallery{display:flex;gap:12px;flex-wrap:wrap}.gallery figure{margin:0;max-width:30%}</style>",
            f"<h1>{html.escape(run['experiment'])}</h1>",
            f"<p>Status: {html.escape(run['status'])}. Requested stage: {html.escape(run['until'])}.</p>",
            "<p>Geometry protocol: all selected train/val/test images participate in SfM. Held-out images do not supervise radiance-field training. This is transductive pose estimation, not train-only reconstruction.</p>",
            link(run_dir / "config.json", "Frozen configuration"),
            " · ",
            link(run_dir / "run.json", "Run evidence"),
        ]
        for row in run["objects"]:
            sections.append(
                f"<article><h2>{html.escape(row['object_id'])}: {html.escape(row['status'])}</h2>"
            )
            if row.get("error"):
                sections.append(f"<pre>{html.escape(row['error'])}</pre>")
            object_dir = run_dir / row["object_id"]
            provenance = object_dir / "data" / "inputs.json"
            if provenance.exists():
                sections += [
                    "<h3>Actual staged model inputs</h3>",
                    link(provenance, "Full input mapping and hashes"),
                    "<div class='gallery'>",
                ]
                for view in read_json(provenance)["views"]:
                    relative = quote(os.path.relpath(view["model_input"], run_dir))
                    sections.append(
                        f"<figure><img loading='lazy' src='{relative}'><figcaption>{html.escape(view['view_id'])} · {view['split']} · {view['width']}×{view['height']}</figcaption></figure>"
                    )
                sections.append("</div>")
            summary = object_dir / "matching" / "summary.json"
            if summary.exists():
                sections += [
                    "<h3>Point matching</h3>",
                    link(summary, "All pair statistics"),
                    "<p>Plots show at most 150 correspondences; full matches are stored in HDF5. Green/red are diagnostic F-RANSAC inliers/outliers, not ground-truth labels.</p>",
                ]
                for pair in read_json(summary)["pairs"]:
                    if pair.get("visualization"):
                        relative = quote(os.path.relpath(pair["visualization"], run_dir))
                        sections.append(f"<img loading='lazy' src='{relative}'>")
            model = object_dir / "training" / "model.json"
            if model.exists():
                sections.append(link(model, "Exact trained model/checkpoint"))
            else:
                sections.append(
                    "<p>No trained model or reconstruction GIF is claimed for this object.</p>"
                )
            sections.append("</article>")
        sections.append("</html>")
        path = run_dir / "report.html"
        path.write_text("\n".join(sections))
        return path

"""Durable four-GPU queue; validation selection precedes full-budget training."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import queue
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core import save_json, sha256


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


class ValidationSelector:
    """Coverage first, then mean per-object validation foreground PSNR."""

    def rank(self, root, methods, objects, frontend=False):
        ranking, blockers = [], []
        for method in methods:
            scores, states = [], {}
            for obj in objects:
                p = root / "results" / method / obj["uid"]
                status = read(p / "status.json").get("state", "MISSING")
                states[obj["uid"]] = status
                if status in {"MISSING", "RUNNING", "SETUP_FAILED"}:
                    blockers.append({"method": method, "uid": obj["uid"], "state": status})
                if status != "COMPLETE":
                    continue
                if frontend:
                    rows = read(p / "gaussian/render_metrics.json").get("views", [])
                    values = [r["foreground_psnr"] for r in rows if r["split"] == "val"]
                    score = (
                        statistics.mean(values)
                        if values and all(v is not None and math.isfinite(v) for v in values)
                        else None
                    )
                else:
                    score = read(p / "val_metrics.json").get("foreground_psnr")
                if score is not None and math.isfinite(score):
                    scores.append({"uid": obj["uid"], "val_foreground_psnr": score})
            ranking.append(
                {
                    "method": method,
                    "coverage": len(scores),
                    "mean_val_foreground_psnr": statistics.mean(
                        r["val_foreground_psnr"] for r in scores
                    )
                    if scores
                    else None,
                    "objects": scores,
                    "states": states,
                }
            )
        ranking.sort(
            key=lambda r: (
                r["coverage"],
                r["mean_val_foreground_psnr"] if r["coverage"] else -math.inf,
            ),
            reverse=True,
        )
        winner = (
            ranking[0]["method"] if ranking and ranking[0]["coverage"] and not blockers else None
        )
        return {
            "winner": winner,
            "ranking": ranking,
            "blockers": blockers,
            "rule": "Maximize completed finite-validation coverage, then macro mean validation foreground PSNR. No test scores read. Missing/setup-failed runs block selection.",
        }


class SparseController:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.plan = read(snapshot / "plan.json")
        self.config = read(snapshot / "matching.json")
        self.task = Path(self.plan["task"])
        self.root, self.gt = Path(self.config["output"]), Path(self.plan["gt_root"])
        self.events, self.pending = queue.Queue(), queue.Queue()
        self.stop = threading.Event()
        self.results = []

    def verify(self):
        for name, expected in read(self.snapshot / "MANIFEST.json").items():
            assert sha256(self.snapshot / name) == expected, name
        for name, expected in self.plan["dependencies"].items():
            assert sha256(Path(name)) == expected, name
        for filename in self.plan["dependency_manifests"]:
            manifest = Path(filename)
            for name, expected in read(manifest).items():
                assert sha256(manifest.parent / name) == expected, name
        for name, expected in read(self.snapshot / "weights.json").items():
            assert sha256(Path(name)) == expected, name
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = [line.split(",") for line in output.strip().splitlines()]
        assert len(rows) == 4 and all(int(row[-1]) >= 12000 for row in rows), output
        save_json(
            self.task / "execution/devices.json", {"checked_at": time.time(), "devices": output}
        )

    def worker(self, index):
        while not self.stop.is_set():
            if (self.task / "CANCEL.json").exists():
                self.stop.set()
                break
            try:
                case = self.pending.get_nowait()
            except queue.Empty:
                break
            identifier = "_".join([case["kind"], case["method"], case.get("uid", "all10")])
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES=str(index), PYTHONPATH=str(self.snapshot / "src")
            )
            python = (
                self.plan["method_python"].get(case["method"], self.plan["python"])
                if case["kind"] in {"matching", "views"}
                else self.plan["python"]
            )
            if case["kind"] not in {"matching", "views"}:
                env["PYTHONPATH"] += ":" + self.plan["gsplat_source"]
            if case["method"] == "mast3r_slam":
                env["PYTHONPATH"] += ":" + self.plan["slam_pythonpath"]
            started = time.time()
            self.events.put({"event": "started", "id": identifier, "gpu": index, "time": started})
            logfile = self.task / "execution/cases" / (identifier + ".log")
            logfile.parent.mkdir(parents=True, exist_ok=True)
            with logfile.open("a") as out:
                result = subprocess.run(
                    [
                        python,
                        "-m",
                        "reconlab.benchmark.sparse_worker",
                        "--snapshot",
                        str(self.snapshot),
                        "--case",
                        json.dumps(case),
                    ],
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {
                "event": "finished",
                "id": identifier,
                "gpu": index,
                "returncode": result.returncode,
                "started_at": started,
                "finished_at": time.time(),
                "log": str(logfile),
            }
            save_json(logfile.with_suffix(".json"), record)
            self.events.put(record)
            self.pending.task_done()

    def phase(self, name, cases):
        for case in cases:
            self.pending.put(case)
        active, last_report = {}, time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.worker, index) for index in range(4)]
            while any(not f.done() for f in futures):
                try:
                    event = self.events.get(timeout=15)
                    if event["event"] == "started":
                        active[str(event["gpu"])] = event
                    else:
                        active.pop(str(event["gpu"]), None)
                        self.results.append(event)
                    print(json.dumps(event), flush=True)
                except queue.Empty:
                    pass
                save_json(
                    self.task / "execution/status.json",
                    {
                        "state": "RUNNING",
                        "phase": name,
                        "checked_at": time.time(),
                        "active": active,
                        "pending": self.pending.qsize(),
                        "finished": self.results,
                    },
                )
                if time.monotonic() - last_report > 600:
                    self.publish()
                    last_report = time.monotonic()
            for future in futures:
                future.result()
        while not self.events.empty():
            event = self.events.get_nowait()
            if event["event"] == "finished":
                self.results.append(event)

    def publish(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(self.snapshot / "src"))
        with (self.task / "execution/report.log").open("a") as out:
            result = subprocess.run(
                [
                    self.plan["python"],
                    "-m",
                    "reconlab.benchmark.report",
                    "--config",
                    str(self.snapshot / "matching.json"),
                    "--gt-root",
                    str(self.gt),
                ],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
            )
        save_json(
            self.task / "execution/report_exit.json",
            {"returncode": result.returncode, "checked_at": time.time()},
        )
        self.final_page()

    def final_page(self):
        selection = read(self.root / "selection.json")
        page = '<!doctype html><meta charset="utf-8"><title>6-view selected 30000-step results</title><style>body{font:16px system-ui;margin:24px}img{width:256px}td{padding:12px;border:1px solid #ddd}</style><h1>6图输入：选优后30000步结果</h1><p>筛选先比较完成覆盖率，再比较验证前景PSNR；测试集不参与选择。GT相机与估计相机支路分别列出。下列GIF均为72帧、512×512、3秒。</p>'
        page += (
            '<p><a href="dashboard.html">3000步筛选、点匹配与SfM对照</a></p><pre>'
            + html.escape(
                json.dumps(
                    {k: v.get("winner") for k, v in selection.items() if isinstance(v, dict)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre><table>"
        )
        for obj in self.config["objects"]:
            for kind in ["gt", "estimated"]:
                winner = selection.get("gaussian", {}).get("winner")
                p = self.gt / "final" / kind / str(winner) / obj["uid"]
                status = read(p / "status.json")
                metrics = read(p / "test_metrics.json")
                gif = p / "reconstruction.gif"
                image = (
                    '<img src="data:image/gif;base64,'
                    + base64.b64encode(gif.read_bytes()).decode()
                    + '">'
                    if gif.exists()
                    else ""
                )
                page += (
                    "<tr><td>"
                    + html.escape(obj["name"])
                    + "</td><td>"
                    + kind
                    + "</td><td>"
                    + html.escape(status.get("state", "PENDING"))
                    + "<br>"
                    + html.escape(str(metrics.get("foreground_psnr", "")))
                    + "</td><td>"
                    + image
                    + "</td></tr>"
                )
        path = self.root / "selected_final.html"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(page + "</table>")
        tmp.replace(path)

    def run(self):
        self.verify()
        self.phase("3000_step_screening", self.plan["screening_cases"])
        if self.stop.is_set():
            return
        selector = ValidationSelector()
        selected = {
            "gaussian": selector.rank(self.gt, self.plan["render_methods"], self.config["objects"]),
            "frontend": selector.rank(
                self.root, self.config["methods"], self.config["objects"], frontend=True
            ),
            "checked_at": time.time(),
        }
        save_json(self.root / "selection.json", selected)
        save_json(self.task / "evaluation/selection.json", selected)
        cases = []
        gaussian, frontend = selected["gaussian"]["winner"], selected["frontend"]["winner"]
        if gaussian:
            for obj in self.config["objects"]:
                cases.append({"kind": "gt_final", "method": gaussian, "uid": obj["uid"]})
                if frontend:
                    cases.append(
                        {
                            "kind": "estimated_final",
                            "method": gaussian,
                            "uid": obj["uid"],
                            "frontend": frontend,
                        }
                    )
        cases.extend({"kind": "views", "method": m} for m in self.config["view_ablation_methods"])
        self.publish()
        self.phase("selected_30000_steps_and_view_ablation", cases)
        self.publish()
        save_json(
            self.task / "execution/status.json",
            {
                "state": "EXECUTION_FINISHED_REVIEW_REQUIRED",
                "checked_at": time.time(),
                "finished": self.results,
                "selection": selected,
                "quality_conclusion": "Inspect coverage, failures and per-object images before claiming a best method.",
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    SparseController(parser.parse_args().snapshot).run()


if __name__ == "__main__":
    main()

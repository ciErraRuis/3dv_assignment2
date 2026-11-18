"""CPU diagnostics and a single-page, on-demand gallery for an entire batch."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .io import file_hash, read_json, utc_now, write_json
from .synthetic import PosedImageDataset


def mean_or_none(values):
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(valid)) if valid else None


class ForegroundMetrics:
    """The alpha mask is used for measurement only, never to change training RGB."""

    @staticmethod
    def measure(target: np.ndarray, prediction: np.ndarray, alpha: np.ndarray) -> dict:
        if target.shape != prediction.shape or target.shape[:2] != alpha.shape:
            raise ValueError("Metric inputs must be aligned")
        mask = alpha >= 128
        error = ((prediction.astype(np.float64) - target.astype(np.float64)) / 255) ** 2

        def psnr(region):
            if not np.any(region):
                return None
            return -10 * math.log10(max(float(error[region].mean()), 1e-12))

        target_visible = np.max(255 - target.astype(np.int16), axis=-1) > 10
        prediction_visible = np.max(255 - prediction.astype(np.int16), axis=-1) > 10
        # A diagnostic visible-colour proxy, not a geometric silhouette metric.
        colour_recall = (
            float(prediction_visible[target_visible].mean()) if target_visible.any() else None
        )
        return {
            "full_psnr_db": -10 * math.log10(max(float(error.mean()), 1e-12)),
            "foreground_psnr_db": psnr(mask),
            "background_psnr_db": psnr(~mask),
            "foreground_fraction": float(mask.mean()),
            "visible_colour_recall_proxy": colour_recall,
            "prediction_visible_fraction": float(prediction_visible.mean()),
            "target_visible_fraction": float(target_visible.mean()),
        }


class ObjectAnalyzer:
    def analyze(self, root: Path, uid: str) -> dict:
        inputs, result = root / "inputs" / uid, root / "results" / uid
        dataset = PosedImageDataset(inputs)
        model = read_json(result / "training/model.json")
        if file_hash(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        views = []
        frames = [x for x in dataset.frames if x["split"] == "test"]
        for index, frame in enumerate(frames):
            target_path = inputs / frame["file_path"]
            saved_input = result / f"evaluation/test_{index:02d}_input.png"
            with Image.open(target_path) as image:
                target = np.array(image)
            with Image.open(saved_input) as image:
                if not np.array_equal(target, np.array(image)):
                    raise ValueError("Report reference differs from actual model input")
            with Image.open(result / f"evaluation/test_{index:02d}_prediction.png") as image:
                prediction = np.array(image)
            with Image.open(inputs / "source_rgba" / target_path.name) as image:
                alpha = np.array(image.getchannel("A"))
            views.append(
                {
                    "index": index,
                    "source_view": frame["file_path"],
                    **ForegroundMetrics.measure(target, prediction, alpha),
                }
            )
        logs = [
            json.loads(line)
            for line in (result / "training/progress.jsonl").read_text().splitlines()
        ]
        losses = [x["loss"] for x in logs]
        best = min(losses)
        late = float(np.median(losses[-5:]))
        unstable = late > max(0.05, best + 0.05) and late > 3 * max(best, 0.001)
        full = mean_or_none([v["full_psnr_db"] for v in views])
        foreground = mean_or_none([v["foreground_psnr_db"] for v in views])
        orbit = []
        for index in range(72):
            with Image.open(result / "orbit" / f"frame_{index:05d}.png") as image:
                prediction = np.array(image)
                if image.size != (512, 512):
                    raise ValueError("Unexpected reconstruction frame size")
            ref = inputs / "reference_orbit" / f"frame_{index:05d}.png"
            rgba = inputs / "reference_rgba" / ref.name
            if ref.exists() and rgba.exists():
                with Image.open(ref) as image:
                    target = np.array(image)
                with Image.open(rgba) as image:
                    alpha = np.array(image.getchannel("A"))
                orbit.append(
                    {"index": index, **ForegroundMetrics.measure(target, prediction, alpha)}
                )
        nearly_missing = [
            x["index"]
            for x in orbit
            if x["target_visible_fraction"] > 0.002
            and x["prediction_visible_fraction"] < 0.1 * x["target_visible_fraction"]
        ]
        with Image.open(result / "reconstruction.gif") as image:
            gif_frames, duration = image.n_frames, 0
            for i in range(gif_frames):
                image.seek(i)
                if image.size != (512, 512):
                    raise ValueError("Unexpected GIF size")
                duration += image.info.get("duration", 0)
        if duration != 3000:
            raise ValueError("Unexpected GIF duration")
        flags = []
        if unstable:
            flags.append("late_loss_growth")
        if min(losses) < -1e-5:
            flags.append("negative_loss")
        if foreground is not None and foreground < 18:
            flags.append("large_foreground_error")
        if foreground is not None and full - foreground > 8:
            flags.append("background_inflates_psnr")
        if nearly_missing:
            flags.append("nearly_empty_orbit_views")
        if logs[-1]["gaussians"] > 160000:
            flags.append("large_gaussian_population")
        data = {
            "uid": uid,
            "checked_at": utc_now(),
            "checkpoint": model["checkpoint"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "input_sha256": file_hash(inputs / "inputs.json"),
            "optimizer_step": model["step"],
            "test_full_psnr_db": full,
            "test_foreground_psnr_db": foreground,
            "test_foreground_fraction": mean_or_none([v["foreground_fraction"] for v in views]),
            "test_visible_colour_recall_proxy": mean_or_none(
                [v["visible_colour_recall_proxy"] for v in views]
            ),
            "test_views": views,
            "orbit_views": orbit,
            "nearly_missing_orbit_indices": nearly_missing,
            "loss_first": losses[0],
            "loss_best": best,
            "loss_final": losses[-1],
            "loss_late_median": late,
            "loss_curve": [[x["optimizer_step"], x["loss"]] for x in logs],
            "training_seconds": model["metadata"]["training_seconds"],
            "gaussians_final": logs[-1]["gaussians"],
            "gif_encoded_frames": gif_frames,
            "gif_duration_ms": duration,
            "png_orbit_frames": 72,
            "flags": flags,
            "metric_protocol": "Saved 8-bit RGB; alpha>=128 defines foreground. White-distance>10 "
            "defines visible-colour proxy, not geometry. Flags are fixed screening "
            "heuristics, not human pass/fail judgments. Masks never affect training.",
        }
        write_json(result / "analysis.json", data)
        return data


class BatchGallery:
    def build(self, root: Path, catalog: Path) -> Path:
        rows = []
        for item in read_json(catalog)["objects"]:
            uid = item["id"]
            row = {
                "uid": uid,
                "name": item["name"],
                "category": item["category"],
                "license": item["license"],
                "author": item["author"].get("displayName"),
                "url": item["url"],
                "flags": [],
                "state": "pending",
                "error": "",
            }
            for phase in ("source", "gpu", "matching", "analysis"):
                path = root / "progress" / uid / f"{phase}.json"
                if path.exists():
                    state = read_json(path)
                    row[phase] = state["state"]
                    if state["state"] == "FAILED":
                        row["error"] += phase + ": " + state.get("error", "") + "\n"
            analysis = root / "results" / uid / "analysis.json"
            if analysis.exists():
                data = read_json(analysis)
                row.update(
                    {
                        k: data[k]
                        for k in [
                            "test_full_psnr_db",
                            "test_foreground_psnr_db",
                            "test_foreground_fraction",
                            "loss_final",
                            "loss_curve",
                            "training_seconds",
                            "gaussians_final",
                            "flags",
                            "gif_encoded_frames",
                            "nearly_missing_orbit_indices",
                        ]
                    }
                )
                row["state"] = "failed" if row["error"] else "complete"
            elif row["error"]:
                row["state"] = "failed"
            elif row.get("gpu") == "COMPLETE":
                row["state"] = "awaiting_analysis"
            elif row.get("source") or row.get("gpu"):
                row["state"] = "running"
            matching = root / "diagnostics" / uid / "matching/summary.json"
            if matching.exists():
                data = read_json(matching)
                row["mean_sift_keypoints"] = mean_or_none(list(data["keypoints"].values()))
                row["matching_pairs"] = len(data["pairs"])
                row["f_inliers"] = sum(x["diagnostic_f_inliers"] for x in data["pairs"])
                if row["mean_sift_keypoints"] < 30:
                    row["flags"].append("few_sift_keypoints")
            audit = root / "progress" / uid / "input_audit.json"
            if audit.exists() and read_json(audit)["empty_views"]:
                row["flags"].append("empty_input_views")
            assets = {
                "input": root / "inputs" / uid / "images/view_003.png",
                "source_gif": root / "inputs" / uid / "source_mesh_reference.gif",
                "reconstruction": root / "results" / uid / "reconstruction.gif",
                "test_input": root / "results" / uid / "evaluation/test_00_input.png",
                "test_prediction": root / "results" / uid / "evaluation/test_00_prediction.png",
                "matching": root / "diagnostics" / uid / "matching/pair_0000.png",
            }
            row["assets"] = {k: os.path.relpath(p, root) for k, p in assets.items() if p.is_file()}
            rows.append(row)
        completed = [x for x in rows if x["state"] == "complete"]
        summary = {
            "checked_at": utc_now(),
            "total": len(rows),
            "states": dict(Counter(x["state"] for x in rows)),
            "flags": dict(Counter(f for x in rows for f in x["flags"])),
            "completed_mean_full_psnr_db": mean_or_none(
                [x["test_full_psnr_db"] for x in completed]
            ),
            "completed_mean_foreground_psnr_db": mean_or_none(
                [x["test_foreground_psnr_db"] for x in completed]
            ),
            "note": "Aggregate appearance metrics cover completed cases only; failures remain in denominator/table. "
            "Flags can overlap and are not causal conclusions or human quality judgments.",
        }
        payload = {"summary": summary, "objects": rows}
        write_json(root / "batch_summary.json", payload)
        columns = [
            "uid",
            "name",
            "category",
            "state",
            "test_full_psnr_db",
            "test_foreground_psnr_db",
            "test_foreground_fraction",
            "loss_final",
            "gaussians_final",
            "mean_sift_keypoints",
            "flags",
            "error",
        ]
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({**x, "flags": ";".join(x["flags"])} for x in rows)
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        target = root / "report.html"
        page = _PAGE.replace("__DATA__", data)
        temporary = target.with_suffix(".html.tmp")
        temporary.write_text(page)
        temporary.replace(target)
        return target


_PAGE = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Objaverse 100 · 3DGS 批量诊断</title><style>
body{font:15px/1.6 system-ui;margin:24px auto;padding:0 20px;max-width:1400px;color:#203047;background:#f3f5f8}
section{background:white;border-radius:12px;padding:20px;margin:18px 0}button,input,select{padding:8px;margin:4px;font:inherit}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}figure{margin:0}img{max-width:100%;height:auto;background:#eee}
figcaption{font-weight:600;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}
tr[data-uid]{cursor:pointer}tr[data-uid]:hover{background:#e5eef7}.scroll{overflow:auto;max-height:520px}.muted{color:#65758a}pre{white-space:pre-wrap}a{color:#175fa9}
</style><h1>100 个 Objaverse 物体 · 同配置 3DGS 诊断</h1>
<p>48 张 512×512 静态输入（36训练 / 6验证 / 6测试）；白底、等比例归一化、完整画面。
gsplat 1.4.0，自定义已知位姿配方，随机初始化、3000步。每条轨迹72张PNG、名义24fps、3秒；GIF可合并相同帧。
原网格只用于生成输入与参照，未作为训练几何监督。SIFT匹配是独立诊断。本轮不包含NeRF或SfM。</p>
<p class="muted">前景指标使用原始RGBA的alpha≥128，训练输入仍为完整白底RGB。自动标记用于筛查，不能直接当成质量达标或因果结论。页面只加载当前选中样本的媒体；运行中可刷新查看最新结果。</p>
<p id="filehelp" hidden>若当前预览无法加载图片，请通过HTTP打开此目录；服务器本地入口为 http://127.0.0.1:8776/report.html，Remote SSH需转发8776端口。</p>
<section><h2>全量进度与问题分布</h2><pre id="summary"></pre><p><a href="summary.csv">下载全量CSV</a> · <a href="batch_summary.json">完整统计JSON</a></p></section>
<section><h2>全量样本</h2><input id="search" placeholder="搜索名称、UID、类别"><select id="state"><option value="">全部状态</option></select><select id="flag"><option value="">全部问题标记</option></select><span id="count"></span><div class="scroll"><table><thead><tr><th>名称 / UID</th><th>类别</th><th>状态</th><th>全图PSNR</th><th>前景PSNR</th><th>最终loss</th><th>标记</th></tr></thead><tbody id="table"></tbody></table></div></section>
<section id="detail"><button id="prev">上一条</button><button id="next">下一条</button><h2 id="title"></h2><p id="metadata"></p><pre id="errors"></pre><div class="grid" id="media"></div><h3>训练 loss</h3><svg id="loss" viewBox="0 0 800 200" style="width:100%;max-height:240px"></svg><p class="muted">曲线按该物体数值范围展示；负loss与后期增长单独标记，图间纵轴范围不相同。</p></section>
<script>const DATA=__DATA__;const rows=DATA.objects;let current=null,visible=rows;const $=x=>document.getElementById(x);
const flagNames={late_loss_growth:'后期loss升高',negative_loss:'出现负loss',large_foreground_error:'前景误差大',background_inflates_psnr:'白底抬高PSNR',nearly_empty_orbit_views:'部分环绕角度近乎空白',large_gaussian_population:'高斯数量较大',few_sift_keypoints:'SIFT点较少',empty_input_views:'输入含空白视角'};
const stateNames={pending:'待运行',running:'运行中',awaiting_analysis:'等待分析',failed:'阶段失败',complete:'产物与分析完成'};
function text(n,s){n.textContent=s}function val(x){return typeof x==='number'?x.toFixed(2):'—'}
text($('summary'),`更新时间：${DATA.summary.checked_at}\n总数 ${DATA.summary.total}；${Object.entries(DATA.summary.states).map(([k,v])=>(stateNames[k]||k)+' '+v).join(' / ')}\n已完成样本平均PSNR：全图 ${val(DATA.summary.completed_mean_full_psnr_db)} dB，前景 ${val(DATA.summary.completed_mean_foreground_psnr_db)} dB\n`+Object.entries(DATA.summary.flags).map(([k,v])=>(flagNames[k]||k)+'：'+v).join('；'));
for(const k of Object.keys(stateNames)){const o=new Option(stateNames[k],k);$('state').add(o)}for(const k of Object.keys(flagNames))$('flag').add(new Option(flagNames[k],k));
function select(uid){current=rows.find(x=>x.uid===uid);if(!current)return;let r=current;text($('title'),r.name);text($('metadata'),`${r.uid} · ${r.category} · ${stateNames[r.state]} · 作者 ${r.author||''} · 许可 ${r.license} · 前景PSNR ${val(r.test_foreground_psnr_db)} dB · 最终高斯 ${r.gaussians_final||'—'}`);text($('errors'),r.error||'');$('media').replaceChildren();const names={input:'实际训练输入示例',matching:'点匹配诊断',source_gif:'原网格参考 GIF（非重建）',reconstruction:'3DGS 重建 GIF',test_input:'实际留出测试输入',test_prediction:'同相机重建预测'};for(const [k,label]of Object.entries(names)){let f=document.createElement('figure'),c=document.createElement('figcaption');text(c,label);f.append(c);if(r.assets[k]){let a=document.createElement('a');a.href=r.assets[k];a.target='_blank';let img=document.createElement('img');img.src=r.assets[k];img.alt=label;img.width=512;a.append(img);f.append(a)}else{let p=document.createElement('p');text(p,'尚未生成');f.append(p)}$('media').append(f)}let svg=$('loss');svg.replaceChildren();let values=r.loss_curve||[];if(values.length){let lo=Math.min(...values.map(x=>x[1])),hi=Math.max(...values.map(x=>x[1]));if(hi===lo)hi=lo+1;let p=document.createElementNS('http://www.w3.org/2000/svg','polyline');p.setAttribute('points',values.map(x=>(50+x[0]/3000*730)+','+(175-(x[1]-lo)/(hi-lo)*150)).join(' '));p.setAttribute('fill','none');p.setAttribute('stroke','#147d93');p.setAttribute('stroke-width','2');svg.append(p);let t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x','10');t.setAttribute('y','15');text(t,`loss范围 ${lo.toPrecision(4)} ～ ${hi.toPrecision(4)}；0 → 3000步`);svg.append(t)}}
function filter(){let q=$('search').value.toLowerCase(),s=$('state').value,f=$('flag').value;visible=rows.filter(r=>(!s||r.state===s)&&(!f||r.flags.includes(f))&&[r.name,r.uid,r.category].join(' ').toLowerCase().includes(q));$('table').replaceChildren();for(const r of visible){let tr=document.createElement('tr');tr.dataset.uid=r.uid;for(const value of [r.name+' / '+r.uid.slice(0,8),r.category,stateNames[r.state],val(r.test_full_psnr_db),val(r.test_foreground_psnr_db),val(r.loss_final),r.flags.map(x=>flagNames[x]||x).join('、')]){let td=document.createElement('td');text(td,value);tr.append(td)}tr.onclick=()=>select(r.uid);$('table').append(tr)}text($('count'),`${visible.length} / ${rows.length} 个`);if(!current||!visible.some(x=>x.uid===current.uid)){if(visible.length)select(visible[0].uid)}}
for(const e of ['search','state','flag'])$(e).oninput=filter;function move(d){if(!visible.length)return;let i=visible.findIndex(x=>x.uid===current?.uid);select(visible[(i+d+visible.length)%visible.length].uid)}$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('filehelp').hidden=location.protocol!=='file:';filter();</script></html>"""

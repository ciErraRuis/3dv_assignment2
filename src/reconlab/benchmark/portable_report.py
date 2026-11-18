"""Static offline reports: exact image/GIF bytes embedded, no asset server or JS."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import uuid
import zipfile
from pathlib import Path


class EmbeddedAssets:
    """Resolve report references and retain source-byte provenance."""

    def __init__(self, root: Path, allowed_roots: list[Path]):
        self.root = root
        self.allowed_roots = [p.resolve() for p in allowed_roots]
        self.records = {}
        self.cache = {}

    def uri(self, relative: str) -> str:
        path = (self.root / relative).resolve()
        if not any(path.is_relative_to(parent) for parent in self.allowed_roots):
            raise ValueError(f"Asset outside experiment roots: {relative}")
        if relative not in self.cache:
            content = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.cache[relative] = f"data:{mime};base64," + base64.b64encode(content).decode(
                "ascii"
            )
            self.records[relative] = {
                "source": str(path),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "mime": mime,
            }
        return self.cache[relative]

    def image(self, relative: str, label: str, *, eager=False) -> str:
        uri = self.uri(relative)
        loading = "eager" if eager else "lazy"
        return (
            f'<figure><img loading="{loading}" src="{uri}" alt="{html.escape(label)}" '
            f'data-source="{html.escape(relative)}"><figcaption>{html.escape(label)}</figcaption></figure>'
        )

    def download(self, relative: str, label: str, filename: str | None = None) -> str:
        filename = filename or Path(relative).name
        return (
            f'<a download="{html.escape(filename)}" href="{self.uri(relative)}">'
            f"{html.escape(label)}</a>"
        )


class PortableReport:
    """One compact overview and ten independent, complete object pages."""

    def __init__(self, root: Path, gt_root: Path, extra_roots=()):
        self.root = root
        self.assets = EmbeddedAssets(root, [root, gt_root, *extra_roots])

    @staticmethod
    def write(path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)

    @staticmethod
    def table(rows, columns):
        def value(v):
            return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))

        return (
            '<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{value(row.get(k))}</td>" for k, _ in columns) + "</tr>"
                for row in rows
            )
            + "</tbody></table></div>"
        )

    @staticmethod
    def header(title, data):
        return (
            '<!doctype html><html lang="zh"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title>"
            "<style>body{font:16px system-ui,sans-serif;background:#f4f6f8;color:#222d3b;line-height:1.65;margin:0}"
            "main{max-width:1250px;margin:auto;padding:24px}section,article{background:white;border:1px solid #dce3eb;border-radius:10px;padding:20px;margin:18px 0}"
            "h1,h2,h3{line-height:1.35}a{color:#165caf}nav{display:flex;flex-wrap:wrap;gap:14px}"
            "figure{margin:12px 0}img{display:block;max-width:100%;height:auto;border:1px solid #e0e5ec;background:#fafafa}"
            "figcaption{font-size:13px;color:#526078;margin:5px 0}.media{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px;align-items:start}"
            ".media img{max-height:540px;object-fit:contain}.note{background:#fff3d3;padding:14px;border-radius:8px}"
            "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dde3eb;padding:9px;text-align:left}th{background:#edf2f8}.scroll{overflow:auto}"
            "details{margin:14px 0}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}"
            "@media(max-width:650px){main{padding:12px}section,article{padding:14px}.media{display:block}}"
            "</style></head><body><main>"
            f"<h1>{html.escape(title)}</h1><p>结果汇总时间（UTC）：{html.escape(data['checked_at'])}</p>"
            f'<p class="note">{html.escape(data.get("execution", {}).get("message", "阶段性结果，尚未完成最终方法选择。"))}</p>'
        )

    @staticmethod
    def footer():
        return "</main></body></html>"

    def card(self, row, branch):
        title = html.escape(row["method"])
        camera = "GT 相机训练" if branch == "gaussian_gt" else "由训练 RGB 估计相机"
        content = f'<article id="{branch}-{title}"><h3>{title}</h3><p>{camera} · {html.escape(row["state"])}</p>'
        content += self.table(
            [row],
            [
                ("registered", "注册图/关键帧数"),
                ("rotation_median_deg", "旋转中位误差°"),
                ("val_foreground_psnr", "验证前景PSNR"),
                ("test_foreground_psnr", "测试前景PSNR"),
                ("best_step", "验证选中step"),
                ("seconds", "执行秒数"),
            ],
        )
        if row.get("error"):
            content += f"<details><summary>执行信息</summary><code>{html.escape(row['error'])}</code></details>"
        content += '<div class="media">'
        matches = row.get("matching", [])
        if matches:
            content += self.assets.image(matches[0], f"{row['method']} · 实际点匹配")
        if row.get("gif"):
            content += self.assets.image(
                row["gif"], f"{row['method']} · {camera} · 72帧 / 512×512 / 3秒"
            )
        elif not matches:
            content += "<p>本项尚无可展示的匹配图或完整 GIF，保留执行状态供核查。</p>"
        content += "</div>"
        if len(matches) > 1:
            content += '<details><summary>其余匹配图</summary><div class="media">'
            content += "".join(
                self.assets.image(p, "实际匹配 · " + Path(p).name) for p in matches[1:]
            )
            content += "</div></details>"
        if row.get("model_inputs"):
            content += '<details><summary>实际送入模型的输入预览</summary><div class="media">'
            content += "".join(
                self.assets.image(p, "模型输入 · " + Path(p).parent.name)
                for p in row["model_inputs"]
            )
            content += "</div></details>"
        if row.get("geometry"):
            content += (
                "<p>"
                + self.assets.download(
                    row["geometry"],
                    "下载交互点云与相机页面",
                    f"{row['uid']}_{row['method']}_geometry.html",
                )
                + "（下载后用浏览器打开）</p>"
            )
        return content + "</article>"

    def object_page(self, obj, data):
        uid = obj["uid"]
        page = self.header(obj["name"] + " · 全部方法", data)
        page += f"<p>对象 UID：<code>{uid}</code></p><p>本页图片和GIF均内嵌原始文件；单独下载本页也可显示。</p>"
        page += '<nav><a href="../../dashboard.html">返回总览（离线包内）</a><a href="#reconstruction">图像重建</a><a href="#gaussian_gt">GT高斯</a><a href="#views">视角数量</a></nav>'
        for branch, title in [
            ("reconstruction", "图像 → 匹配 → 重建 → 统一高斯"),
            ("gaussian_gt", "固定GT相机的高斯对比"),
        ]:
            page += f'<section id="{branch}"><h2>{title}</h2><p>不同支路的相机来源与训练预算不同，分表比较。</p></section>'
            page += "".join(self.card(row, branch) for row in data[branch] if row["uid"] == uid)
        figure = f"analysis/{uid}_cameras.png"
        if (self.root / figure).exists():
            page += (
                "<section><h2>相机布局诊断</h2>"
                + self.assets.image(figure, "GT仅用于事后对齐；注册数不代表姿态准确")
                + "</section>"
            )
        rows = [r for r in data["view_ablation"] if r["uid"] == uid]
        page += '<section id="views"><h2>视角数量与分布</h2><p>每组独立运行SfM；这里没有为每个子集训练高斯。选图来自原36张训练池，实际索引保存在各组输入清单。</p>'
        page += self.table(
            rows,
            [
                ("method", "前端"),
                ("subset", "选图"),
                ("state", "状态"),
                ("registered", "注册数"),
                ("rotation_median_deg", "旋转中位误差°"),
            ],
        )
        geometry = [r for r in rows if r.get("geometry")]
        if geometry:
            page += "<details><summary>下载各子集的交互点云页面</summary><ul>"
            page += "".join(
                "<li>"
                + self.assets.download(
                    r["geometry"],
                    r["method"] + " / " + r["subset"],
                    f"{uid}_{r['method']}_{r['subset']}.html",
                )
                + "</li>"
                for r in geometry
            )
            page += "</ul></details>"
        return page + "</section>" + self.footer()

    def build(self, data, summary, gt_summary):
        pages = []
        for obj in data["objects"]:
            path = self.root / "portable/objects" / (obj["uid"] + ".html")
            self.write(path, self.object_page(obj, data))
            pages.append(path)
            self.assets.cache.clear()  # keep page construction bounded, retaining hashes
        index = self.header("十物体重建实验 · 离线可显示版", data)
        index += '<p>本页的图片、GIF和图表已直接内嵌，不需要图片服务器或JavaScript。查看每个物体的全部方法，请下载<a href="matching10_offline.zip" download>完整离线包</a>，解压后打开 dashboard.html；各物体单页也可独立下载。</p>'
        index += "<p>固定10个公开Objaverse样例；组号、成员分工和GitHub地址均待填写。尚未宣称使用作业1清洗数据。</p>"
        columns = [
            ("method", "方法"),
            ("complete", "完整结果/10"),
            ("statuses", "状态"),
            ("rotation", "对象旋转中位误差的均值°"),
            ("val_fg", "验证前景PSNR"),
            ("test_fg", "测试前景PSNR"),
        ]
        index += f"<section><h2>方法对比</h2><p>图像支路：{data.get('input_views', 36)}张512×512训练RGB，估计相机后统一训练3000步3DGS。注册失败/关键帧选择会改变实际参与训练的图数。只对完成项取均值，须同时看覆盖率，不能直接比较不同成功子集。</p>"
        index += self.table(summary, columns) + "</section>"
        index += (
            "<section><h2>GT相机高斯对比</h2><p>"
            + html.escape(data.get("gt_protocol", "按配置记录的固定GT相机协议比较。"))
            + "</p>"
        )
        index += self.table(gt_summary, [c for c in columns if c[0] != "rotation"]) + "</section>"
        index += "<section><h2>十物体匹配与重建预览</h2><p>总览固定展示GLUEMAP，进度与失败逐项列出；这不代表最终最佳方法。</p><nav>"
        index += (
            "".join(
                f'<a href="#object-{o["uid"]}">{html.escape(o["name"])}</a>'
                for o in data["objects"]
            )
            + "</nav></section>"
        )
        for obj in data["objects"]:
            uid = obj["uid"]
            row = next(
                r for r in data["reconstruction"] if r["uid"] == uid and r["method"] == "gluemap"
            )
            index += f'<section id="object-{uid}"><h2>{html.escape(obj["name"])}</h2><p><a href="portable/objects/{uid}.html">查看此物体全部方法（离线包内）</a> · {html.escape(row["state"])}</p><div class="media">'
            if row.get("matching"):
                index += self.assets.image(row["matching"][0], "GLUEMAP · 实际匹配图", eager=True)
            if row.get("gif"):
                index += self.assets.image(
                    row["gif"], "GLUEMAP · 估计相机重建GIF · 72帧、512×512、3秒", eager=True
                )
            index += "</div></section>"
        figure = "analysis/view_count_diagnostics.png"
        index += f"<section><h2>视角数实验：{len(data['view_ablation'])}/{data['view_target']}组记录</h2><p>包含失败项；未运行的方法不当作失败。旋转误差图只统计可用对象，格内n显示样本数。</p>"
        if (self.root / figure).exists():
            index += self.assets.image(figure, "参考图数量与取样范围的阶段性诊断")
        index += "</section><section><h2>下载统计与复核</h2><p>"
        index += " · ".join(
            self.assets.download(name, name)
            for name in [
                "reconstruction.csv",
                "gaussian_gt.csv",
                "view_ablation.csv",
                "comparison.json",
            ]
        )
        index += (
            "</p><p>内嵌媒体为原PNG/GIF字节，无重编码、无裁剪；页面缩放只影响显示。GIF帧数与时长不变。数据来源与SHA-256见离线包内 portable/asset_manifest.json。</p></section>"
            + self.footer()
        )
        self.write(self.root / "dashboard.html", index)
        manifest = {
            "checked_at": data["checked_at"],
            "format": "static embedded media, no external image requests, no JavaScript dependency",
            "assets": self.assets.records,
        }
        self.write(
            self.root / "portable/asset_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        archive = self.root / "matching10_offline.zip"
        temporary = archive.with_suffix(".tmp")
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as stream:
            for path in [
                self.root / "dashboard.html",
                *pages,
                self.root / "portable/asset_manifest.json",
            ]:
                stream.write(path, path.relative_to(self.root))
        temporary.replace(archive)
        self.assets.cache.clear()

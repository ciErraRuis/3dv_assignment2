"""CPU matching and an auditable gallery for the Objaverse pilot."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

from .data import IdentityPreprocessor
from .features import SIFTExtractor
from .frontend import MatchingFrontend
from .io import read_json, write_json
from .matching import FundamentalVerifier, RatioMatcher
from .selection import PairGenerator
from .synthetic import PosedImageDataset
from .types import ObjectViews, View


class PilotMatching:
    def run(self, inputs: Path, output: Path, objects: list[str]) -> None:
        for name in objects:
            dataset = PosedImageDataset(inputs / name)
            views = tuple(
                View(
                    Path(f["file_path"]).stem,
                    dataset.root / f["file_path"],
                    f["split"],
                    azimuth_deg=f["azimuth_deg"],
                )
                for f in dataset.frames
            )
            root = output / name
            prepared = IdentityPreprocessor().prepare(ObjectViews(name, views), root / "prepared")
            frontend = MatchingFrontend(
                SIFTExtractor(),
                RatioMatcher(),
                FundamentalVerifier(),
                PairGenerator("sequential", 2),
                max_visualizations=6,
            )
            frontend.run(prepared, root / "matching")


class PilotGallery:
    def build(self, root: Path, catalog: Path) -> Path:
        rows = []
        complete = True
        review_path = root / "quality_review.json"
        review = read_json(review_path) if review_path.exists() else None

        def image(path: Path, label: str, width: int = 400) -> str:
            if not path.is_file():
                return f"<p>{html.escape(label)}：尚未生成</p>"
            relative = html.escape(os.path.relpath(path, root), quote=True)
            return f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{relative}"><img loading="lazy" width="{width}" src="{relative}"></a></figure>'

        for item in read_json(catalog)["objects"]:
            name = item["id"]
            source, result = root / "inputs" / name, root / "results" / name
            if (
                not (result / "COMPLETE.json").is_file()
                or not (result / "reconstruction.gif").is_file()
            ):
                complete = False
            metrics = result / "evaluation" / "metrics.json"
            if metrics.exists():
                d = read_json(metrics)
                summary = f"验证 PSNR {d['val']['mean_psnr_db']:.2f} dB；测试 PSNR {d['test']['mean_psnr_db']:.2f} dB。"
            else:
                summary = "3DGS 训练或评估尚未完成。"
                complete = False
            if review and name in review["objects"]:
                item_review = review["objects"][name]
                summary += " 质量检查：" + item_review["description"]
                summary += (
                    f" 源序列72张PNG；GIF实际编码{item_review['gif_encoded_frames']}帧，总时长3秒。"
                )
            rows.append(
                f'<section><h2>{html.escape(name)} — {html.escape(item["name"])}</h2><p>{summary}</p><p>UID: {item["uid"]} · 作者：{html.escape(item["author"]["displayName"])} · Objaverse 许可标记：{html.escape(item["license"])}</p><a href="{html.escape(item["url"], quote=True)}">原模型来源</a>'
            )
            rows.append(
                '<div class="pair">'
                + image(source / "images/view_003.png", "实际输入示例：PNG，512×512")
                + image(
                    root / "diagnostics" / name / "matching/pair_0000.png",
                    "SIFT + ratio + F-RANSAC 点匹配；绿色为诊断内点",
                    600,
                )
                + "</div>"
            )
            rows.append(
                '<div class="pair">'
                + image(source / "source_mesh_reference.gif", "原始网格参考轨迹（不是重建结果）")
                + image(result / "reconstruction.gif", "从 RGB 学习得到的 3DGS 重建")
                + "</div>"
            )
            rows.append(
                '<div class="pair">'
                + image(result / "evaluation/test_00_input.png", "实际留出测试输入")
                + image(result / "evaluation/test_00_prediction.png", "同相机的重建预测")
                + "</div></section>"
            )
        status = (
            "三个样例均完成训练、评估和结果导出；完成不等于质量达标"
            if complete
            else "准备中：下方明确标注各项实际完成情况"
        )
        page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Objaverse 3DGS Pilot</title>
<style>body{font:16px/1.65 system-ui;margin:28px auto;max-width:1180px;padding:0 20px;color:#233040;background:#f4f6f9}section{background:white;border-radius:12px;padding:24px;margin:22px 0}.pair{display:flex;flex-wrap:wrap;gap:20px}figure{margin:12px 0}img{max-width:100%;height:auto;border:1px solid #e0e5eb}figcaption{margin-bottom:8px}p{max-width:1000px}</style>
<h1>Objaverse → 多视角图像 → 3DGS</h1>"""
        page += f"<p>{status}</p><p>每物体 48 张 512×512 静态图：36 训练 / 6 验证 / 6 测试。两圈俯角 20°、40°，轨迹俯角 30°；统一白底、等比例归一化、完整入镜。每条轨迹包含72张PNG，按24 fps名义帧率导出3秒GIF；GIF量化可合并相同显示帧，用40/50 ms时长近似24 fps。PNG序列为准，网页仅按CSS缩放展示。</p>"
        if review:
            page += "<p><strong>人工质量检查：</strong>" + html.escape(review["summary"]) + "</p>"
        page += "<p>这是已知位姿、随机初始化、固定 3000 步的 gsplat 1.4.0 试跑，不是最终作业的十物体实验，也不是 SfM 或消融比较。点匹配是独立诊断，本轮不用于相机或高斯初始化。PSNR 在完整白底图上计算，不能单独代表几何质量。无预训练模型、无网格顶点或深度监督。</p>"
        page += "".join(rows) + "</html>"
        target = root / "report.html"
        target.write_text(page)
        write_json(
            root / "report_status.json",
            {
                "all_reconstructions_complete": complete,
                "objects": [i["id"] for i in read_json(catalog)["objects"]],
                "quality_review": review,
            },
        )
        return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--match", action="store_true")
    args = parser.parse_args()
    objects = [item["id"] for item in read_json(args.catalog)["objects"]]
    if args.match:
        PilotMatching().run(args.root / "inputs", args.root / "diagnostics", objects)
    print(PilotGallery().build(args.root, args.catalog))


if __name__ == "__main__":
    main()

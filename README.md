# Assignment 2 — 3D Reconstruction and Rendering

多视角图像 → RoMa v2匹配与前景、置信度过滤 → SfM → 3D Gaussian Splatting。

- **Pipeline documentation：** [docs/assignment2_report.md](docs/assignment2_report.md)，包含实现、分析及成员分工。
- **Prompts：** [prompts/user_requests.json](prompts/user_requests.json)。

## 推理示例

使用仓库自带的[Doll多视角图像](assets/doll/images)，从6张RGB图像生成匹配结果、相机与点云以及渲染GIF。以下命令适用于配套服务器环境；其他机器需替换绝对路径，并按[示例说明](assets/doll/README.md)准备依赖、RoMa权重和源码缓存。

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/afs/260010098/data/envs/3dv-matching/bin/python \
  /mnt/afs/260010098/code/3dv_assignment/scripts/infer.py \
  --images /mnt/afs/260010098/code/3dv_assignment/assets/doll/images \
  --roma-source /mnt/afs/260010098/code/experiment_snapshots/20261115_matching10_stageb_r01/thirdparty/romav2 \
  --torch-home /mnt/afs/260010098/data/3dv_assignment/matching10_r01/cache/torch \
  --gsplat-extension /mnt/afs/260010098/data/3dv_assignment/matching10_r01/cache/extensions/gsplat_cuda/gsplat_cuda.so \
  --output /mnt/afs/260010098/data/3dv_assignment/doll_inference \
  --steps 3000
```

渲染结果保存到 `/mnt/afs/260010098/data/3dv_assignment/doll_inference/gaussian/centered.gif`。重复运行时请更换输出目录。

## 十物体展示

每行两个物体，共5行4列，依次为“匹配连线、渲染GIF、匹配连线、渲染GIF”。

![10个物体的匹配与高斯渲染总览](assets/showcase_overview.gif)

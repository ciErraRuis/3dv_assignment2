# Doll 多视角推理示例

这里的6张512×512 RGB图像，是README中Doll展示实际使用的训练输入。UID为`6f98c7c8ef0d456f8ce0a76e406c4696`，视图编号为001、009、017、025、033、041；保留完整画面与白色背景，没有新增裁剪或缩放。`manifest.json`记录逐图SHA-256。

来源：[Doll模型页面](https://sketchfab.com/3d-models/6f98c7c8ef0d456f8ce0a76e406c4696)。源元数据的许可标记为`by`，请遵循该模型页面的署名与许可要求。仓库只提供这些派生RGB图像，不包含源网格或预训练网络权重。

本示例不带GT相机或深度。推理入口会从图像重新求对应、估计相机和点云、优化该物体的3DGS，再渲染居中GIF。完整运行需要CUDA、pycolmap、gsplat、RoMa v2.0.1源码与权重，以及固定版本的DINOv3源码缓存。环境版本见[实验依赖](../../experiments/roma100/provenance/runtime_versions.json)，源代码版本见[RoMa版本记录](../../experiments/roma100/provenance/upstream_sources.json)。

CPU输入核验（不加载网络、不创建训练结果）：

```bash
/mnt/afs/260010098/data/envs/3dv-matching/bin/python \
  /mnt/afs/260010098/code/3dv_assignment/scripts/infer.py \
  --images /mnt/afs/260010098/code/3dv_assignment/assets/doll/images \
  --check-inputs
```

在配套GPU环境中完整运行：

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

以上绝对路径对应现有服务器。迁移机器时改为本机的仓库、Python环境、源码、缓存与输出路径；`--gsplat-extension`只适用于匹配PyTorch/CUDA/GPU的已编译扩展，其他环境可省略此参数并使用本机兼容的gsplat安装。脚本要求权重已存在，不隐式下载。Torch Hub缓存中须包含`hub/checkpoints/romav2.0.1.pt`以及DINOv3提交`adc254450203739c8149213a7a69d8d905b4fcfa`对应的源码目录。

主要输出为`matches`中的全部图像对对应、`visuals`中的匹配连线图、`geometry.npz`中的估计相机与点云，以及`gaussian/gaussians.pt`、`gaussian/centered.gif`与72张渲染PNG。它不读取原实验的匹配、相机或高斯结果，也不计算需要留出图和GT相机的评价指标。已有输出目录会被拒绝，重新运行请指定新目录。

默认3000步与README主展示预算一致；`--steps 15000`可使用更长优化预算。此单例入口保存训练终点，逐3000步保存的正式图数实验由`experiments/view_count`实现。新入口复用已运行的算法实现；本次仓库整理核验了CPU输入与接口，未把它记作一次新GPU实验。

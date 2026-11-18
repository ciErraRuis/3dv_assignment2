# Doll 多视角推理示例

这里的6张512×512 RGB图像，是README中Doll展示实际使用的训练输入。UID为`6f98c7c8ef0d456f8ce0a76e406c4696`，视图编号为001、009、017、025、033、041；保留完整画面与白色背景，没有新增裁剪或缩放。`manifest.json`记录逐图SHA-256。

来源：[Doll模型页面](https://sketchfab.com/3d-models/6f98c7c8ef0d456f8ce0a76e406c4696)。源元数据的许可标记为`by`，请遵循该模型页面的署名与许可要求。仓库只提供这些派生RGB图像，不包含源网格或预训练网络权重。

本示例不带GT相机或深度。推理入口会从图像重新求对应、估计相机和点云、优化该物体的3DGS，再渲染居中GIF。完整运行需要CUDA、pycolmap、gsplat、RoMa v2.0.1源码与权重，以及固定版本的DINOv3源码缓存。环境版本见[实验依赖](../provenance/runtime_versions.json)，源代码版本见[RoMa版本记录](../provenance/upstream_sources.json)。

以下命令均在**仓库根目录**执行，路径相对于仓库根目录。先激活已安装所需依赖的Python环境。

CPU输入核验（不加载网络、不创建训练结果）：

```bash
python scripts/infer.py \
  --images assets/doll/images \
  --check-inputs
```

完整运行前，准备以下外部依赖（不随仓库提供）：

- 将固定版本的RoMa v2源码放在 `third_party/romav2`。
- 将RoMa权重放在 `weights/torch/hub/checkpoints/romav2.0.1.pt`。
- 将DINOv3源码缓存放在 `weights/torch/hub/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa`，其中须包含 `hubconf.py`。
- 安装与当前PyTorch、CUDA和GPU兼容的gsplat；默认使用本机安装的CUDA扩展。

准备好GPU环境后完整运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/infer.py \
  --images assets/doll/images \
  --roma-source third_party/romav2 \
  --torch-home weights/torch \
  --output outputs/doll \
  --steps 3000
```

源码与缓存若放在其他位置，请修改命令中的相对路径。脚本检查RoMa权重SHA-256，不隐式下载权重。若需加载自行编译且兼容当前环境的CUDA扩展，可额外指定 `--gsplat-extension path/to/gsplat_cuda.so`；常规运行省略该参数。

主要输出为`matches`中的全部图像对对应、`visuals`中的匹配连线图、`geometry.npz`中的估计相机与点云，以及`gaussian/gaussians.pt`、`gaussian/centered.gif`与72张渲染PNG。它不读取原实验的匹配、相机或高斯结果，也不计算需要留出图和GT相机的评价指标。已有输出目录会被拒绝，重新运行请指定新目录。

默认3000步与README主展示预算一致；`--steps 15000`可使用更长优化预算。此单例入口保存训练终点，匹配、过滤、SfM和渲染直接调用`src/reconlab/benchmark`，不需要批量实验配置。新入口复用已运行的算法实现；本次仓库整理核验了CPU输入与接口，未把它记作一次新GPU实验。

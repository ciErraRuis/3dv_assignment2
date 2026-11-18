# RoMa v2 → SfM → 3DGS：实际实验实现

此目录归档生成报告结果的科学代码、参数与依赖版本。`source_manifest.json`逐文件记录原位置与SHA-256；清理时将相同的算法模块合并到仓库`src`，并更新仓库内导入路径；原运行快照保持不变。它仍保留原服务器路径，并非已在全新机器验证的一键安装包。

## 模块

| 文件 | 作用 |
| --- | --- |
| `runtime.py` | 四个独立worker，每组25物体；匹配、SfM、高斯优化、指标和GIF |
| `filtering.py` | 边界白色背景mask、overlap硬阈值、过滤审计、原对应缓存 |
| `geometry.py` | 原10例三组过滤对照中的几何重算 |
| `sources.py` | 在批准的固定输入协议下生成缺失的多视角图 |
| `presentation.py` | 根据训练高斯与相机生成居中直立环绕轨迹 |
| `../../src/reconlab/benchmark` | 实际匹配适配器、轨迹、pycolmap重建、3DGS训练和指标 |
| `../../src/reconlab/backends/blender_source.py` | 实际源图渲染实现 |
| `../../configs/roma100.json` | 100个UID、来源及GLB哈希、6图名单、四组划分、路径和冻结参数 |
| `provenance` | 实际环境版本与上游源码提交记录 |

通用可扩展接口另见仓库主目录的 `src/reconlab/interfaces.py`。本次批实验使用上表中的具体实现，不能把框架中尚未验证的适配器算作已完成实验。

## 固定配置

- RoMa v2 `romav2.0.1.pt`，SHA-256 `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`。权重与第三方代码分别从作者官方发布获取，不把它们混入本项目原创代码。
- Python 3.11、PyTorch 2.7.1+cu126、pycolmap 3.13.0、gsplat 1.4.0；完整依赖记录在 `provenance/runtime_versions.json`。
- 源图512×512；RoMa precise内部800/1280；6训练、6验证、6测试；15对匹配，每对原始候选5000。
- 原RGB通道均≥250为近白候选，边界4连通背景排除，前景膨胀1像素；两端均在前景且overlap≥0.5才保留；不回填拒绝点。
- 共享SIMPLE_PINHOLE内参，SfM估计焦距及位姿；3DGS优化3000步、seed42，主实验不联合优化相机。
- 当前100例使用原3DGS后端。GT相机对照中的3DGS-MCMC不是这100例的后端。

## 复现范围与运行前提

原运行实例使用配置中的冻结路径、数据缓存、已编译CUDA扩展及4张H100。`runtime.py run`依赖已完成的原10例比较及选择记录；`worker`入口仍依赖相应输入、缓存和输出布局。不要在原结果目录直接重复启动实验。

迁移机器时需要显式配置数据/权重/第三方仓库/输出路径，仓库算法与源图子进程已统一从 `src` 导入，并为目标PyTorch、CUDA和GPU重新构建gsplat扩展。路径及二进制变化须作为新运行记录保存；原服务器扩展哈希不能用作另一环境的通用兼容性保证。本归档未声称已验证这些迁移步骤。

报告可以在原环境仅使用CPU重新汇总，无需重新训练：

```bash
/mnt/afs/260010098/data/envs/3dv-matching/bin/python /mnt/afs/260010098/code/3dv_assignment/scripts/build_assignment_report.py
```

该命令读取原实验记录，更新报告时间快照并导出README媒体。指标CSV在 `assets/tables`，10例完整过滤后点匹配、过滤统计和输入溯源在 `assets/showcase`。报告区分未运行、算法失败、已完成与人工质量判断，不按结果替换UID。

## 数据、许可与提交信息

按报告的数据来源前提，输入来自作业1清洗后保留的Objaverse物体，其多视角图像由合成渲染生成。各模型的来源及许可在 `../../configs/roma100.json` 和README展示卡片中列出；第三方方法的源码、权重按各自许可证使用。归档不包含账号凭据、调度认证代码或模型大权重。

组号、成员贡献和实际GitHub URL待填写。提示词记录收录现有对话中可追溯的相关请求，尚不能保证覆盖全部历史调用。

面向单个RGB图像目录的入口与Doll asset示例见[运行说明](../../assets/doll/README.md)。源码清单中的`sha256`保留原运行文件哈希，`export_sha256`（若存在）表示仅路径整理等仓库修改后的文件哈希。

# 参考视角数量对照

本目录保留报告Analysis第3.5节新增实验的实际科学代码。固定10个展示物体，比较6、12、24、36张训练图；均匀覆盖与连续相邻取样分别测试，36张共用。各组独立求解SfM及3DGS，只复用原始匹配预测。均匀6张组复用相同配置的15k训练结果。

源图512×512，RoMa v2 precise内部800/1280；前景mask与overlap≥0.5过滤；固定6张验证、6张测试。每组15000步，每3000步保存。GIF为72帧、512×512、3秒。固定总步数不等于每张图获得相同优化次数。

| 文件 | 内容 |
| --- | --- |
| `viewsets.py` | 两种嵌套取样与训练/留出图隔离 |
| `inputs.py` | 输入哈希核验、子集目录与公共展示轨迹 |
| `pair_cache.py` / `filtering.py` | 原始匹配缓存、mask与分数过滤 |
| `gaussian.py` | 实际15k训练、每3k回调与留出视角评估 |
| `runtime.py` | 独立SfM、训练、checkpoint保存及四卡物体分片 |
| `../../configs/view_count.json` | 固定UID、48张输入哈希、模型checkpoint及完整参数 |
| `source_manifest.json` | 归档代码的原位置与SHA-256 |

完整逐阶段指标见[CSV](../../assets/tables/view_count_15k_milestones.csv)，报告分析的逐物体证据见[JSON](../../assets/tables/view_count_analysis.json)。失败组合记入分析JSON，不会作为缺行从成功率分母中删除；CSV仅包含已完成评估的阶段。

这是实际运行代码的归档，保留原服务器路径，并非已验证的跨机器一键运行包。其匹配/SfM依赖代码已归档于`../../src/reconlab/benchmark`；原6图15k基线、48视角数据、RoMa权重和gsplat原生扩展保存在原实验目录，未放入Git。迁移时须配置这些依赖与路径。原环境的任务冻结与提交脚本不属于算法实现，未纳入提交。提交器、账号认证和后台同步程序不属于科学管线，未收录。

本目录不修改README现有的10例3k展示GIF。新15k图数对照用于补充分析，与旧3k预算实验区分。

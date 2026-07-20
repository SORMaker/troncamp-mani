# TronCamp Mani：统一 T1–T4 ACT 方案

> 主办方原始安装、任务和提交流程见 [README_TRONCAMP.md](./README_TRONCAMP.md)。

## 1. 项目介绍、任务配置与成绩

本项目面向 TronCamp Mani 赛道，使用 ACT（Action Chunking Transformer）完成 Tron2 双臂机器人的四项递进式操作任务。最终版本将 T1–T4 合并在同一套代码中：数据处理、模型实现、训练循环和部署接口全部复用，仅通过独立命名的训练脚本和部署配置区分任务参数。

| 赛道 | 任务 | 数据量 | 训练入口 | ACT 配置（chunk / hidden / FFN） | Epochs | 训练设备 | 成绩 |
| --- | --- | ---: | --- | --- | ---: | --- | ---: |
| T1 | `adjust_bottle` | 200 | `train_T1.sh` | 50 / 512 / 3200 | 6000 | 1 GPU | ✓ 59 |
| T2 | `grab_roller` | 200 | `train_T2.sh` | 50 / 512 / 3200 | 6000 | 1 GPU | ✓ 66 |
| T3 | `stack_bowls_two` | 400 | `train_T3.sh` | 100 / 768 / 3200 | 6000 | 1 GPU | ✓ 74 |
| T4 | `stack_bowls_three` | 1021 | `train_T4.sh` | 100 / 1024 / 4096 | 8000 | 3 GPU DDP | **64.3** |

四项任务共用以下训练参数：

- Policy：ACT，16-D 双臂状态与动作；
- Cameras：`cam_high`、`cam_right_wrist`、`cam_left_wrist`；
- Batch size：8/GPU；
- Learning rate：`1e-5`；
- KL weight：10；
- Optimizer：AdamW；
- Scheduler：CosineAnnealingLR，最低学习率 `1e-6`。

T4 最终采用 T3 稳定代码路径和 T4 模型结构/权重。训练共运行 8000 epochs，最佳验证权重出现在 epoch 6400，验证损失为 0.024073。

### 使用入口

```bash
# 1. 数据采集（按表格替换 task/config/GPU）
bash collect_data.sh <task> <task_config> <gpu_id>

# 2. 转换为 ACT 数据格式
(
  cd external/robotwin_local/policy/ACT
  bash process_data.sh <task> <task_config> <episode_num>
)

# 3. 按赛道训练
bash external/robotwin_local/policy/ACT/train_T1.sh 0 0
bash external/robotwin_local/policy/ACT/train_T2.sh 0 0
bash external/robotwin_local/policy/ACT/train_T3.sh 0 0
bash external/robotwin_local/policy/ACT/train_T4.sh 0 0,1,2

# 4. 本地评测（示例：T4）
python starter/eval_local.py --track T4 --ckpt-dir <checkpoint_dir>
```

四个部署配置分别为 `deploy_policy_T1.yml` 至 `deploy_policy_T4.yml`；本地评测会根据 `--track` 自动选择对应配置。

## 2. 项目结构与创新点

### 项目结构

```text
troncamp-mani/
├── README.md                         # 项目主页
├── README_TRONCAMP.md                # 主办方原始说明
├── collect_data.sh                   # 统一数据采集入口
├── external/robotwin_local/
│   ├── task_config/                  # T1–T4 数据采集/评测配置
│   └── policy/ACT/
│       ├── process_data.py           # ACT 数据转换
│       ├── imitate_episodes.py       # 共享训练循环
│       ├── train.sh                  # 公共训练启动器
│       ├── train_T1.sh ... train_T4.sh
│       └── deploy_policy_T1.yml ... deploy_policy_T4.yml
├── starter/eval_local.py             # 本地评测入口
└── submit/submit.py                   # 比赛提交入口
```

### 创新点

1. **统一的 T1–T4 训练与部署框架**

   四项任务共享同一套实现，任务差异集中在薄配置层，避免维护四份重复代码。训练脚本和部署 YAML 按赛道明确命名，并保证模型结构与 checkpoint 一致。

2. **通用的训练稳定性改进**

   在共享训练循环中加入余弦学习率调度、可配置验证频率、权重级断点续训、最佳 checkpoint 原子保存、NaN/Inf 快速失败和续训 epoch 统计修复。这些改进不绑定具体任务，可直接复用于 T1–T4。

3. **T3 稳定代码与 T4 权重的兼容集成**

   最终方案对齐 T4 的 1024 hidden dimension、4096 FFN 和 100-step action chunk，并严格保持三路相机顺序、16-D 动作定义及数据归一化统计一致，完成 230.32M 参数 checkpoint 的全键匹配加载。

ACT/CVAE/Transformer、RoboTwin、Tron2 资产、DDP 基础能力及官方评测协议来自主办方或上游项目；本项目的工作重点是统一配置、训练可靠性改进和最终方案集成。

## 3. 总结

本项目以最小的任务专用改动构建了一个完整的 T1–T4 合并版本。四项任务复用统一的数据、训练、部署和评测流程，并通过独立配置保持参数清晰可追踪。最终成绩依次为 **T1 ✓59、T2 ✓66、T3 ✓74、T4 64.3**。

T4 最终采用经过验证的 T3 稳定执行路径加载 T4 权重，在有限时间内兼顾了模型容量、工程稳定性和交付可复现性。

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

#### 面向长周期训练的稳定性与可恢复性改造

T4 需要进行数千 epochs 的长周期训练。原始训练流程一旦出现进程中断、数值异常或最佳权重没有及时落盘，就可能损失数小时甚至数天的计算结果。本项目没有修改 ACT 的网络主体，而是围绕训练可靠性重构了训练循环；同一套改进也可以直接用于 T1–T4。

| 原始训练方式 | 改进后 | 优势 |
| --- | --- | --- |
| 学习率在整个训练过程中保持固定 | 使用 `CosineAnnealingLR` 从 `1e-5` 平滑衰减至 `1e-6` | 前期保持学习速度，后期以更小步长细化参数，减少长训练末期震荡 |
| 每个 epoch 都执行完整验证 | 通过 `--val_freq` 控制验证间隔，并保证最后一个 epoch 必验 | 减少重复验证的计算与数据读取开销，把更多时间用于训练 |
| 中断后只能从头开始 | 通过 `--resume_ckpt` 和 `--start_epoch` 恢复模型，并同步学习率调度进度 | 机器重启或任务异常后可以继续训练，降低长周期实验的重跑成本 |
| 最佳模型主要保存在内存中，训练结束后再统一写盘 | 验证指标改善时立即写入临时文件，再用 `os.replace` 原子更新 `policy_best.ckpt` | 即使随后异常退出，也能保留最近一次最佳权重，并避免写盘中断产生半文件 |
| NaN/Inf 可能继续传播到后续 batch | 每个 batch 检查 loss 是否有限，异常时输出 epoch、batch、rank 和各损失分量 | 快速定位数据或数值问题，避免 GPU 长时间运行无效训练 |
| epoch 统计默认训练从 0 开始 | 使用当前 epoch 最新 batch 的尾部切片汇总损失 | 从非零 epoch 续训时不再出现空切片或错误统计 |
| 日志只显示当前验证损失 | 同时记录当前/历史最佳 epoch、最佳 loss 和实时学习率 | 更容易判断收敛状态并选择最终 checkpoint |

这些改动的核心优势是把一次“必须从头跑完”的训练，变成一个**可中断、可恢复、可诊断、最佳结果可持续保存**的训练过程。它不依赖具体任务或模型规模，因此比单独调某个任务的超参数更具复用价值。

## 3. 总结

本项目以最小的任务专用改动构建了一个完整的 T1–T4 合并版本。四项任务复用统一的数据、训练、部署和评测流程，并通过独立配置保持参数清晰可追踪。最终成绩依次为 **T1 ✓59、T2 ✓66、T3 ✓74、T4 64.3**。

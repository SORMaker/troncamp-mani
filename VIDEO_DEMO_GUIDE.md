# TronCamp Mani 代码创新点与演示视频指南

本文用于说明本项目相对主办方初始 ACT 训练代码所做的修改，并作为最终演示视频的录制提纲。建议视频控制在 **3 分钟以内**，重点展示代码思路、关键改动和实际运行入口，不播放完整训练过程。

## 1. 项目与成绩概览

本项目使用一套共享的 ACT 训练代码完成 T1–T4 四项任务，最终成绩为：

**T1: 59 · T2: 66 · T3: 74 · T4: 64.3**

项目沿用主办方提供的 ACT/CVAE/Transformer、RoboTwin、Tron2 资产、DDP 和评测协议。主要代码工作集中在长周期训练的稳定性、可恢复性和实验可追踪性。

## 2. 修改代码总览

| 文件 | 主要修改 |
| --- | --- |
| `external/robotwin_local/policy/ACT/imitate_episodes.py` | 学习率调度、验证频率、断点续训、最佳权重保存、数值检查和训练统计修复 |
| `external/robotwin_local/policy/ACT/detr/main.py` | 注册 `val_freq`、`resume_ckpt` 和 `start_epoch` 参数 |
| `external/robotwin_local/policy/ACT/train.sh` | 统一训练启动器，集中生成 ACT 训练参数和单卡/DDP 命令 |
| `external/robotwin_local/policy/ACT/train_T1.sh` … `train_T4.sh` | 保存各任务的数据规模和模型配置差异 |
| `external/robotwin_local/policy/ACT/deploy_policy_T1.yml` … `deploy_policy_T4.yml` | 保存各任务对应的部署模型结构 |
| `starter/eval_local.py` | 根据 `--track` 自动选择相应的部署配置 |
| `submit/submit.py` | 打包时排除日志和训练产物，避免提交冗余文件 |

## 3. 核心创新：长周期训练的稳定性与可恢复性

### 3.1 余弦学习率调度

原始训练流程使用固定学习率。长周期训练后期继续使用相同步长，容易产生损失震荡。本项目在共享训练循环中加入余弦退火：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
)
```

每个 epoch 结束后更新并记录学习率：

```python
scheduler.step()
print(f"LR: {scheduler.get_last_lr()[0]:.8f}")
```

优势：训练前期保持较大的更新幅度，后期逐渐减小步长，使长周期训练具有更平滑的收敛过程。该实现位于公共训练代码中，因此 T1–T4 都能直接使用。

### 3.2 可配置验证频率

原始流程每个 epoch 都运行完整验证。对于数千 epochs 的训练，这会产生大量重复验证开销。本项目新增 `--val_freq`：

```python
should_validate = (epoch % val_freq == 0) or (epoch == num_epochs - 1)
if is_main and should_validate:
    # validation
```

优势：可以按任务规模设置验证间隔，同时保证最后一个 epoch 一定执行验证。在不改变训练目标的情况下，减少验证计算和数据读取时间。

### 3.3 权重级断点续训

长时间训练可能因为机器重启、任务超时或进程异常而中断。本项目增加：

```text
--resume_ckpt <checkpoint-path>
--start_epoch <epoch>
```

训练启动时恢复模型权重，并从指定 epoch 继续：

```python
state_dict = torch.load(resume_ckpt, map_location="cpu")
policy.load_state_dict(state_dict)

for epoch in range(start_epoch, num_epochs):
    ...
```

学习率调度器也推进到相同 epoch，避免恢复后重新从初始学习率开始。该机制恢复模型权重和训练进度，但不恢复 optimizer state；这是代码中明确记录的设计边界。

优势：中断后无需从 epoch 0 重新训练，能够保留已经完成的大量计算，降低长周期实验的时间和算力损失。

### 3.4 最佳 checkpoint 原子保存

原始流程主要在训练结束时保存最佳模型。如果程序提前退出，内存中的最佳权重可能丢失。本项目在验证指标改善时立即保存：

```python
best_ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
best_ckpt_tmp_path = f"{best_ckpt_path}.tmp"
torch.save(best_state_dict, best_ckpt_tmp_path)
os.replace(best_ckpt_tmp_path, best_ckpt_path)
```

先写临时文件，再通过 `os.replace` 原子替换正式文件。

优势：

- 训练异常退出时，最近一次最佳模型仍然存在；
- 写盘过程中发生异常时，不会破坏已有的最佳 checkpoint；
- 可以随时使用 `policy_best.ckpt` 进行评测。

### 3.5 NaN/Inf 快速失败与诊断

本项目在每个 batch 反向传播前检查 loss：

```python
if not torch.isfinite(loss):
    raise FloatingPointError(
        f"Non-finite loss at epoch={epoch} batch={batch_idx} rank={rank}: {details}"
    )
```

错误信息同时包含 epoch、batch、DDP rank 和各损失分量。

优势：数值异常发生后立即停止，避免 GPU 继续运行无效训练；日志能够直接定位发生异常的位置和损失来源。

### 3.6 续训统计与日志改进

原始 epoch 统计按“从 epoch 0 连续训练”计算历史切片，恢复到非零 epoch 时可能得到空切片或错误结果。本项目改为只统计当前 epoch 最新的 batch：

```python
epoch_summary = compute_dict_mean(train_history[-(batch_idx + 1):])
```

验证日志同时显示当前和历史最佳结果：

```text
Val loss: <current> @ epoch <current_epoch> |
Best val loss: <best> @ epoch <best_epoch>
```

优势：恢复训练后统计仍然正确，并且可以从日志快速判断模型是否继续改善以及应该选择哪个 checkpoint。

## 4. T1–T4 配置组织

四个任务没有复制四份训练实现，而是采用公共启动器和薄配置入口：

```text
train.sh
├── train_T1.sh → adjust_bottle，200 episodes，hidden 512
├── train_T2.sh → grab_roller，200 episodes，hidden 512
├── train_T3.sh → stack_bowls_two，400 episodes，hidden 768
└── train_T4.sh → stack_bowls_three，1021 episodes，hidden 1024，3-GPU DDP
```

这样所有训练稳定性改进只需要在 `imitate_episodes.py` 中实现一次，四个任务即可共同受益。部署端使用同样的命名规则，使训练结构与加载 checkpoint 的结构一一对应。

## 5. 三分钟视频结构

| 时间 | 画面 | 讲解重点 |
| --- | --- | --- |
| 0:00–0:20 | GitHub README 与成绩表 | 项目背景、四项任务和最终成绩 |
| 0:20–0:40 | `policy/ACT` 文件结构 | 一套共享代码、四个任务配置入口 |
| 0:40–1:10 | `imitate_episodes.py` scheduler 代码 | 固定学习率与余弦退火的对比和优势 |
| 1:10–1:40 | 验证频率与 resume 代码 | 降低验证开销，支持训练中断恢复 |
| 1:40–2:10 | 原子保存和 NaN/Inf 检查 | 保护最佳权重，快速停止无效训练 |
| 2:10–2:35 | 统计修复与日志输出 | 续训统计正确，当前/最佳结果可追踪 |
| 2:35–2:50 | 四个 `train_T*.sh` dry-run | 展示配置差异和公共训练命令 |
| 2:50–3:00 | README 总结 | 强调可恢复、可诊断和可复用 |

## 6. 建议讲稿

> 本项目基于主办方提供的 ACT 和 RoboTwin 代码，完成了 T1 到 T4 四项任务，成绩分别为 T1 59、T2 66、T3 74 和 T4 64.3。
>
> 我的主要代码改进不是修改 ACT 网络结构，而是解决长周期训练的稳定性问题。首先，我加入了余弦学习率调度，使学习率从一乘十的负五次方逐渐下降到一乘十的负六次方，前期保证学习速度，后期使用更小步长进行优化。
>
> 第二，我增加了可配置验证频率。原始代码每个 epoch 都进行完整验证，现在可以根据训练规模设置间隔，并保证最后一个 epoch 一定验证，从而减少重复验证开销。
>
> 第三，我实现了权重级断点续训，并让学习率调度与恢复 epoch 对齐。机器重启或任务异常后，不需要从头训练。
>
> 第四，当验证结果改善时，代码会通过临时文件和原子替换立即更新最佳 checkpoint。即使之后训练异常退出，也不会丢失最佳结果。同时，每个 batch 都检查 NaN 和 Inf，一旦出现数值异常，就打印 epoch、batch、rank 和各项 loss 后立即停止。
>
> 最后，我修复了从非零 epoch 续训时的损失统计，并在日志中同时记录当前验证结果、历史最佳结果和学习率。最终训练过程变得可中断、可恢复、可诊断，并且最佳结果能够持续保存。这些改动位于共享训练代码中，因此可以直接复用于 T1 到 T4。

## 7. 录屏命令

以下命令只展示代码与最终训练命令，不会启动正式训练：

```bash
# 定位核心修改（远端服务器自带 grep）
grep -nE "CosineAnnealingLR|val_freq|resume_ckpt|torch.isfinite|os.replace|Best val loss" \
  external/robotwin_local/policy/ACT/imitate_episodes.py

# 展示四项任务最终展开的训练命令
for track in T1 T2 T3 T4; do
  ACT_DRY_RUN=1 bash "external/robotwin_local/policy/ACT/train_${track}.sh"
done

# 展示相对主办方初始提交的核心训练差异
git diff 7630a1a -- \
  external/robotwin_local/policy/ACT/imitate_episodes.py \
  external/robotwin_local/policy/ACT/detr/main.py
```

## 8. 录制前检查

- 视频长度不超过 3 分钟；
- 浏览器和终端字号调大，确保代码可读；
- 关闭通知，并隐藏服务器地址、用户名、token 和其他凭据；
- 不展示 31GB checkpoint、中间数据或无关训练日志；
- 提前运行 dry-run，确认四个训练命令能够正常展开；
- 讲解重点放在“原始问题 → 代码改动 → 实际优势”，不要逐行朗读代码；
- 最后回到 README 成绩表，用一句话总结贡献。

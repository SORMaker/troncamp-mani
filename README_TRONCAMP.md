# TronCamp · ACT 四任务套餐选手包

> **归档说明：**这是主办方参赛包的原始通用说明。当前项目的最终技术方案、T1–T4
> 命名训练/部署配置和可复现命令以[仓库首页](./README.md)为准；不要用本页的通用
> `train.sh` 示例替代首页中的 `train_T1.sh` 至 `train_T4.sh`。

Tron2 双臂机器人,四个**难度递增**的操作任务(难度只来自任务本身),单张 4090 训 **ACT**
(Action Chunking Transformer,~80M)。clone 本仓 → 按文档分步装环境 → 用下发的调好专家**自采数据** →
训 ACT → 公开 seed 本地自评 → `submit.py` 提交。**只有 T4 计分**,T1/T2/T3 pass/fail 顺序解锁。

- **适用对象**:高校 / 开发者具身智能参赛队;有单张 24G 级显卡(如 4090),会用 conda 与 PyTorch。
- **当前状态**:面向 2026 年 7 月 TronCamp 比赛的公开版本。
- **不做什么**:不提供预训练权重 / 现成数据集(数据由选手用下发的调好专家自采);不做真机部署;
  官方计分口径与阈值以主办方后端为准,本包仅用于本地自评与提交。
- **许可**:LimX 自研代码采用 Apache-2.0;内嵌的第三方组件各自遵循其上游许可证(内嵌 NVIDIA
  cuRobo 为 Apache-2.0,v0.8.0);Tron2 机器人模型为 LimX 自有专有资产,仅供 TronCamp 比赛内使用
  (见 [`NOTICE`](NOTICE) item 5)。详见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。

## 任务

| 赛道 | 任务 |
|---|---|
| T1 | adjust_bottle(单臂调正瓶子) |
| T2 | grab_roller(双臂抓举滚筒) |
| T3 | stack_bowls_two(双臂叠 2 碗) |
| T4 | stack_bowls_three(双臂叠 3 碗,**主榜**) |

四题共用同一条「采集 → 训练 → 评测」流水线;难度只来自任务由易到难,没有额外的自由度 / 技能分层。

## 快速开始

> **完整分步见参赛文档**:安装、采集、数据处理、训练、评测、提交的逐步命令与说明见
> [参赛文档](https://limx-troncamp.github.io/troncamp-web-mani/doc.html)(安装分步见
> [§安装](https://limx-troncamp.github.io/troncamp-web-mani/doc.html#install)、提交见
> [§提交](https://limx-troncamp.github.io/troncamp-web-mani/doc.html#submit))。下面是命令速查。

> **环境名提示**:安装建 / 用本赛事独立命名的 conda 环境 `troncamp_env`,与 RoboTwin 官方教程默认的
> `RoboTwin` 环境天然隔离,不会复用或污染你已有的环境。想换名字,替换安装命令里的 `troncamp_env` 即可。

```bash
# 0. 装环境 —— 见参赛文档「§安装」的透明分步(建 conda 环境 + RoboTwin/SAPIEN/cuRobo + ACT 依赖,
#    含关键的 __KIT_ROOT__ 占位还原);装完 `python setup/env_check.py` 自检。

# 1. 自采数据 —— 用下发的调好专家(envs/<task>.py 的 play_once())采 RoboTwin demo,只留成功 episode。
#    T1 给了一份 turnkey 采集 config(adjust_bottle_200ep);T2–T4 照它自己写(拷贝改名、调 episode_num / 场景)。
bash collect_data.sh adjust_bottle adjust_bottle_200ep 0

# 2. 转 ACT 16-D 训练格式(collect / process_data / train 用同一 config 名串起来 → 数据键 sim-<task>-<config>-<num>)
( cd external/robotwin_local/policy/ACT && bash process_data.sh adjust_bottle adjust_bottle_200ep 200 )

# 3. 训 ACT(turnkey 超参写死在 train.sh:kl_weight 10 / chunk 50 / hidden 512 / lr 1e-5 / state_dim 16,单卡可训)
( cd external/robotwin_local/policy/ACT && bash train.sh adjust_bottle adjust_bottle_200ep 200 0 0 )
#    产出 → act_ckpt/act-adjust_bottle/adjust_bottle_200ep-200/(policy_best.ckpt / policy_last.ckpt + dataset_stats.pkl)

# 4. 公开 seed 本地自评(官方评测同一内核,同分布不同种子)
python starter/eval_local.py --track T1 \
  --ckpt-dir external/robotwin_local/policy/ACT/act_ckpt/act-adjust_bottle/adjust_bottle_200ep-200

# 5. 提交(唯一通道;T1 只交权重,T2/T3/T4 加 --code-dir external/robotwin_local。token 用 --token-file/env,不裸传)
CK=external/robotwin_local/policy/ACT/act_ckpt/act-adjust_bottle/adjust_bottle_200ep-200/policy_best.ckpt
python submit/submit.py --token-file <你的 token 文件> --track T1 --ckpt $CK
#    默认已连官方入口 https://submit.troncamp-mani.limxdynamics.com(可用 --server / env TRONCAMP_SERVER 覆盖);
#    顺序解锁、每日限额、--code-dir 用法、后端拒绝提示等详见参赛文档「§提交」
```

> **T2–T4**:把 `external/robotwin_local/task_config/adjust_bottle_200ep.yml` 拷成 `<task>_200ep.yml`,
> 自己定采集集数 / 场景,再把三步命令里的 `adjust_bottle` 换成对应任务名。评测用的 `<task>_clean.yml`
> 已随包下发、勿改。本地自评评的是你 `--ckpt-dir` 目录里的 `policy_last.ckpt`;提交时 `--ckpt` 交哪个评哪个。

## 计分

- **T4 得分 /100**:末态按栈建到第几层给分,三层各占 1/3、逐层相加、叠满 = 100(锚点 C=[0,−0.1]
  的 xy ±4cm 内)。队伍分 = 私有 100 seed 末态得分均值,主榜按此降序。
- **T1/T2/T3**:分级 SR ≥ 阈值 τ 即达标解锁(过即锁),本身不计分。阈值由主办方赛前实测标定。
- 公开 100 seed 随包下发(本地无限次自测);私有 100 seed 只在官方评测机。

## 目录

```
collect_data.sh                       # 自采:打 Tron2 自碰撞 patch 后跑 RoboTwin 采集(专家 = envs/<task>.py)
external/robotwin_local/policy/ACT/    # ACT 训练栈:process_data.sh / train.sh / imitate_episodes.py
starter/eval_local.py                  # 本地自评(包装官方评测内核 run_act_eval,--ckpt-dir 指 act_ckpt 目录)
starter/watch_rollout.py               # 单条 rollout 数值核查(末态 sr/graded)
starter/public_seeds.json              # 公开 100 seed
setup/env_check.py                     # 环境自检(装完验证:curobo 0.8.0 / sapien / mplib / ffmpeg)
submit/submit.py                       # 唯一提交通道(--ckpt 单文件 + 可选 --code-dir,带 token)
recipes/eval/                          # 官方评测内核(选手本地与官方同一套)
external/robotwin_local/               # RoboTwin 2.0 + Tron2 embodiment + 四任务调好专家 + eval 配置
embodiments/                           # Tron2 机器人网格 / 标定
```

## 许可 / License

- LimX 自研文件(`recipes/`、`starter/`、`setup/`、`submit/`、Tron2 接入与任务配置)采用
  **Apache License 2.0**(见 [`LICENSE`](LICENSE))。
- 本包再分发的第三方组件各自遵循其上游许可证,清单见 [`NOTICE`](NOTICE)。内嵌的 NVIDIA
  **cuRobo** 为 **Apache-2.0**(v0.8.0);Tron2 机器人模型(URDF / MJCF / mesh)为 LimX
  自有专有资产,仅供 TronCamp 比赛内使用(见 [`NOTICE`](NOTICE) item 5)。
- 参与 / 反馈见 [`CONTRIBUTING.md`](CONTRIBUTING.md);安全问题见 [`SECURITY.md`](SECURITY.md)。

平台基于 [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin);机器人 Tron2 由
[LimX Dynamics](https://github.com/limxdynamics) 提供。

---

## English summary

**TronCamp · ACT Four-Task Suite — contestant kit.** Train **ACT** (Action Chunking
Transformer, ~80M) on a single 24 GB GPU (e.g. RTX 4090) to solve four increasingly
hard Tron2 dual-arm manipulation tasks: **T1** `adjust_bottle` → **T2** `grab_roller`
→ **T3** `stack_bowls_two` → **T4** `stack_bowls_three` (**main leaderboard**).
Difficulty comes only from the task; all four share one *collect → train → evaluate*
pipeline, unlock sequentially, and **only T4 is scored** (graded final-state /100).

- **For:** student / developer embodied-AI teams comfortable with conda + PyTorch.
- **Status:** public release for the July 2026 TronCamp event.
- **Does NOT:** ship pretrained weights or ready-made datasets (you self-collect data
  with the provided tuned experts); no real-robot deployment; official scoring lives on
  the organizers' backend — this kit is for local self-eval and submission only.
- **Quick start:** follow the numbered `bash` commands in the *快速开始* section above
  (they are language-agnostic). Submit only via `submit/submit.py`.
- **License:** LimX-authored files are Apache-2.0 ([`LICENSE`](LICENSE)); bundled
  third-party components keep their own licenses ([`NOTICE`](NOTICE)) — the bundled
  NVIDIA cuRobo is Apache-2.0 (v0.8.0). The proprietary Tron2 robot *model* is
  © LimX Dynamics, provided for use within the TronCamp competition
  (see [`NOTICE`](NOTICE) item 5).

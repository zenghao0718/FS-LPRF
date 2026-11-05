# 训练验证调度重构 v1.0

本轮依据 `FS-LPRF_Codex_Training_Validation_Refactor_v1.0.md` 修改训练调度、报告和恢复。XLS-R 冻结、固定来源的前端实现、25 组表示、Attention/Bottleneck、Episode、原型/α/Loss、AdamW 与训练预算均保持原有设置。不执行真实训练、Dev/Test 或原型库构建。2026-11-03 已将必要前端与音频代码整合进项目并移除 Submodule，本文描述的调度及 Checkpoint 协议不变，验证记录见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

## 调度与配置

默认 4000 Steps：轻量节点为 200/400/600、1000/1200/1400、1800/2000/2200、2600/2800/3000、3400/3600/3800，共 15 次；完整节点为 800/1600/2400/3200/4000，共 5 次。完整节点不另做轻量验证。

```yaml
train:
  light_validation_interval_steps: 200
  full_validation_interval_steps: 800
validation:
  light_seed: 42
  light_max_query_per_class: 1000
```

两周期必须是正整数，full 必须为 light 的整数倍，4000 必须被 full 整除。轻量 seed 固定为 42、配额允许 1..1000。通过原有 `--set` 机制覆盖。旧 `train.validation_interval_steps` 会明确报迁移错误。当前协议固定每 Step 8 Episode，每 Episode 15 Query、每类 5 Support；音频微批次可按设备容量配置。

## 固定轻量子集

完整 `evaluation/dev_episode_manifest.json` 的格式与抽取规则不变。轻量清单保存为 `evaluation/dev_light_manifest.json`，直接复用 `supports["42"]`，不抽新 Support。

每个类别的候选只来自完整清单的 `queries[class]`，因此已经排除原五种子全部 Support。候选不足配额时全部保留，1500 条时取 1000、80 条时保留 80；所有 32 Fake 类都参加，Real Query 在任务之间共用。每类按稳定排序、源清单 SHA 和类别名派生局部 RNG，不使用 Python `hash()`、不消耗训练全局 RNG，也不改变官方音频裁剪函数。

清单包含来源 SHA、数据指纹、seed、shot、配额和实际 ID；加载时重新核对确定性结果与不重叠关系。来源文件（包括文件字节）、数据、配置或 ID 被改动时拒绝复用，不静默重抽。轻量结果只监控，不能进入 Best。

## 报告与实时记录

每次验证保留原有 no-grad 评分路线、RNG 保存/恢复和模型训练模式恢复。实际 Query 数不等长时仍逐类等权 Macro，ACC 继续固定零阈值；CSV 为比例、日志为百分比，所有 Std 均用 ddof=0。

缓存管线默认从独立 Light/Full、各 seed 的 XLS-R 25 层缓存继续执行 Attention/Bottleneck，Support/Query 规则与 Real 复用关系不变。训练前显式构建并校验缓存；Test 仅由最终 Best 后的评估入口处理。新的 Run Identity 绑定实际缓存内容、schema 与逐样本 RNG 版本，排除已验证不改变输入的 Worker/预取/pin memory 等性能项。新管线必须使用新输出目录，不能混用旧 Run 的中间状态。详见 [DATA_CACHE.md](DATA_CACHE.md)。

完整 step 800 的独立报告位于 `evaluation/full/`：

- `dev_full_per_class_seed_step0800.csv`：160 行，含 Step/seed/类别/实际 Query 数和 EER/threshold/AUC/AP/ACC。
- `dev_full_per_class_mean_std_step0800.csv`：32 行，含四指标五种子 Mean/Std 和每类 Query 数范围。
- `dev_full_macro_per_seed_step0800.csv`：5 行，各种子先对 32 类等权 Macro。
- `dev_full_macro_mean_std_step0800.csv`：4 行，五种子 Macro 的 Mean/Std。
- `dev_full_predictions_step0800.csv`：全部逐 Query 预测，含 ID、Fake 类、seed、标签、连续分数、预测。

其他里程碑各自保留同样文件。轻量报告使用 `evaluation/light/dev_light_*_step0200.csv` 等命名，单种子原始明细 32 行；后续轻量节点不会覆盖前一节点，更不会覆盖 full 或原 Test 报告。恢复后重做同一未完成验证时允许原子替换该次半成品报告。

终端与 `train.log` 每 Step 输出 Loss/LR/Gradient Norm，标准 Logging Handler 每次 emit 刷新。每轮 full 打印全部 32 类四指标 Mean±Std、实际数量、Macro，以及当时 Best Step/EER；light 也打印实际数量和各类监控值。两模式都记录开始、结束、实际耗时和报告路径。

TensorBoard 每步保留 `train/loss`、`train/learning_rate`、`train/gradient_norm`。轻量使用 `dev/light_macro_{eer,auc,ap,acc}`；完整使用 `dev/full_macro_{eer,auc,ap,acc}_{mean,std}`。事件后台每 10 秒刷新、验证后主动 flush。恢复时隐藏恢复节点之后的旧训练事件，并从完成标记补录当前节点验证摘要，以覆盖“报告已提交但事件未刷盘”的中断窗口，不重复推理。

```bash
tail -f /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf/train.log
tensorboard --logdir /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf/tensorboard --host 0.0.0.0 --port 6006
```

## Checkpoint、Best 与中断恢复

每个里程碑先原子发布 `checkpoints/step_0800.pt` 等独立节点，再执行完整 Dev。默认只生成 5 个 `.pt`，不再写逐步 `last.pt` 或复制 `best.pt`。节点保留后端、Optimizer、Scheduler、RNG、完成 Step、配置、完整 Dev SHA、训练数据指纹及绑定完整/轻量配置的 run identity；冻结 XLS-R 不进入节点，继续 `weights_only=True` 加载。

完整报告先在临时目录写完并发布，全部成功后才原子写 `evaluation/full/dev_full_complete_step0800.json`，绑定该节点和五份报告的 SHA-256。完成标记是历史与 Best 的唯一依据；节点中旧的 `best_eer`/`dev_metrics` 仅描述存盘前状态，不用于恢复选模。

仅五种子完整 Dev 的 Macro EER 均值选 Best，按 `(mean_eer, step)` 最小值决定，平局取较早节点。`checkpoints/best_checkpoint.json` 包含相对节点名、Step、EER、节点 SHA 和 run/manifest/data 身份；解析器拒绝路径逃逸、节点名/Step 不一致、哈希或身份不匹配。

```bash
python scripts/train.py --config configs/speechfake.autodl.yaml --resume /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf/checkpoints/step_0800.pt
# 全部训练和Dev选模完成后才执行Test：
python scripts/evaluate.py --config configs/speechfake.autodl.yaml --checkpoint best --shot 10
```

恢复行为：

- 当前节点没有完成标记，或其报告缺失/损坏：先补做该节点 full，再继续下一 Step；若已是 4000，只补验证，不增加训练步数。
- 标记已完成、Best 索引未写完：从不晚于恢复节点的完整历史重建索引，验证不重跑。无有效历史的旧索引会保留原内容并标为 pending，解析 Best 会拒绝，直至补做 full 成功。
- 目录中有晚于恢复节点的 CKPT/完成标记：明确拒绝倒退，避免覆盖已有节点或混入未来历史。请恢复最新节点。较早节点历史缺失/损坏时，保留所有文件并恢复匹配的完整输出历史。
- 配置、音频微批次、worker 数、数据或轻量/完整清单发生变化：拒绝混用同一 run；CUDA ordinal 可改变。新训练目录只允许预先生成的 Manifest，不能覆盖已存在的训练产物。
- 中断最多丢失上一里程碑后尚未存盘的 799 Steps；旧日志保留，续跑 Episode 写入新的 resume 段。

本轮不做旧训练目录/旧调度 Checkpoint 的自动迁移；保留旧目录，为新调度选择新输出目录。旧底层 Checkpoint 加载接口仍保留用于历史 Smoke 与接口回归。

## 本地验收

2026-11-02 在项目 `.venv`（PyTorch 2.7.1+cu128）执行 `python -m pytest -q --junitxml=outputs/refactor_validation/pytest.xml`，结果 **89 Passed、0 Failed、0 Skipped，38.36 秒**；`python -m pip check` 无依赖冲突。72 条告警来自既有官方 RawBoost/librosa 弃用接口。覆盖默认 15/5/5 调度、轻量配额/来源/RNG、160/32 行报告、全部类别日志、Best 平局、节点先写、缺失验证补做、完成标记/报告绑定、未来节点拒绝、TensorBoard tags、实际 Logger 双输出、报告与历史衔接和原 Test CLI 回归。机器可读测试报告留在被 Git 忽略的输出目录。

本轮修改未触及模型/音频/损失，上轮真实 XLS-R Smoke 记录保留在 `docs/LOCAL_VALIDATION.md`；本轮不重复真实前端 Smoke，不下载权重，不执行正式训练或真实 Dev/Test。正式 AutoDL 步骤见 [AUTODL.md](AUTODL.md)。

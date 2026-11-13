# FS-LPRF-LW v1

本分支唯一研究变量：Support 类内/类间统计可靠性 → 25 个独立、跨 Episode 共享的全局参数。

`FSLPRF.layer_logits = nn.Parameter(torch.zeros(25))`，`model.layer_weights()` 经唯一的标准 Softmax 实现返回 α。初始 α 均为 0.04，无温度、截断、权重约束或额外正则。25 个参数由原 AdamW 参数收集逻辑自动纳入；可训练参数总数由 364800 增至 364825。原 AdamW weight decay 和全模型梯度范数裁剪协议保持不变。

训练每层对全部 Query 求平均交叉熵，再按 α 加权；8 个 Episode 共用同一个 α 张量，平均后只执行一次 backward/update。α 不 detach，Support 原型与 Query 的原梯度路径保留。K-shot 的 Support 只构建 Real/Fake 原型，评分仍为 `sum(alpha * (S_fake - S_real))`，阈值 0。SpeechFake Light/Full Dev、Test 与外部 ASVspoof5 均经同一个 `build_support_tasks` 和评分入口。

已删除 `methods/reliability.py` 及类内离散度计算。0-shot Bank 仅作必要适配：保存对应 LW Checkpoint 的 Logits 快照，通过同一 Softmax 获取 α，拒绝旧版 Bank；不扩展 0-shot 实验。

Checkpoint 使用 `method_version=FS-LPRF-LW-v1`；保存、加载与 Best 解析均检查 25 个有限 Logits，模型键必须严格匹配。Run Identity 也包含方法身份。原节点不可覆盖、原子发布、Best 哈希/历史绑定、Optimizer/Scheduler/RNG 恢复、崩溃后补做 Full Dev 等机制保留。旧方法 Checkpoint 不可作为 LW 初始化或续训文件。

TensorBoard 按原训练日志节奏（每 Step）记录 `layer_weight/layer_00`…`layer_weight/layer_24`、`layer_weight/max`、`layer_weight/min` 和 `layer_weight/entropy`。熵仅供观察；用 log_softmax 计算以处理浮点下溢，不改变训练权重。

## AutoDL 正式运行（本次未执行）

先在 AutoDL 切换到 `feature/learnable-layer-weights`。默认 YAML 已启用 Offline Codec，但不能仅凭默认配置认定与上一轮完全一致。本地未找到上一轮 `config_resolved.yaml`，因此尚未完成真实历史配置的逐项核对。将以下 `PREVIOUS_RUN` 改成已完成的 Codec Offline v1 实验目录，用其 `config_resolved.yaml` 精确复用数据路径、Waveform/Codec 缓存、增强、微批次、Worker、随机种子和验证设置。不得覆盖旧输出或加载其 Checkpoint。

```bash
export PREVIOUS_RUN=/实际路径/上一轮_Codec_Offline_v1_实验目录
export LW_CONFIG=configs/speechfake_lw_v1.autodl.yaml
python - <<'PY'
import copy
import os
from pathlib import Path
from utils.config import load_config, save_config, validate_config

source = Path(os.environ['PREVIOUS_RUN']) / 'config_resolved.yaml'
cfg = load_config(source)
previous = copy.deepcopy(cfg)
codec = cfg['data']['codec_augment']
assert codec['enabled'] is True and codec['mode'] == 'offline', '必须使用上一轮 Offline Codec 配置'
assert cfg['data']['train_cache_mode'] == 'waveform'
out = Path('/root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf_lw_v1')
assert out.resolve() != source.parent.resolve()
assert not out.exists() or not any(out.iterdir()), '输出目录必须全新且独立'
cfg['runtime']['output_dir'] = str(out)
validate_config(cfg)
check = copy.deepcopy(cfg)
check['runtime']['output_dir'] = previous['runtime']['output_dir']
assert check == previous, '除输出目录外不得改变上一轮配置'
target = Path(os.environ['LW_CONFIG'])
assert not target.exists(), '保留已生成的正式配置，勿意外覆盖'
save_config(cfg, target)
reloaded = load_config(target)
reloaded['runtime']['output_dir'] = previous['runtime']['output_dir']
assert reloaded == previous, '保存并重新加载后必须仍然只改变输出目录'
print(target, '->', out)
PY

# 检查生成的 YAML、上一轮 cache_bindings.json 和数据/缓存可用性后，由用户启动：
python scripts/train.py --config "$LW_CONFIG"
```

不重新构建或修改任何缓存。上一轮若是单种子开发验证，其配置也会被原样保留；不要在这个对照实验中擅自改成五种子，正式对照采用的协议必须与上一轮一致。原 Light/Full 周期和 Test 五种子协议均由已解析配置及现有校验器约束。新运行生成独立的 manifests、cache_bindings 和完整日志；可核对上一轮相同数据/种子产生的 manifest 内容与缓存身份。

从**新 LW 运行的最新已保存节点**恢复，保留整个新输出目录：

```bash
python scripts/train.py --config "$LW_CONFIG" \
  --resume /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf_lw_v1/checkpoints/step_0800.pt
```

`step_0800.pt` 仅为示例，请使用实际最新节点；不能回退覆盖后续节点，不能用上一轮手工权重模型继续训练。训练完成后，SpeechFake Test 使用新运行 Best：

```bash
python scripts/evaluate.py --config "$LW_CONFIG" --checkpoint best --shot 10
```

ASVspoof5 继续使用上一轮相同的 `scripts/evaluate_external.py` 命令、输入 mapping、Support 10-shot、种子与分组协议，仅将 `--config` 指向 LW 配置、`--checkpoint` 指向新 Best 索引选出的实际节点文件、`--output-dir` 改为独立 LW 外部评估目录。外部入口要求显式节点路径，不接受隐式 `best`。本任务不运行这些正式评估。

## 本机轻量验证

```bash
python -m pytest -q tests/test_methods.py tests/test_layer_weights.py \
  tests/test_engine.py tests/test_checkpoint_refactor.py \
  tests/test_integrated_components.py tests/test_cached_features.py \
  tests/test_evaluation.py tests/test_external_evaluation.py \
  tests/test_validation_schedule.py tests/test_validation_reports.py \
  tests/test_development_validation.py tests/test_eval_loader_reuse.py \
  tests/test_cache_integration.py
```

均为 CPU 单元测试、合成特征/临时合成音频或 mock；不加载 XLS-R-300M 权重，不执行真实数据训练、完整验证或全量 ASVspoof5 评估。

2026-11-13 本机执行上述验收集合：**195 passed，2 warnings，81.15 秒**。警告为现有 PyTorch weight_norm 与 librosa importlib-resources 弃用提示。覆盖生产构造的全零初始化、Softmax/梯度解析定义、8 Episode 共享权重、Support 原型梯度、AdamW 更新、非均匀权重推理、Checkpoint/Best/历史保护和恢复后下一 Step 与连续运行逐参数一致。

# FS-LPRF · SpeechFake-BD Phase 1

依据 `FS-LPRF_Method_Design_v2.4.md` 与 `FS-LPRF_Phase1_Codex_Implementation_Spec_v1.3.md` 实施。第一阶段仅包含 SpeechFake-BD：3-way episodic training、五种子 Dev 选模与 Test 10-shot。正式训练、Dev/Test 和真实原型库构建由用户在 AutoDL 执行；本地仅做合成数据单元测试与真实 XLS-R 小批量 Smoke Test。

## 官方代码复用

Layer-Wise：[XIAOYixuan/tomatoDD](https://github.com/XIAOYixuan/tomatoDD)，分支 `interspeech25-layer-wise`，固定 commit `88efe1376dfc5f5b9b62d016d33a7381c158d8af`，以 `third_party/tomatoDD` Submodule 引用。上游文件保持原样。

- `GeneralFAD` / `FADTrim` / `audio_util`：读取、16 kHz 重采样、第一声道、首尾静音裁剪、−20 dB RMS、重复补足、随机 4 秒裁剪。
- `RawBoostAudioAugmentation`：RIR 1/3、Algorithm 5 为 1/5，独立抽样，同时触发时先 RIR 后 RawBoost。项目子类只覆盖 RIR 资源路径。
- `XLSRAdapterBase.extract_features()` 与 `XLSRTimeAttnOnly`：完整 25 组 Hidden States、逐层独立 4-head 时间 Attention。
- `ClassificationBase.state_dict()`：保存时排除冻结前端；EER 使用官方 `compute_eer`。

上游仓库未声明整体许可证；保留来源和 Submodule 引用，不将其源码复制后作为本项目代码再分发。

## 环境

先检查 `conda env list`。优先复用已有兼容环境，不向 `base` 安装项目依赖。本地实际使用 Miniforge Python 3.11 创建项目内独立 venv；也可创建 Conda 环境：

```bash
conda create -n fs-lprf python=3.11 -y
conda activate fs-lprf
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

本机 PowerShell 可直接使用项目环境，无需修改系统 Python：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe scripts/check_model.py --config configs/speechfake.yaml
```

`requirements.txt` 不包含 torch/torchaudio，避免覆盖已匹配 CUDA 的版本。AutoDL 环境选择和全部命令见 [docs/AUTODL.md](docs/AUTODL.md)。

## XLS-R 权重

仅使用 [官方 XLS-R-300M Fairseq checkpoint](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt)：

```bash
python scripts/download_frontend.py
```

默认写入 `pretrained/xlsr2_300m.pt`，同时记录来源、文件大小和 SHA-256。脚本使用官方 HTTP Range 分段续传；重试时保持相同 `--connections`。合并前检查 SHA-256，保留已下载分片；文件已完整且校验正确时直接复用，不再发起网络下载。该官方完整训练快照实际为 **3,808,868,242 字节**，不是文档估计的约 1.2GB。数据、权重、缓存和输出均被 Git 忽略。

原始文件保存 `cfg.model` 和 `model`；上游 `tools/fairseq_model_converter.py` 期望先用 Fairseq 解析并导出 `cfg: Namespace`、`state_dict`。项目在上游 `w2v2` 模块的读取边界完成等价的内存封装适配，保留原始 `.pt` 与全部张量，不安装旧版 Fairseq、不改特征提取代码。只补齐 Fairseq 自身的历史默认配置字段，依据 [Fairseq v0.12.2 的 Wav2Vec2Config](https://github.com/facebookresearch/fairseq/blob/v0.12.2/fairseq/models/wav2vec/wav2vec2.py)。加载首先尝试默认 `torch.load(..., map_location="cpu")`；仅当 weights-only 兼容错误且文件 SHA-256 与已验证官方文件完全一致时，才允许 `weights_only=False`。不会全局替换 `torch.load`。

## Manifest 和路径配置

数据准备、去重和清单生成由用户在 AutoDL 完成。`data.data_path` 目录应包含 `train/dev/test` 各一份 `.tsv` 与 `.txt`，全部无表头、Tab 分隔。

`*.tsv` 严格两列：

```text
uttid<TAB>audio_absolute_path
```

`*.txt` 严格五列：

```text
uttid<TAB>origin_ds<TAB>speaker<TAB>attacker<TAB>label
```

Real 使用 `attacker=real`、`label=bonafide`；Fake 使用 `label=spoof`，`attacker` 保留任务类型与模型的组合，例如 `TTS__ModelA`、`VC__ModelA`。各 split 必须包含相同的 32 个 Fake 类别和一个统一 Real 类；代码不合并同名模型。`origin_ds` 原样保留；将来构建原型库时 Real 来源应使用 `AISHELL-1`、`AISHELL-3`、`CommonVoice`、`LibriTTS`、`VCTK`。

编辑 `configs/speechfake.yaml` 的路径，或为每条命令使用 `--set section.key=value`：

```bash
python scripts/check_data.py --config configs/speechfake.yaml \
  --set data.data_path=/root/autodl-tmp/speechfake/manifests \
  --set data.rir_root=/root/autodl-tmp/resources/RIRS_NOISES \
  --set model.frontend_path=/root/autodl-tmp/pretrained/xlsr2_300m.pt
```

也支持 `FS_LPRF_RIR_ROOT` 环境变量。RIR 使用 OpenSLR 28 原始资源。读取异常会报告具体 ID/路径，不随机换成其他类别的样本。

## 模型与训练协议

XLS-R 参数冻结、无梯度，调用 `model.train()` 后仍保持前端 eval。25 组 Hidden States（Transformer 前表示 + 24 层输出）经过官方逐层 Attention，输出 `[N,25,1024]`，再用一个共享 `Linear(1024,256)` 得到 `[N,25,256]`。不包含重建头、传统分类头、温度或额外归一化。

每 Episode 独立均匀抽取 Real + 两个不同 Fake，每类 5 Support；默认每 Episode 15 Query、每 Step 8 Episode、音频微批次 8。每个 Episode 独立计算均值原型、余弦分类、Support 可靠性及逐层 CE。类间距离只累加 Real–Fake；类内离散为各类均值之和；可靠性为 `(inter+1e-8)/(intra+1e-8)`，开平方归一化后只 detach α。Support 原型仍保留梯度。一个 Step 的 Episode Loss 取平均后只 backward 一次、AdamW 更新一次。

固定训练种子 42、4000 Steps、AdamW LR 1e-4、betas (0.9,0.999)、weight decay 1e-4、warmup 100 Steps、cosine 降至 1e-6、梯度裁剪 1.0。配置校验阻止修改固定模型与协议；Query 数、Episode 数、音频微批次和验证间隔按文档参数化。

## Dev / Test 与恢复

Dev/Test 都是 10-shot、种子 `[17,42,71,100,123]`。每个种子的 Real Support 在 32 个任务间共用。预先保存各种子 Support；Query 为各类全量样本排除五组 Support 并集后的剩余 ID。Manifest 只创建一次，复用时核对元数据指纹和划分完整性。

所有 split 均保留官方随机裁剪。每次评估按种子重置 Python/NumPy/PyTorch CPU/CUDA RNG，固定数据顺序和 DataLoader generator；workers 由 `torch.initial_seed()` 初始化，`persistent_workers=False`。同一 Dev 种子跨 Checkpoint 的裁剪序列一致，不同种子允许不同。结束后恢复训练 RNG 与训练模式。

默认每 200 Steps 及最后一步验证，按五种子 Dev 类别宏平均 EER 的均值选择 `best.pt`；每步保存 `last.pt`。恢复同时加载模型后端、Optimizer、Scheduler、训练 RNG，核对训练数据指纹与 Dev Manifest SHA-256：

```bash
python scripts/train.py --config configs/speechfake.yaml
python scripts/train.py --config configs/speechfake.yaml --resume outputs/speechfake_fs_lprf/checkpoints/last.pt
python scripts/evaluate.py --config configs/speechfake.yaml \
  --checkpoint outputs/speechfake_fs_lprf/checkpoints/best.pt --shot 10
```

正式命令只在 AutoDL 执行。输出包含解析配置、日志、TensorBoard、Episode 类别和 Support/Query ID、Dev/Test Manifest、各类指标、各种子宏平均、五种子 Mean/Std 与逐 Query 的 ID/类别/种子/标签/连续分数/预测。指标以 Fake=1、分数为 Fake-minus-Real；ACC 使用 `score>=0`，EER 阈值扫描，AUC/AP 使用连续分数，先逐类计算再宏平均。Std 使用 `numpy.std(ddof=0)`。

## 轻量测试

```bash
python -m pytest -q
python scripts/check_model.py --config configs/speechfake.yaml
```

Pytest 使用合成向量和临时 WAV，包含梯度、损失聚合、一次更新、音频处理、五种子固定划分、0/2 workers 的裁剪复现、RNG 恢复、合成 33 类原型库与 Checkpoint 往返测试。`check_model.py` 使用正式模型、真实权重和一条合成 4 秒 WAV，核对 25 组 Hidden States、池化/投影形状、前端冻结、所有层 Attention 与共享 Bottleneck 梯度，以及官方后端 Checkpoint 保存逻辑；不执行优化器更新。报告写入 `outputs/local_smoke/model_report.json`。本地实测记录见 [docs/LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md)。

## 0-shot 接口（本阶段仅合成测试）

`methods.inference` 提供参数化 K-shot、33 类 PrototypeBank 与 0-shot 分数；0-shot 在每层取 32 个 Fake 原型的最大相似度，再减 Real 并使用统计 α 融合。`scripts/build_prototype_bank.py` 已实现训练集无增强构建，每类不放回 1024 条，Real 配额为 205/205/205/205/204，容量不足明确报错。它记录最终 Checkpoint、样本 ID、数据指纹和版本。本阶段不运行真实构建或 0-shot 实验，不包含其他数据集、基线或消融。

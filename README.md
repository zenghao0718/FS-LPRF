# FS-LPRF · SpeechFake-BD Phase 1

依据 `FS-LPRF_Method_Design_v2.4.md` 与 `FS-LPRF_Phase1_Codex_Implementation_Spec_v1.3.md` 实施。第一阶段仅包含 SpeechFake-BD：3-way episodic training、五种子 Dev 选模与 Test 10-shot。正式训练、Dev/Test 和真实原型库构建由用户在 AutoDL 执行；本地仅做合成数据单元测试与真实 XLS-R 小批量 Smoke Test。

## 项目布局与运行入口

核心模块采用平级科研代码布局：`data/`、`models/`、`methods/`、`engine/`、`utils/` 直接位于项目根目录，与 `configs/`、`scripts/`、`tests/` 和 `docs/` 平级。配置加载与来源信息分别位于 `utils/config.py`、`utils/provenance.py`。完整迁移前后目录树及逐文件清单见 [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)。

以下命令均从克隆后的项目根目录执行。安装 `requirements.txt` 中的依赖后即可直接运行脚本，不需要可编辑安装本项目或手动设置 Python 路径；脚本会自动定位项目根目录。`data/` 是源码目录，数据资源仍通过 `data.data_path` 和 `data.rir_root` 指定；其中的本地非 Python 数据及子目录继续被 Git 忽略。

## 整合的官方代码与独立源码

项目已将实际使用的 Layer-Wise 代码整合到自身功能模块，**普通 `git clone` 即可获得全部项目源码，无需 Git Submodule 或外部 `tomato` 包**。代码来源为 [XIAOYixuan/tomatoDD](https://github.com/XIAOYixuan/tomatoDD) 的 `interspeech25-layer-wise` 分支、固定 commit `88efe1376dfc5f5b9b62d016d33a7381c158d8af`；模型与实验协议保持不变。

- `data/audio.py`：保留 GeneralFAD/FADTrim 实际使用的读取、16 kHz 重采样、第一声道、首尾静音裁剪、−20 dB RMS、重复补足和随机 4 秒裁剪。
- `data/augmentation.py`：RIR 1/3、RawBoost Algorithm 5 为 1/5，独立抽样，同时触发时先 RIR 后 RawBoost；RIR 候选严格来自 OpenSLR 28 的四份 `rir_list` 文本索引。`noise_prob=1/5` 表示 RawBoost Algorithm 5 的触发概率，不使用 OpenSLR 28 的普通环境噪声文件。
- `models/xlsr.py`、`xlsr_layers.py`、`xlsr_attention.py`：官方 Fairseq XLS-R-300M 的卷积、Transformer、自注意力及必要基础组件，直接加载原始官方权重并保留全部参数名称。未整合 Whisper、HuBERT、AASIST、FACodec 或其他无关模型。
- `models/fs_lprf_model.py`：完整 25 组 Hidden States、逐层独立 4-head 时间 Attention 与共享 `Linear(1024,256)`；仍只有 364800 个可训练参数，训练 Checkpoint 仍排除冻结前端。
- `utils/metrics.py`：整合官方 DET/EER，保留稳定排序、阈值和数值约定。

来源、作者、逐文件映射和已有许可声明见 [docs/CODE_PROVENANCE.md](docs/CODE_PROVENANCE.md)。`utils/provenance.py` 仅保存来源 commit 与官方权重信息；既有 Checkpoint/原型库中的 `upstream_commit` 字段及其值保持不变，兼容原有恢复与 Best 索引。

## 环境

优先使用项目已经验证的 **Python 3.11**。`requirements.txt` 固定 `torch==2.7.1` 和 `torchaudio==2.7.1`，其他依赖保持原有版本。CUDA Wheel 由实际 GPU 驱动、GPU 架构及部署环境决定，不在依赖文件中固定 `+cu126` 或 `+cu128`，也不需要额外安装 `torchvision`。**修改 `requirements.txt` 不等于强制所有 GPU 使用相同的 CUDA 构建**；已安装且兼容的 `2.7.1+cu126`、`2.7.1+cu128` 等构建均满足 `==2.7.1` 约束。

AutoDL 是专门用于训练的实例，默认直接使用预装的 `base` 环境，不要求新建 Conda 或 venv。部署前必须按 [docs/AUTODL.md](docs/AUTODL.md) 检查预装 Python、torch、torchaudio 和 CUDA/驱动兼容性。已有环境符合要求时直接复用，不重复下载 PyTorch；只有缺失、版本或 CUDA 构建不兼容时，才按该文档选择官方 CUDA Wheel 安装或替换对应组件。

确认 PyTorch/torchaudio 符合要求后，使用国内镜像补齐其他依赖，不添加 `--upgrade` 或 `--force-reinstall`：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
python -m pytest -q
```

本机继续使用已有的项目 `.venv`，无需创建新环境或修改系统 Python。当前 Python 3.11.18、torch/torchaudio `2.7.1+cu128`、CUDA runtime 12.8 已满足要求，保留现有构建，不重复下载安装。PowerShell 验证命令：

```powershell
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pytest -q
```

历史本地验收记录见 [docs/LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md)；AutoDL 环境检查、条件安装和正式运行命令见 [docs/AUTODL.md](docs/AUTODL.md)。

## XLS-R 权重

仅使用 [官方 XLS-R-300M Fairseq checkpoint](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt)：

```bash
python scripts/download_frontend.py
```

默认写入 `pretrained/xlsr2_300m.pt`，同时记录来源、文件大小和 SHA-256。脚本使用官方 HTTP Range 分段续传；重试时保持相同 `--connections`。合并前检查 SHA-256，保留已下载分片；文件已完整且校验正确时直接复用，不再发起网络下载。该官方完整训练快照实际为 **3,808,868,242 字节**，不是文档估计的约 1.2GB。数据、权重、缓存和输出均被 Git 忽略。

原始文件保存 `cfg.model` 和 `model`；`models/frontend_compat.py` 直接在加载边界完成等价的内存适配，供整合后的 XLS-R 读取，保留原始 `.pt` 与全部张量，不重新转换权重、不安装旧版 Fairseq。只补齐 Fairseq 自身的历史默认配置字段，依据 [Fairseq v0.12.2 的 Wav2Vec2Config](https://github.com/facebookresearch/fairseq/blob/v0.12.2/fairseq/models/wav2vec/wav2vec2.py)。加载前始终校验官方 SHA-256，再首先尝试默认 `torch.load(..., map_location="cpu")`；仅当 weights-only 兼容错误且哈希已验证时，才允许 `weights_only=False`。不会全局替换 `torch.load`。

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

Real 使用 `attacker=real`、`label=bonafide`；Fake 使用 `label=spoof`，`attacker` 保留任务类型与模型的组合，例如 `TTS__ModelA`、`VC__ModelA`。各 split 必须包含相同的 32 个 Fake 类别和一个统一 Real 类；代码不合并同名模型。`origin_ds` 原样保留；当前 BD baseline 的 Real 来源为 `AISHELL-1`、`AISHELL-3`、`LibriTTS`、`VCTK`，四个来源共同组成一个 Real 类，CommonVoice 不属于本次主实验。

编辑 `configs/speechfake.yaml` 的路径，或为每条命令使用 `--set section.key=value`：

```bash
python scripts/check_data.py --config configs/speechfake.yaml \
  --set data.data_path=/root/autodl-tmp/speechfake/manifests \
  --set data.rir_root=/root/autodl-tmp/resources/RIRS_NOISES \
  --set model.frontend_path=/root/autodl-tmp/pretrained/xlsr2_300m.pt
```

也支持 `FS_LPRF_RIR_ROOT` 环境变量。`data.rir_root` 指向 OpenSLR 28 的 `RIRS_NOISES` 根目录，例如 `/root/autodl-tmp/resources/RIRS_NOISES`；程序读取 `simulated_rirs/{smallroom,mediumroom,largeroom}/rir_list` 和 `real_rirs_isotropic_noises/rir_list` 共四份**文本索引文件**，仅使用其中列出的 WAV。支持 Kaldi `--rir-id`/`--room-id` 等字段、以 `RIRS_NOISES/` 开头的路径、资源根目录或索引所在目录的相对路径及绝对路径。解析后检查文件存在、WAV 扩展名及根目录边界，再去重排序；缺失/空索引或坏记录会明确报错，并在记录错误中指出索引及行号。未入索引的普通噪声不会纳入候选，禁止回退到递归扫描全部 WAV。完整原始资源预期为 60,000 个模拟 RIR 和 325 个真实 RIR，程序不写死候选数量；详情见 [docs/AUTODL.md](docs/AUTODL.md)。读取异常会报告具体 ID/路径，不随机换成其他类别的样本。

## 模型与训练协议

XLS-R 参数冻结、无梯度，调用 `model.train()` 后仍保持前端 eval。25 组 Hidden States（Transformer 前表示 + 24 层输出）经过官方逐层 Attention，输出 `[N,25,1024]`，再用一个共享 `Linear(1024,256)` 得到 `[N,25,256]`。不包含重建头、传统分类头、温度或额外归一化。

每 Episode 独立均匀抽取 Real + 两个不同 Fake，每类 5 Support；默认每 Episode 15 Query、每 Step 8 Episode、音频微批次 8。每个 Episode 独立计算均值原型、余弦分类、Support 可靠性及逐层 CE。类间距离只累加 Real–Fake；类内离散为各类均值之和；可靠性为 `(inter+1e-8)/(intra+1e-8)`，开平方归一化后只 detach α。Support 原型仍保留梯度。一个 Step 的 Episode Loss 取平均后只 backward 一次、AdamW 更新一次。

固定训练种子 42、4000 Steps、AdamW LR 1e-4、betas (0.9,0.999)、weight decay 1e-4、warmup 100 Steps、cosine 降至 1e-6、梯度裁剪 1.0。本轮依据训练验证重构 v1.0 固定每 Step 8 Episode、每 Episode 15 Query；音频微批次及两档验证周期通过配置管理，模型与训练算法不变。

## Dev / Test 与恢复

Dev/Test 都是 10-shot、种子 `[17,42,71,100,123]`。每个种子的 Real Support 在 32 个任务间共用。预先保存各种子 Support；Query 为各类全量样本排除五组 Support 并集后的剩余 ID。Manifest 只创建一次，复用时核对元数据指纹和划分完整性。

所有 split 均保留官方随机裁剪。每次评估按种子重置 Python/NumPy/PyTorch CPU/CUDA RNG，固定数据顺序和 DataLoader generator；workers 由 `torch.initial_seed()` 初始化，`persistent_workers=False`。同一 Dev 种子跨 Checkpoint 的裁剪序列一致，不同种子允许不同。结束后恢复训练 RNG 与训练模式。

调度依据 `FS-LPRF_Codex_Training_Validation_Refactor_v1.0.md` 更新为：

- Step 200/400/600 等非 800 倍数：轻量 Dev，共 15 次。只用 seed=42 的原始 10-shot Support；每类从完整 Dev 的固定 Query 中最多取 1000 条，小类使用全部剩余 Query。32 Fake 类全部参加、Real Query 共享，仍逐类等权 Macro。轻量结果仅用于监控，不选 Best、不存 CKPT。
- Step **800/1600/2400/3200/4000**：先保存 `checkpoints/step_XXXX.pt`，再执行完整五种子、32 类、全量 Query 的 Dev。每轮输出 160 行逐类逐种子明细、32 行逐类 Mean±Std、Macro 和逐条预测，并在终端/日志打印全部类别。里程碑不重复轻量验证。
- 仅完整 Dev 的五种子 Macro EER 均值更新 Best，平局取较早 Step。`checkpoints/best_checkpoint.json` 指向真实节点，不再每 Step 写 `last.pt`，也不复制 `best.pt`。默认共 5 份可恢复节点。

轻量清单固定保存为 `evaluation/dev_light_manifest.json`，绑定源完整清单 SHA-256、数据指纹与配额；使用 SHA 派生的局部 RNG，不改变训练 RNG。源文件、数据或轻量配置变化时拒绝复用。原完整 Dev/Test Support/Query 排除规则不变。

恢复需要原输出目录、配置、两个 Dev Manifest、节点和完整 Dev 历史。模型后端、Optimizer、Scheduler 与 RNG 一起恢复。节点已有但完整 Dev 完成标记/报告缺失时，会先补做验证再执行下一 Step。完成标记绑定节点和全部 5 份报告的 SHA-256；Best 从已完成历史重建。目录中有未来节点时拒绝倒退覆盖，请恢复最新节点；先前节点历史损坏则需恢复匹配的完整文件。中断最多丢失上一里程碑后未保存的 **799 Steps**，这是降低存盘频率的明确权衡。

```bash
python scripts/train.py --config configs/speechfake.yaml
python scripts/train.py --config configs/speechfake.yaml --resume outputs/speechfake_fs_lprf/checkpoints/step_0800.pt
# 完成训练和Dev选模后，才执行正式Test：
python scripts/evaluate.py --config configs/speechfake.yaml --checkpoint best --shot 10
# 也可明确指定任一节点路径（测试应使用最终选出的节点）：
python scripts/evaluate.py --config configs/speechfake.yaml --checkpoint outputs/speechfake_fs_lprf/checkpoints/step_0800.pt --shot 10
```

以上正式命令只在 AutoDL 执行。每步训练实时写终端、`train.log` 和 TensorBoard 的 `train/loss`、`train/learning_rate`、`train/gradient_norm`；验证分开写 `dev/light_macro_*` 和 `dev/full_macro_*_mean/std`，不混为一条曲线。恢复时 `purge_step` 隐藏未保存训练步的旧事件，Episode 追加到新的 resume 日志段，保留旧尝试记录。

报告位于 `evaluation/light/` 与 `evaluation/full/`，每个 Step 独立命名。CSV 数值为比例（0~1），日志显示百分比；EER threshold 仅为 EER 产物，ACC 始终按 `score>=0` 判断 Fake。Test 不参与训练内监控、选模或调参，原 Test 输出格式不变。详细配置、文件清单及恢复行为见 [docs/VALIDATION_SCHEDULE.md](docs/VALIDATION_SCHEDULE.md)。

## 轻量测试

```bash
python -m pytest -q
python scripts/check_model.py --config configs/speechfake.yaml
```

Pytest 使用合成向量和临时 WAV，包含梯度、损失聚合、一次更新、音频处理、五种子固定划分、0/2 workers 的裁剪复现、RNG 恢复、合成 33 类原型库与 Checkpoint 往返测试，并覆盖阻止外部包导入时的全部模块加载、小型 Transformer 与原固定版本输出对照、逐层 Attention 数值/梯度及 EER 同分阈值。`check_model.py` 使用正式模型、真实权重和一条合成 4 秒 WAV，核对 25 组 Hidden States、池化/投影形状、前端冻结、所有层 Attention 与共享 Bottleneck 梯度，以及原有后端 Checkpoint 保存逻辑；不执行优化器更新。报告写入 `outputs/local_smoke/model_report.json`。本地实测记录见 [docs/LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md)。

## 0-shot 接口（本阶段仅合成测试）

`methods.inference` 提供参数化 K-shot、33 类 PrototypeBank 与 0-shot 分数；0-shot 在每层取 32 个 Fake 原型的最大相似度，再减 Real 并使用统计 α 融合。`scripts/build_prototype_bank.py` 已实现训练集无增强构建：Real 从 AISHELL-1、AISHELL-3、LibriTTS、VCTK 各不放回抽取 256 条，共 1024 条，仍作为一个 Real 类；32 个 Fake 类各不放回抽取 1024 条。无需 CommonVoice，任一来源或类别容量不足均明确报错，不重复使用音频补足配额。它记录最终 Checkpoint、样本 ID、数据指纹和版本。本阶段不运行真实构建或 0-shot 实验，不包含其他数据集、基线或消融。

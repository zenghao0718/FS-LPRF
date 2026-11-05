# AutoDL A100 40GB 运行步骤

以下命令用于后续正式运行。本地开发阶段不执行训练、完整 Dev/Test 或真实原型库构建。项目只需要一张 A100；默认 FP32、冻结前端，不额外启用混合精度或修改 Episode 协议。

## 1. 拉取代码

```bash
cd /root/autodl-tmp
git clone https://github.com/zenghao0718/FS-LPRF.git
cd FS-LPRF
```

运行所需的 Layer-Wise/XLS-R 源码已整合到 `data/`、`models/`、`methods/`、`engine/` 和 `utils/`，无需初始化任何子模块或另外安装 `tomato`/Fairseq。已有旧版工作目录时，建议保留其数据、权重和运行输出，在新目录普通 clone 后复用资源，避免覆盖旧实验。

后续命令均在上面的 `FS-LPRF` 项目根目录执行。依赖仍从 `requirements.txt` 安装；不需要可编辑安装本项目或手工配置 Python 路径。目录和模块说明见 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)，既有数据、权重和运行输出路径不变。

## 2. 环境部署

AutoDL 是专门用于训练的实例，**默认直接使用预装的 `base` 环境，不要求新建 Conda 或 venv**。优先选择项目已经验证的 **Python 3.11** 镜像；若当前激活了其他环境，先通过下列命令确认，再切回预装 `base`。部署前必须检查预装 Python、torch、torchaudio、GPU 驱动和 CUDA 兼容性，先检查再决定是否安装：

```bash
nvidia-smi
conda env list
which python
python --version
python -m pip --version
python -m pip show torch torchaudio
python - <<'PY'
import sys
print('Python:', sys.version)
assert sys.version_info >= (3, 11), '项目要求 Python >=3.11，优先使用已验证的 3.11'
import torch
import torchaudio
print('torch:', torch.__version__)
print('torchaudio:', torchaudio.__version__)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
assert torch.__version__.split('+', 1)[0] == '2.7.1', '需要 torch 2.7.1'
assert torchaudio.__version__.split('+', 1)[0] == '2.7.1', '需要 torchaudio 2.7.1'
assert torch.__version__.partition('+')[2] == torchaudio.__version__.partition('+')[2], 'torch/torchaudio 需要匹配的 CUDA 构建'
assert torch.version.cuda is not None and torch.cuda.is_available(), '需要可用的 CUDA 构建及兼容驱动'
print('GPU:', torch.cuda.get_device_name(0))
print('GPU capability:', torch.cuda.get_device_capability(0))
print('CUDA tensor check:', torch.ones(1, device='cuda').sum().item())
PY
```

导入失败、版本断言失败或 CUDA 张量检查失败时，先定位缺失或不兼容的组件。Python 不符合项目要求时，优先改用预装 Python 3.11 的 AutoDL 镜像及其 `base`，无需新建环境。

`requirements.txt` 固定的是 `torch==2.7.1`、`torchaudio==2.7.1`，不固定 `+cu126` 或 `+cu128`。`2.7.1+cu126`、`2.7.1+cu128` 等兼容 CUDA 构建都满足该版本约束。**修改 `requirements.txt` 不等于强制所有 GPU 使用相同的 CUDA 构建**；本机的 `+cu128` 组合无需照搬到 AutoDL。本项目不需要额外安装 `torchvision`。

环境检查通过时，**直接复用预装 PyTorch/torchaudio，跳过下面的 Wheel 安装命令，不重复下载**，直接进入其余依赖安装。

仅当 torch/torchaudio 缺失、版本不是 2.7.1、二者构建不匹配，或现有 CPU/CUDA 构建不兼容时，才安装或替换这两个组件。依据实际 GPU 驱动、GPU 架构和部署环境，从 [PyTorch 官方 2.7.1 历史版本说明](https://pytorch.org/get-started/previous-versions/#v271) 中选择兼容的 `cu118`、`cu126` 或 `cu128` Wheel。以下 `cu126` **仅为已确认驱动与 GPU 兼容后的示例**，不能无条件照抄：

```bash
# 仅在需要安装/替换时执行；按实际兼容性选择 cu118、cu126 或 cu128。
TORCH_CUDA_WHEEL=cu126
python -m pip install \
  "torch==2.7.1+${TORCH_CUDA_WHEEL}" "torchaudio==2.7.1+${TORCH_CUDA_WHEEL}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_WHEEL}"
```

这里仅在条件安装命令中指定所选构建，确保同为 2.7.1 的 CPU 或其他不兼容构建也会被替换；`requirements.txt` 仍只固定公开版本号。替换范围为 torch、torchaudio 及它们所需的配套 CUDA 运行库，由 pip 解析；其他项目依赖继续按原有固定版本安装。安装后重新执行上面的环境检查，确认版本、导入和 CUDA 张量计算均通过。

随后优先使用国内镜像补齐其他依赖，并验证。不要添加 `--upgrade` 或 `--force-reinstall`；已满足版本约束的 PyTorch/torchaudio 会被复用，保留现有 CUDA 构建：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
```

## 3. 权重与 RIR 资源

```bash
mkdir -p /root/autodl-tmp/pretrained
python scripts/download_frontend.py --output /root/autodl-tmp/pretrained/xlsr2_300m.pt
```

该官方原始 Fairseq 文件为 3,808,868,242 字节；下载器生成 `.source.json` 记录 SHA-256。已有完整且哈希正确的文件会直接复用，不重复下载。保留 `.pt` 原始格式，不替换为 Hugging Face 权重；整合后的模型仍直接加载该文件并严格核对参数名称。

环境与权重就绪后，先运行不依赖真实数据集或 RIR 的单元测试和真实前端 Smoke Test：

```bash
python -m pytest -q
python scripts/check_model.py --config configs/speechfake.yaml \
  --set model.frontend_path=/root/autodl-tmp/pretrained/xlsr2_300m.pt \
  --report /root/autodl-tmp/fs_lprf_runs/autodl_model_smoke.json
```

Smoke Test 只使用一条合成 4 秒 WAV，检查 25 组 Hidden States、Attention `[1,25,1024]`、投影 `[1,25,256]`、前端冻结、364800 个可训练参数的梯度及后端 Checkpoint 往返，不执行优化器更新。通过后再准备 RIR 和真实数据，后续正式实验由用户在 AutoDL 执行。

RIR 资源由用户准备；若尚未准备，可在 AutoDL 下载 OpenSLR 28：

```bash
mkdir -p /root/autodl-tmp/resources
wget -c https://openslr.magicdatatech.com/resources/28/rirs_noises.zip \
  -O /root/autodl-tmp/resources/rirs_noises.zip
unzip /root/autodl-tmp/resources/rirs_noises.zip -d /root/autodl-tmp/resources
```

上述地址是 [OpenSLR 28 页面列出的中国镜像](https://www.openslr.org/28/)。`data.rir_root` 指向解压后的 `RIRS_NOISES` 根目录，例如 `/root/autodl-tmp/resources/RIRS_NOISES`。`rir_list` 是普通文本索引文件，程序必须读取以下四份索引：

- `simulated_rirs/smallroom/rir_list`
- `simulated_rirs/mediumroom/rir_list`
- `simulated_rirs/largeroom/rir_list`
- `real_rirs_isotropic_noises/rir_list`

索引记录支持 [Kaldi 的 RIR 元数据字段](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/data/reverberate_data_dir.py)，例如 `--rir-id=example --room-id=Room001 RIRS_NOISES/simulated_rirs/smallroom/Room001/a.wav`。路径可以是 `RIRS_NOISES/` 前缀路径、相对资源根目录的 `simulated_rirs/...` 或 `real_rirs_isotropic_noises/...`、相对索引文件目录的路径，或根目录以内的绝对路径；带空格的路径使用引号。

仅索引中列出的实际 WAV 文件进入候选池，解析后验证根目录边界、去重并排序。任一索引缺失/为空，或记录指向缺失文件、非 WAV、越界路径时均明确报错；坏记录包含索引路径与行号，不返回部分候选池。未索引的真实目录噪声及 `pointsource_noises` 不会纳入，也不会递归扫描 WAV 来补全索引。

完整原始资源预期为 smallroom/mediumroom/largeroom 各 20,000 个模拟 RIR，加上 325 个真实 RIR，共 60,325 个；该数量用于资源核对，未写死在生产代码中。用户在 AutoDL 准备好现有资源后，可仅核对索引候选数量，不读写 Manifest：

```bash
python -c "from data.speechfake_dataset import ConfiguredRawBoost; print(len(ConfiguredRawBoost('/root/autodl-tmp/resources/RIRS_NOISES').rir))"
```

RIR 概率仍为 1/3，`noise_prob=1/5` 仍指 RawBoost Algorithm 5 的触发概率，同时触发时先 RIR 后 RawBoost；不会混入 OpenSLR 28 的普通环境噪声。RIR 卷积、功率归一化和 RawBoost 参数保持原有实现。

## 4. 数据路径与配置

用户完成 SpeechFake-BD 清理后，将无表头的 `train.tsv/train.txt/dev.tsv/dev.txt/test.tsv/test.txt` 放在例如 `/root/autodl-tmp/speechfake/manifests`。TSV 为 `uttid<TAB>audio_absolute_path`；TXT 为 `uttid<TAB>origin_ds<TAB>speaker<TAB>attacker<TAB>label`，详细约定见 README。三个 split 不得交叉使用同一音频。

BD baseline 的 Real 仍是一个类别，来源为 AISHELL-1、AISHELL-3、LibriTTS、VCTK；不需要为 Prototype Bank 加入 CommonVoice。后续仅从未增强的 Train 构建原型库时，四个来源各抽取 256 条，Real 合计 1024 条；32 个 Fake 类各自仍为 1024 条，不放回抽样，容量不足直接报错。

生成一份独立的 AutoDL 配置，保留已固定的模型与实验参数：

```bash
cp configs/speechfake.yaml configs/speechfake.autodl.yaml
python - <<'PY'
from pathlib import Path
import yaml
p = Path('configs/speechfake.autodl.yaml')
cfg = yaml.safe_load(p.read_text())
cfg['data']['data_path'] = '/root/autodl-tmp/speechfake/manifests'
cfg['data']['rir_root'] = '/root/autodl-tmp/resources/RIRS_NOISES'
cfg['model']['frontend_path'] = '/root/autodl-tmp/pretrained/xlsr2_300m.pt'
cfg['runtime']['device'] = 'cuda:0'
cfg['cache']['root'] = '/root/autodl-tmp/fs_lprf_cache'
cfg['runtime'].pop('num_workers', None)
cfg['runtime']['train_num_workers'] = 4
cfg['runtime']['eval_num_workers'] = 2
cfg['runtime']['output_dir'] = '/root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf_cache_v1'
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

若沿用旧配置，请删除 `train.validation_interval_steps`，加入 `train.light_validation_interval_steps: 200`、`train.full_validation_interval_steps: 800`、`validation.light_seed: 42`、`validation.light_max_query_per_class: 1000`。旧正式训练目录不会自动迁移为新调度；请保留它们并另选新输出目录。

默认 Train 4 / Eval 2 个多进程 Worker；训练 Loader 与持续 Worker 跨 Step 复用。可独立设置 0/2/4/8，并调整预取、pin memory、Worker Torch 线程；旧 `runtime.num_workers` 会发弃用提示并作为兼容覆盖值，应从新配置移除。逐样本 RNG 已与 Worker 生命周期解耦，恢复可更换纯性能参数。`audio_microbatch_size=32` 沿用当前 main，基准时固定；显存不足时只在新 Run 明确调整微批次，不改变每 Step Episode、Support/Query 或预算。Feature Cache 预取的 RAM 代价较大，应实测较小 prefetch（如 1）。

新管线默认 Train `waveform`、Dev/Test `xlsr_feature`、FP32。必须使用新的 `output_dir`，保留旧实验目录；旧输入管线的中间状态不可直接接着训练。缓存构建、Train feature 对照与 Baseline 的完整说明见 [DATA_CACHE.md](DATA_CACHE.md)。

## 5. 开跑前检查

```bash
python scripts/check_data.py --config configs/speechfake.autodl.yaml --splits train dev
```

数据准备完成后才执行本节检查，核对 Train/Dev 路径、类别与容量，保存固定 Dev Manifest。检查器仍保留默认三个 split 的原检查，也可显式选择 split；训练前使用上面的选择不打开 Test。训练启动前再次验证两个 Dev Manifest；清单或配置变化明确报错。第 3 节 Smoke 应已通过；更换权重或环境后重新执行。

先估算空间，使用少量样本做缓存 Smoke，通过后再构建完整缓存（`--limit` 产物禁止正式训练使用）：

```bash
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode waveform --dry-run
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode waveform --limit 1000
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split dev --mode xlsr_feature \
  --validation-modes light,full --limit 100 --seeds 42
# 完整正式缓存
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode waveform
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split dev --mode xlsr_feature \
  --validation-modes light,full --dry-run
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split dev --mode xlsr_feature \
  --validation-modes light,full
```

超出磁盘阈值需确认或显式 `--confirm-large-cache`；中断使用 `--resume`，重建 `--force` 保留旧缓存。训练只检查匹配缓存，不会隐式生成。先在 AutoDL 比较 `none,waveform` 的 Worker 0/2/4/8（5 warmup + 20 timed），准备完整 Train feature 缓存后再加入 `xlsr_feature`，命令与总实验时间计算见 [DATA_CACHE.md](DATA_CACHE.md)。真实性能与完整 Dev 空间尚需 AutoDL 实测。

## 6. 正式训练与恢复

```bash
python scripts/train.py --config configs/speechfake.autodl.yaml
```

共 4000 次优化器更新，每步 8 个独立 Episode。每 200 Steps 的非 800 倍数执行轻量 Dev（seed 42、每类最多 1000 Query）；800/1600/2400/3200/4000 先保存独立节点，再执行完整五种子 Dev。共 15 次轻量 + 5 次完整，只有完整 Dev 的 Macro EER 均值参与 Best，平局取较早节点。Best 由 JSON 索引指向现有节点，不复制 `best.pt`、不生成逐步 `last.pt`。中断最多损失最近节点后 799 个未保存更新。

恢复时保留原配置、两个 Dev Manifest 和完整输出历史，使用最新可用节点。若该节点的完整 Dev 未成功结束，先补做再继续下一 Step：

```bash
python scripts/train.py --config configs/speechfake.autodl.yaml \
  --resume /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf_cache_v1/checkpoints/step_0800.pt
```

可在另一个终端查看日志和 TensorBoard：

```bash
tail -f /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf_cache_v1/train.log
tensorboard --logdir /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf_cache_v1/tensorboard \
  --host 0.0.0.0 --port 6006
```

## 7. Test 10-shot

本机不执行真实 Test。后续训练、Dev 选模与调参全部完成后，以下入口才解析/加载 Best，并生成或读取 Test 前端缓存，随后按原结构输出正式指标。显式节点路径必须与 Best 索引一致；普通缓存命令不能生成 Test。可先用同一命令加 `--cache-dry-run` 只估算，不计算指标；超阈值需确认后加 `--confirm-large-cache`。首次缓存创建与正式评估耗时分开记录，不能把缓存后重复读取速度当作首次收益。

```bash
python scripts/evaluate.py --config configs/speechfake.autodl.yaml \
  --checkpoint best \
  --shot 10
```

程序按 17、42、71、100、123 运行，固定共用 Query，逐类计算 EER/AUC/AP/ACC。保存五个 `test_per_class_seed*.csv`、`test_macro_per_seed.csv`、`test_macro_mean_std.csv`、`test_predictions.csv` 和 `test_run.json`。分数越大越倾向 Fake，ACC 阈值为 0。所有指标为比例数（例如 0.05 = 5%），Std 使用 ddof=0。

## 8. 交付产物与范围

正式输出目录包含 `config_resolved.yaml`、`run_state.json`、`cache_bindings.json`、`cache_metadata/`、`train.log`、Episode 日志、`tensorboard/`、5 份节点、Best 索引及 `evaluation/`。完整/轻量报告与完成标记保留原结构。续跑应保留整个目录及匹配缓存，缓存内容/模式/RNG schema 不一致会拒绝恢复；纯 Worker 性能参数可调整。具体文件名和恢复限制见 [VALIDATION_SCHEDULE.md](VALIDATION_SCHEDULE.md)。

不要将数据、RIR、预训练文件或任何训练 Checkpoint 提交 Git。当前阶段不运行真实 0-shot、其他 Shot、基线、消融或 ASVspoof。原型库构建脚本仅供后续阶段调用，已通过合成数据接口测试。

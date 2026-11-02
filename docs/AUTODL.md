# AutoDL A100 40GB 运行步骤

以下命令用于后续正式运行。本地开发阶段不执行训练、完整 Dev/Test 或真实原型库构建。项目只需要一张 A100；默认 FP32、冻结前端，不额外启用混合精度或修改 Episode 协议。

## 1. 拉取代码

```bash
cd /root/autodl-tmp
git clone --recurse-submodules https://github.com/zenghao0718/FS-LPRF.git
cd FS-LPRF
git submodule update --init --recursive
git -C third_party/tomatoDD rev-parse HEAD
# 必须为 88efe1376dfc5f5b9b62d016d33a7381c158d8af
```

## 2. 环境部署

先检查当前环境与驱动：

```bash
nvidia-smi
conda env list
python --version
python -c "import torch,torchaudio; print(torch.__version__,torchaudio.__version__,torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

已有 Python 3.11、兼容 CUDA 的 PyTorch/torchaudio 时直接复用。缺少合适环境时：

```bash
conda create -n fs-lprf python=3.11 -y --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda activate fs-lprf
```

仅当该环境没有兼容 PyTorch 时，按 AutoDL 驱动选择对应版本；例如兼容 CUDA 12.6 的 A100 环境可安装：

```bash
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

本机 CUDA 12.8 组合无需照搬到 AutoDL。PyTorch 官方历史版本命令见 [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)。随后安装其余依赖并检查：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
python -m pytest -q
```

## 3. 权重与 RIR 资源

```bash
mkdir -p /root/autodl-tmp/pretrained
python scripts/download_frontend.py --output /root/autodl-tmp/pretrained/xlsr2_300m.pt
```

该官方原始 Fairseq 文件为 3,808,868,242 字节；下载器生成 `.source.json` 记录 SHA-256。保留 `.pt` 原始格式，不替换为 Hugging Face 权重。

RIR 资源由用户准备；若尚未准备，可在 AutoDL 下载 OpenSLR 28：

```bash
mkdir -p /root/autodl-tmp/resources
wget -c https://openslr.magicdatatech.com/resources/28/rirs_noises.zip \
  -O /root/autodl-tmp/resources/rirs_noises.zip
unzip /root/autodl-tmp/resources/rirs_noises.zip -d /root/autodl-tmp/resources
```

上述地址是 [OpenSLR 28 页面列出的中国镜像](https://www.openslr.org/28/)。`data.rir_root` 指向解压后的 `RIRS_NOISES` 根目录，保持官方递归 WAV 搜索与增强逻辑。

## 4. 数据路径与配置

用户完成 SpeechFake-BD 清理后，将无表头的 `train.tsv/train.txt/dev.tsv/dev.txt/test.tsv/test.txt` 放在例如 `/root/autodl-tmp/speechfake/manifests`。TSV 为 `uttid<TAB>audio_absolute_path`；TXT 为 `uttid<TAB>origin_ds<TAB>speaker<TAB>attacker<TAB>label`，详细约定见 README。三个 split 不得交叉使用同一音频。

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
cfg['runtime']['num_workers'] = 0
cfg['runtime']['output_dir'] = '/root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf'
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

默认 `num_workers=0` 最容易核对裁剪复现；需要提高吞吐量时可在训练开始前改为 4 等值，之后验证与恢复保持一致。`audio_microbatch_size=8` 是 A100 默认；如显存受其他进程影响，仅调整音频微批次，不静默改变每 Step Episode 数、每 Episode Support/Query 或训练预算。

## 5. 开跑前检查

```bash
python scripts/check_data.py --config configs/speechfake.autodl.yaml
python scripts/check_model.py --config configs/speechfake.autodl.yaml \
  --report /root/autodl-tmp/fs_lprf_runs/autodl_model_smoke.json
```

数据检查核对所有路径、类别与容量，并保存固定 Dev/Test Manifest。训练启动前会再次验证 Dev Manifest；已存在时直接复用，清单或配置变化会明确报错。Smoke Test 只检查一条临时 WAV 的真实模型链路。

## 6. 正式训练与恢复

```bash
python scripts/train.py --config configs/speechfake.autodl.yaml
```

共 4000 次优化器更新，每步 8 个独立 Episode。每 200 Steps 及最后一步执行五种子 Dev 验证，以五种子 Macro EER 均值最低者保存 `best.pt`；每步更新 `last.pt`。恢复时保留原输出目录及 Dev Manifest：

```bash
python scripts/train.py --config configs/speechfake.autodl.yaml \
  --resume /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf/checkpoints/last.pt
```

可在另一个终端查看日志和 TensorBoard：

```bash
tail -f /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf/train.log
tensorboard --logdir /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf/tensorboard \
  --host 0.0.0.0 --port 6006
```

## 7. Test 10-shot

训练完成后，只使用选出的 Best Checkpoint：

```bash
python scripts/evaluate.py --config configs/speechfake.autodl.yaml \
  --checkpoint /root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf/checkpoints/best.pt \
  --shot 10
```

程序按 17、42、71、100、123 运行，固定共用 Query，逐类计算 EER/AUC/AP/ACC。保存五个 `test_per_class_seed*.csv`、`test_macro_per_seed.csv`、`test_macro_mean_std.csv`、`test_predictions.csv` 和 `test_run.json`。分数越大越倾向 Fake，ACC 阈值为 0。所有指标为比例数（例如 0.05 = 5%），Std 使用 ddof=0。

## 8. 交付产物与范围

正式输出目录包含 `config_resolved.yaml`、`train.log`、`episodes.jsonl`、`tensorboard/`、`checkpoints/best.pt`、`checkpoints/last.pt`、`evaluation/`。续跑时这些文件应一起保留，尤其是原 Dev Manifest。

不要将数据、RIR、预训练文件或任何训练 Checkpoint 提交 Git。当前阶段不运行真实 0-shot、其他 Shot、基线、消融或 ASVspoof。原型库构建脚本仅供后续阶段调用，已通过合成数据接口测试。

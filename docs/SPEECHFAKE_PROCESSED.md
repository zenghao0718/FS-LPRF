# SpeechFake_Processed

独立数据包目录。Train/Dev 使用已经构建的 FP32 Waveform Cache，Test 使用原始 WAV。六份 Manifest 保持构建缓存时的原始字节，其中的绝对路径仍然是缓存身份的一部分，不能改写。

```text
SpeechFake_Processed/
├── manifests/
│   ├── train.tsv train.txt dev.tsv dev.txt test.tsv test.txt
├── train/waveform/<fingerprint>/
├── train/waveform/references/
├── dev/waveform/<fingerprint>/
├── dev/waveform/references/
└── test/
    └── 与 /root/autodl-tmp/SpeechFake_Data/ 之后相对路径相同的 Test WAV
```

解压到 `/root/autodl-tmp/SpeechFake_Processed` 后，使用唯一配置 `configs/speechfake.yaml`。其中的资源路径是：

```yaml
data:
  data_path: /root/autodl-tmp/SpeechFake_Processed/manifests
  train_cache_mode: waveform
  dev_cache_mode: waveform
  test_cache_mode: none
cache:
  root: /root/autodl-tmp/SpeechFake_Processed
```

`cache.root` 指向数据包根目录。程序会在其下读取 `train/waveform` 和 `dev/waveform`。Test 打开文件时，把清单中的前缀 `/root/autodl-tmp/SpeechFake_Data/` 映射到 `<data_path 的父目录>/test/`。映射不修改 `uttid2path`，因此数据集指纹、Manifest SHA256 和缓存目录名保持不变。

Train/Dev 不读取原始 WAV。`data/audio.py` 的字节不能改变，它的 SHA256 写在缓存身份里。XLS-R 权重和 RIR 仍由部署环境单独提供。

检查：

```bash
python scripts/build_cache.py --config configs/speechfake.yaml --split train --mode waveform --check
python scripts/build_cache.py --config configs/speechfake.yaml --split dev --mode waveform --check
python scripts/check_data.py --config configs/speechfake.yaml --cache-only --splits train dev --read-samples 8
python scripts/check_data.py --config configs/speechfake.yaml --splits test --read-samples 8
```

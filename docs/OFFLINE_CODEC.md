# 离线 Codec 波形缓存与开发单种子 Dev

本轮入口只读取已经完整的 **Train Waveform Cache** 和原 Train 清单；不读 Train WAV、不重建 Train/Dev 源缓存、不修改模型、标签或 Episode/优化器。默认 `enabled=false` 保持旧 Baseline 的张量、随机数和运行身份。旧配置 `enabled=true` 仍默认为 online。

## 数据与缓存身份

`codec_waveform` 是独立 FP32 `[1,T]` 缓存，T 等于源缓存整条语音长度。每条语音按 `(build_seed, uttid, builder_version)` 用局部 RNG 等概率抽家族，再等概率抽该家族预设，**只抽一次**。训练每次抽样以 50% 概率读取原版或该固定编码版，之后走原有随机 4 秒裁剪、RIR、RawBoost。离线训练不构造 CodecAugmentor、不调用编码器、不加载 EnCodec。

身份绑定源缓存 fingerprint/content fingerprint、历史源音频摘要、清单哈希、ID 顺序、预处理、build_seed、家族/预设/实际参数、实现文件 SHA256、软件版本、ffmpeg 可执行文件及 Linux 动态编码库哈希、AMR 动态库哈希、EnCodec 已加载权重 SHA256（按排序后的 state_dict 名称/形状/类型/原始字节计算）和源码 Commit。自己的分片/index/content fingerprint 进入训练 cache_bindings 和 run_identity。训练只比对记录、源码和源缓存，不探测编码后端。更新编码实现会拒绝旧 Codec Cache；原 waveform/xlsr_feature 身份保持原格式。

复用 CacheWriter 的 OS 写锁、fsync、原子分片、完整索引、分片 SHA256、原子 resume.json 和发布机制。未完成缓存只留在 `.building`，只有通过校验的完整缓存才发布。中断只重算尚未提交的分片；不提供覆盖参数。相同目录重复构建会报错，变更身份拒绝续建；用新目录保留旧缓存。失败样本在 `.building/failure.json` 留下 ID、参数和错误，成功记录在每条 index 的 `codec` 中。成功续建后，历史 failure.json 可留作诊断，不代表当前完成状态。

编码前逐个家族/预设真实探测，任何缺失都报错，不能静默减少预设。使用非平稳浊音 chirp 的语音频带相关逐预设校准延迟，取代旧的脉冲起点阈值；校准相关不足直接拒绝。编码增加保护尾，记录原始/解码/输出长度、重复/保护尾长度、裁剪/补齐比例、峰值、削波比例、起止 RMS 和编码耗时。严重缺样（大于 min(16 samples, 1%)）、严重削波（峰值 >2 或原音频窗口越界比例 >1%）、空/非有限输出、非静音塌缩会报错。预检还对源缓存最短/中位/最长真实波形逐预设检查起止窗口的语音频带相关与残余延迟；低相关明确标记 inconclusive，不冒充已确认对齐。可靠相关下超过 30 ms 残余延迟拒绝构建。极短音频允许先重复到编码器最低长度，再恢复原长。

## AutoDL：先做 1000–3000 条 smoke

以下为后续 AutoDL 命令，**本机没有执行**。仓库实际目录以 AutoDL checkout 为准。先激活原项目 Python 环境，再在仓库根目录执行：

```bash
git pull --ff-only
git status --short --branch
df -h /root/autodl-tmp
python scripts/check_data.py --config configs/speechfake.yaml --cache-only --splits train dev --read-samples 8
ffmpeg -version
ffmpeg -hide_banner -encoders | grep -E 'libmp3lame|libopus'
ldconfig -p | grep -E 'opencore-amrnb|opencore-amrwb|vo-amrwbenc'
python -c 'import encodec, torch; print(encodec.__file__); print(torch.__version__)'
```

缺依赖时显式安装当前 AutoDL 系统对应的 ffmpeg（需 libmp3lame/libopus）、OpenCORE AMR-NB/WB、VisualOn AMR-WB 编码库，以及 requirements.txt 中的 encodec；不要把删掉家族当作默认修复。EnCodec 预检会加载 24 kHz 预训练权重，首次可能需要联网下载，失败会明确报错。

```bash
SMOKE=/root/autodl-tmp/fs_lprf_codec_smoke_2000_seed42
python scripts/build_codec_cache.py --config configs/speechfake.yaml \
  --output-dir "$SMOKE" --limit 2000 --dry-run
python scripts/build_codec_cache.py --config configs/speechfake.yaml \
  --output-dir "$SMOKE" --limit 2000
python scripts/build_codec_cache.py --config configs/speechfake.yaml \
  --output-dir "$SMOKE" --verify
```

`--output-dir` 就是完整缓存目录，没有隐藏路径后缀。`--limit` 按 33 类和各类长度范围分层选样，保持源 ID 顺序，明确标记 smoke；即使 limit 恰好等于源样本数也不能用于正式训练。`--verify` 不需要编码后端，始终检查全部分片哈希、索引、记录、源绑定和 PCM 有限值。空间估计从当前源索引的实际长度求 FP32 字节数，额外留元数据/分片余量，默认保留 5 GiB；失败即停止。续建扣除已提交的 PCM 字节，源 PCM 不会写入。

在上述首次构建开始并打印至少一个 `cache done/total` 后，按 Ctrl-C 中断，然后使用**相同参数**续建；不要等已完成后再尝试模拟恢复：

```bash
python scripts/build_codec_cache.py --config configs/speechfake.yaml \
  --output-dir "$SMOKE" --limit 2000 --resume
python scripts/build_codec_cache.py --config configs/speechfake.yaml \
  --output-dir "$SMOKE" --verify
```

保留 metadata.json、index.json 和 build_report.json。报告含每预设真实样本预检、实际构建耗时/PCM 占用、按真实源长度估计的全量 PCM 字节数以及基于 smoke 速度的全量耗时投影。预检成本、一次性构建成本和以后每次训练成本应分别看待。全量投影不等于实测；建议核查源样本数（历史为 682,205，实际以索引为准）、每类覆盖、输入/输出长度、极低码率、静音与起止波形。可用下面脚本额外抽样导出少量原版/编码版，输出仅供本地核查，不提交 Git：

```bash
python scripts/inspect_codec_cache.py --config configs/speechfake.yaml \
  --cache-dir "$SMOKE" --samples 12 --output-dir /root/autodl-tmp/codec_inspection_seed42
FS_LPRF_CODEC_SPEECH_FIXTURE=/root/autodl-tmp/codec_inspection_seed42/011_original.wav \
  python -m pytest -q tests/test_codec_augmentation.py tests/test_codec_real_boundaries.py
```

后一个命令使用刚从源缓存导出的最长样本作真实语音 fixture，逐家族/预设检查短音频、静音、边界及低码率；它只做 CPU 编解码测试。没有系统依赖或没有该真实语音 fixture 时会明确 skip，不能计为真实 Codec 通过。已安装但编码失败、权重加载失败或预设不支持会 fail。

## AutoDL：受限 Episode 测速

```bash
python scripts/benchmark_pipeline.py --config configs/speechfake.yaml \
  --modes waveform,waveform_codec,waveform_codec_offline \
  --workers 4,8 --warmup 5 --steps 30 \
  --smoke-codec-cache "$SMOKE" \
  --output /root/autodl-tmp/fs_lprf_codec_benchmark_seed42
```

此命令使用真实模型和临时优化器，最多每个模式/worker 35 Step，不启动正式训练，不保存训练 Checkpoint，不执行 Dev/Test。全部模式限定在该 smoke 缓存共同覆盖的池内，保持原 3-way、每类 5 Support/5 Query、每 Step 8 Episode、microbatch 和相同 Episode ID（报告 `episode_sha256`），每类不足 10 条直接报错。正式训练入口不允许 smoke；不带 `--smoke-codec-cache` 的离线测速同样拒绝 smoke。

JSON/CSV 分别记录 mean/p50/p90 Step、数据等待、吞吐、NVML GPU 利用率、CPU/RSS、峰值显存，以及源/Codec Cache 构建时间和字节数。NVML 不可用时返回空值和原因。在线 latency probe 逐预设计时，用于独立估计构建成本；构建报告同时含每条实际耗时和输入长度。`projected_4000_step_training_seconds` 是基于该受限抽样的训练耗时投影，不能报告为全量实测；`one_time_codec_build_seconds` 单独列出。没有本次 Dev 实测就不生成实验总耗时。历史 2.694 / 11.990 秒每步和五种子 Dev 约 21.75 小时仅供参考，不代入本次实测。

构建器目前为单进程顺序编码、父进程独占写分片；训练读取支持 workers=4/8。全量构建应优先参考 build_report.json 的实际单进程 smoke 速度和投影。旧 latency 估算接口中的 `parallel_seconds` 只是理想均摊假设，本构建 CLI 没有并行编码参数，不能把该字段当作本实现可达到的构建耗时。

## 开发 Full Dev 单种子

只接受 `[42]` 或官方 `[17,42,71,100,123]`。默认 YAML 和 Test 五种子不变，ASVspoof5 独立 `--seeds` 不变。Light 仍固定 seed42、Query 是对应 Full 的确定性子集，Support/Query 不相交。每 200/800 Step 的时序不变，仅完成的 Full 选 Best。单种子报告显示 seed_count=1、aggregation=single_seed、std=0，不写成五种子均值；结果只作开发选模。

先用轻量纯逻辑/合成指标测试确认 Manifest、指标、Best/Resume 和跨协议拒绝（不会跑一次真实 Full Dev）：

```bash
python -m pytest -q tests/test_development_validation.py tests/test_light_manifest.py tests/test_validation_schedule.py
```

将来获确认后，开发运行使用 `--set 'validation.seeds=[42]'`，并指定全新的 `runtime.output_dir`；离线模式额外指定 `data.codec_augment.enabled=true`、`data.codec_augment.mode=offline` 和完整 Codec Cache 路径。本文刻意不提供可直接触发 4000 Step 的完整训练命令。Manifest、run_identity、Checkpoint、完成标记和 Best 都绑定协议；改变种子不能复用旧运行目录或 Resume。旧五种子 Baseline Checkpoint 不因新增默认关闭配置而失效。

**用户确认前，不执行全量 Cache 构建、正式 4000 Step 或一次完整真实 Full Dev。** AutoDL smoke/测速后汇报实际耗时、磁盘占用、依赖/权重、波形核查和剩余风险，再决定全量方案。

## 本机验收记录（2026-11-09）

基线为 `736ec767c0571c22e4a0587c913c6093a78cc9db`。Windows / Python 3.11.18 / PyTorch 2.7.1+cu128：`python -m pytest -q` 最终 **432 passed, 33 skipped**，用时 414.30 秒。跳过项为缺少 ffmpeg/AMR/EnCodec 的真实 Codec 用例及未提供真实语音 fixture 的用例，不计为真实编解码通过。保留现有 NumPy/librosa/weight_norm 警告；模拟测速的无更新优化器触发一条 scheduler 顺序提示，不影响正式训练器。

合成小缓存已覆盖读回/整条长度/标签、原缓存不变、去除 WAV 后构建与训练读取、中断/恢复/身份冲突、分片/索引/元信息破坏、强制源/目标校验、smoke 正式训练拒绝、0/4/8 workers 分支一致、CLI dry-run/build/verify/抽样导出、三模式同 Episode 的受限测速，以及单种子 Dev 的 Manifest/指标/Best/Resume/跨协议拒绝。两个新 CLI 的 `--help` 独立执行通过，`git diff --check` 通过。模型、methods、data/audio.py、data/augmentation.py 无修改。

本机未执行真实数据全量构建、正式训练或完整真实 Full Dev；没有本次 AutoDL GPU 速度/全量容量实测。后续按本文 smoke 命令取得本次真实数据，再决定耗时操作。

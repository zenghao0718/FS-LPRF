# 数据缓存与 DataLoader 管线

按《FS-LPRF 数据缓存与 DataLoader 加速 Codex 实施规范 v1.0》实现。属于工程优化，Episode、Loss、优化器、4000 Step、15 次 Light / 5 次 Full Dev、Full-only Best 和后端 Checkpoint 参数键保持原样。以下正式命令由用户在 AutoDL 执行，本机不运行真实训练、完整 Dev/Test 或原型库。

## 默认模式与边界

- Train 默认 `data.train_cache_mode: waveform`：缓存完整基础预处理 `[1,L]` float32 波形；每次取样仍调用原来的 `FADTrim.sample_segment()`，再动态执行 RIR 1/3 与 RawBoost Algorithm 5 1/5，独立抽样、先 RIR 后 RawBoost。Support/Query 均如此。增强后不新增归一化，模型输入仍严格为有限的 `[1,64000]`。
- Dev 默认 `data.dev_cache_mode: xlsr_feature`：分别保存 Light seed42、Full seeds17/42/71/100/123 的完整前端状态；Support/Query 均无增强。每个模式/种子有独立身份，同一身份供所有 Checkpoint 复用。
- Test 默认 `data.test_cache_mode: xlsr_feature`：只有最终 `evaluate.py` 解析并加载已选出的 Best 后才能规划、创建或读取。训练只打开 Train/Dev，普通缓存构建命令拒绝 Test。首次 Test 仍需生成缓存，不承诺降低首次评测总时间。
- `none` 保留原始 WAV、基础预处理、裁剪、增强及在线 XLS-R 路径。对照时将 Train/Dev 设置为 `none`，Test 也可单独设为 `none`。
- Train 可选 `xlsr_feature`：每条 uttid 只预生成一个固定种子裁剪/增强 View，随后重复使用，训练不再裁剪、增强或运行 XLS-R。Episode 仍动态抽样。它牺牲在线随机性，不与 Waveform 方案宣称等价，不能作为默认。

缓存特征位于 Attention **之前**，每条为 `[25,T,1024]`：第 0 组为 `post_features`，之后严格对应 `layer_results[0..23][0]`。T 来自实际前端输出，不写死。读取 FP16 时在 Attention 前转回 FP32；25 组独立 Attention 和共享 Bottleneck 继续训练，前端保持冻结/eval。缓存从不保存 Attention/Bottleneck 输出、原型、α 或 logits。

`data/audio.py` 只抽出基础读取入口，SoundFile、重采样、首声道、原 trim 参数、-20 dB RMS、幅值裁剪和空音频错误处理不变。通用 `st/dur` 随机文件片段读取分支仍保留在原流程；完整波形缓存遇到这类记录会拒绝构建，提示使用 `none`，不会把随机片段冒充完整波形。当前 SpeechFake TSV/TXT 不含这些字段。

0-shot 库只允许在 Best 后使用无增强 Train。构建脚本直接读取原始 WAV；即使 Train 设置为 `xlsr_feature`，也绝不复用它的增强 View。库函数会明确拒绝 Train Feature Cache 输入，不使用 Dev/Test。

## 配置与新实验目录

保留原数据、RIR 和权重路径。AutoDL 建议缓存根目录 `/root/autodl-tmp/fs_lprf_cache/`，正式实验使用新的输出目录，例如 `/root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf_cache_v1`。旧输出与 Checkpoint 不迁移、不覆盖、不混合续训。本机默认 `cache/fs_lprf/` 已被 Git 忽略；项目内缓存根目录限于已忽略的缓存/输出目录，外部绝对路径也可用。

```yaml
data:
  train_cache_mode: waveform
  dev_cache_mode: xlsr_feature
  test_cache_mode: xlsr_feature
cache:
  root: /root/autodl-tmp/fs_lprf_cache
  schema_version: 1
  feature_dtype: float32
  feature_build_batch_size: 8
  shard_size: 64
  verify_checksums: true
  train_feature_seed: null
  warn_if_estimated_gb_above: 100
runtime:
  train_num_workers: 4
  eval_num_workers: 2
  persistent_train_workers: true
  prefetch_factor: 2
  pin_memory: true
  worker_torch_num_threads: 1
```

这是附加配置示意，不能用来覆盖完整实验 YAML。Train feature seed 为 null 时使用 `train.seed`。`feature_build_batch_size` 独立于训练 `audio_microbatch_size=32`。`runtime.num_workers` 旧字段仍兼容，但会发弃用提示，并作为兼容覆盖值同时作用于 Train/Eval；使用新配置时删除该旧字段，再分别设置两个 Worker 数。

DataLoader Worker 是多进程。训练启动只建一个 Loader，主进程在完整消费当步后更新 Sampler；持续 Worker 跨 Step 存活，结束或异常时关闭。零 Worker 不传预取参数、不启用持续进程。pin memory 可配置，GPU 传输保留 `non_blocking=True`。Feature Cache 每条记录很大，优先实测较小的预取值（如 1）和 Worker 数，不要凭默认值假定 RAM 足够。

Worker 线程上限在子进程启动前通过 OMP/MKL/OpenBLAS 环境传入，初始化函数再调用 `torch.set_num_threads`；仅等到初始化函数才限制，无法约束 Windows spawn 的导入阶段。父进程环境在启动后或异常时立即恢复，不需要用户全局设置。OMP/MKL 变量语义见 [PyTorch 官方线程环境说明](https://docs.pytorch.org/docs/stable/threading_environment_variables.html)，实际兼容性在项目 2.7.1 环境中测试。

## 构建、估算与检查

以下示例沿用 `configs/speechfake.autodl.yaml`，环境先按 [AUTODL.md](AUTODL.md) 检查并复用 base 中已满足的 PyTorch。

```bash
python scripts/check_data.py --config configs/speechfake.autodl.yaml --splits train dev
python scripts/check_model.py --config configs/speechfake.autodl.yaml

python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split train --mode waveform --dry-run
python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split train --mode waveform --limit 1000
python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split train --mode waveform --limit 1000 --check

python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split dev --mode xlsr_feature --validation-modes light,full --limit 100 --seeds 42 --dry-run
python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split dev --mode xlsr_feature --validation-modes light,full --limit 100 --seeds 42
```

`--limit` 总是标记 smoke/incomplete，身份与完整缓存不同，禁止正式训练/评估使用。少量样本检查通过后，去掉 `--limit` 与 `--seeds` 构建完整缓存：

```bash
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode waveform
python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split dev --mode xlsr_feature --validation-modes light,full --dry-run
python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split dev --mode xlsr_feature --validation-modes light,full
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode waveform --check
python scripts/build_cache.py --config configs/speechfake.autodl.yaml \
  --split dev --mode xlsr_feature --validation-modes light,full --check
```

Waveform 估算是重采样后、trim 前的上界；Feature 估算用现有前端的一条合成固定长度输入测得 T，不生成正式特征。估算不含 JSON/文件系统开销。超过阈值会警告，正式生成需交互确认或显式 `--confirm-large-cache`。构建打印进度、吞吐量和 ETA，结束报告大小、样本数、dtype、fingerprint、生成时间并校验所有记录。一次性准备/模型加载开销计入首个新缓存的构建时间；重用已有缓存不会伪造一次新的构建时间。

中断使用相同配置与参数加 `--resume`。只续用身份和 SHA 匹配的已完成分片；未完成临时记录不视为可用。`--force` 显式重新生成时保留原完整缓存及原临时目录为 `.preserved-*`，不删除用户旧缓存；保留副本也占磁盘空间。

## 存储与身份校验

存储使用分片二进制 + JSON 索引与 metadata，按需 memmap 读取；不为每条音频创建 `.pt`，不把整个 Train/特征 split 载入内存或 GPU。Feature 生成按独立小批次提取，并逐层复制到 CPU，避免额外构建整块 GPU 堆叠副本。分片内每条记录存 uttid、字节 offset、shape、length、dtype。

身份包含 schema、split/mode/validation mode/seed、完整有序 uttid、两份数据 Manifest SHA、元数据指纹、源音频内容 SHA、基础预处理及实现 SHA、RNG schema。Train feature 还绑定 View seed、RIR 内容及原 RawBoost 参数/实现；Feature 必须绑定实际 XLS-R 文件 SHA 和完整 25 层顺序。Waveform 不绑定前端权重。metadata 保存实际 shape、分片长度/记录数/SHA、完成标记与构建时间。

只有校验完成后才原子发布缓存目录；OS 文件锁阻止同一目标的并发构建。恢复快照以单文件原子写入，避免索引/metadata 写入间中断破坏已完成分片的续建。读取校验身份、索引 SHA、顺序、去重、offset、dtype、shape、文件长度与配置要求的分片 SHA，记录读取再次拒绝 NaN/Inf。错误一律终止，不回退 WAV 或混用缓存。

源音频与分片内容哈希在同一进程内按文件状态复用，避免逐个基准场景重复读盘；状态变化会重新计算。Dataset 的审计 memo 不随 Dataset 序列化到 Worker。新进程的首次完整内容审计需要读取源文件，属于一次性准备成本，不是训练 Step 时间。身份 fingerprint 与分片内容 fingerprint 都进入训练 Run Identity，`cache_bindings.json` 和 `cache_metadata/*.json` 保存到训练输出供追溯。

## RNG 与恢复

训练样本种子以 SHA-256 派生，包含 RNG schema、全局 Train seed、Step、展平位置和 uttid。同一音频在不同位置或 Step 获得不同种子。样本处理前保存 Python/NumPy/CPU Torch 状态，裁剪/增强后恢复；Worker 不操作 CUDA RNG。零 Worker 同样隔离主进程状态，EpisodeSampler 继续使用原主进程 `random` 状态。

Dev/Test View 种子包含 split、validation mode、seed、uttid 和 RNG schema，继续调用原裁剪函数。不同 Worker 数不会改变同一 View；Light/Full 是不同身份。建立的是新的固定裁剪视图，不保证与旧运行逐条裁剪位置相同，因此正式实验必须用新目录。

Run Identity 纳入缓存内容、schema、模式、训练数据与两个 Dev SHA，排除设备 ordinal、Worker 数、持续进程、预取、pin memory 和 Worker Torch 线程这些性能设置。0/2/4/8 Worker 一致性通过后，允许恢复时调整这些性能项。仍保留原 Checkpoint 的主进程 RNG、Optimizer/Scheduler 与 Best 历史恢复；缓存内容或 RNG schema 改变时拒绝恢复。旧版输入管线的 Run Identity 不可混入新运行。

## AutoDL 性能对照

完整三种模式对照需要分别具备完整 Waveform 和 Train Feature Cache。Train feature 的小样本检查：

```bash
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode xlsr_feature --limit 1000
# 检查后再去掉 limit 构建完整可用于基准的特征缓存
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode xlsr_feature --dry-run
python scripts/build_cache.py --config configs/speechfake.autodl.yaml --split train --mode xlsr_feature

python scripts/benchmark_pipeline.py --config configs/speechfake.autodl.yaml \
  --modes none,waveform,xlsr_feature --workers 0,2,4,8 --warmup 5 --steps 20
# 显式进行真实 Dev 计时；可耗时较长，只在 AutoDL 决定后运行
python scripts/benchmark_pipeline.py --config configs/speechfake.autodl.yaml --dev-only
# 也可一次完成训练管线与 Dev 计时
python scripts/benchmark_pipeline.py --config configs/speechfake.autodl.yaml --include-dev
```

基准使用临时模型/Optimizer 状态、相同预生成 Episode 和相同后端初值，每 Step 保持 8 Episode / 240 音频、相同微批次。结果写入独立 UUID 目录的 JSON/CSV，不写正式 Checkpoint、TensorBoard 或训练日志。GPU 阶段以 CUDA events 计时，Step 墙钟前后同步；CPU 数据等待单独计时。记录构建时间/空间、samples/s、mean/P50/P90 Step、前端、Attention/Bottleneck、反向更新、显存、dtype/fingerprint 及 Worker 配置。AutoDL Linux 还采样主进程+Worker RSS 与 CPU；RSS 为各进程求和，含共享页，属于观测值而非独占内存；Windows 未采集项明确为 null。

Dev 单列 Light 缓存生成、Full 五种子缓存生成、缓存后两档耗时及总空间。缺少真实 Dev 计时时总耗时为 null，不代入外部“约三小时”。完整结果计算 `T_cache + 4000*T_step + 15*T_light + 5*T_full`；Test 不计入训练期间的重复收益，单独记录首次缓存及最终评测。可通过 `--dev-timings <先前 benchmark.json>` 复用同条件测量。

最终按总实验时间、空间、指标与稳定性选择，默认保持 Waveform。若决定 Train feature 对照，显式设置 `data.train_cache_mode=xlsr_feature`，使用独立新 `output_dir`；Baseline 设置 Train/Dev 为 `none`，同样用独立目录。不同方案不能混合续训。

## 正式运行和最终 Test

```bash
python scripts/train.py --config configs/speechfake.autodl.yaml
# 全部训练与 Dev 选模完成后，才检查或构建 Test 缓存
python scripts/evaluate.py --config configs/speechfake.autodl.yaml --checkpoint best --shot 10 --cache-dry-run
python scripts/evaluate.py --config configs/speechfake.autodl.yaml --checkpoint best --shot 10
```

Test 空间超过阈值时，在确认空间后加 `--confirm-large-cache`。普通 `build_cache.py` 没有 `--split test`。Test 缓存生成只运行冻结前端，不计算或打印指标；随后最终评估才计算原报告，`test_run.json` 将首次缓存耗时/空间与评估结果分开记录。`--checkpoint` 使用显式路径时，Feature Cache 模式要求它与已验证的 Best 索引一致。

本机验收与未验证事项见 [CACHE_IMPLEMENTATION_REPORT.md](CACHE_IMPLEMENTATION_REPORT.md)。真实数据吞吐量、完整 Dev 空间/指标及 AutoDL 总实验收益均需实际测量。

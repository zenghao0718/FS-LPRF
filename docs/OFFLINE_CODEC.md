# 离线 Codec Cache：增量构建与并行编码

本机只修改、轻量测试和提交代码；以下 AutoDL 命令仅供后续交接，本次不部署、不做实际大规模构建或 GPU 性能实验。

## 研究方法保持不变

唯一音频输入为完整 Train Waveform Cache；不读已删除 Train WAV，不重建原 Train/Dev 缓存。每条变长波形按 `build_seed + uttid` 固定抽一次 Codec 家族/预设，真实编解码后恢复 16 kHz、单声道、FP32、原始长度。四家族及既有码率不变，训练仍以 50% 概率读原版/固定编码版，再走 4 秒随机裁剪、RIR、RawBoost。XLS-R、Episode、Prototype、α、损失和 Train/Dev/Test 协议不变。

EnCodec 保留官方 24 kHz 单声道预训练模型，检查实际 `segment is None`，整条待编码波形（含短语音重复和保护尾）一次 encode/decode，不自行分段或修改 overlap-add。五带宽仍为 `[1.5,3,6,12,24]` kbps；环境元数据读取真实 segment/overlap，`overlap_active=false`。整段长语音仍有内存与速度风险，不能承诺绝不 OOM。

## 严重削波自动回退

`CodecClippingError` 仅表示原有规则 `peak > 2 OR clipping_fraction > 0.01` 触发。阈值、延迟补偿、`_fit_length()`、归一化和正常成功输出不变。离线串行、CPU Worker 和专用 CUDA Worker 调用同一个 `encode_cached_sample()`，只捕获这个类型。原始/解码 NaN/Inf、长度或延迟补偿错误、初始化/编码/解码故障、OOM 等继续失败，不能通过含有相同文字的普通 RuntimeError 触发回退。

削波时丢弃异常解码结果，保存该 Worker 已读取的原始 Waveform Cache FP32 单声道 PCM 的副本，长度和每个采样值保持一致；不重读 Train WAV，不额外归一化。记录仍占该 uttid 的唯一索引位置，`status="fallback_original"`，保留分配的 `family/variant/param/parameters`，以及 `fallback_reason="severe_codec_clipping"`、异常类型/消息、`decoded_peak`、`clipping_fraction`。只对回退样本发警告。正常记录仍为 `status="ok"`。

Reader 保留 shape/dtype/length/offset/顺序/身份检查，并验证回退原因、原始异常证据和 `choose_spec()` 分配一致；任意其他状态都拒绝。训练直接读取缓存 PCM，回退样本在原始和 Codec 两个 50% 分支均为原音频，这是预期行为。全量完成仍必须覆盖全部 **682,205** 条；本机合成测试不代表真实全量已完成。

`build_report.json` 的 `codec_results`（或构建完成后 `--verify` 的同名字段）从索引生成：`total_samples`、`codec_success_count`、`fallback_original_count`、`fallback_fraction`、`fallback_by_family`、`fallback_details`。明细含 uttid、原分配参数、原因和异常峰值/比例，不维护黑名单；真实 Codec 成功数不包括回退数。预检记录也保留状态，但不计入正式样本总数。

## 已有 322 个 v2 分片：受控兼容续建

仅允许精确旧实现 `5f538056ec070b2e1ad6938bcf9a21af5de4ffa0` 到本次严重削波修复的转换。`configs/codec_clipping_recovery_v1.json` 固定旧/新实现文件摘要，是发布版本范围，不是样本黑名单；其他版本没有通用绕过开关。保留 `BUILDER_VERSION=1`、磁盘格式 v2 和 choose_spec。

`scripts/recover_codec_cache.py` 在与 Writer 相同的独占锁下验证源缓存历史身份、完整有序 ID、seed/Codec 参数、设备/线程、精确旧实现，以及 progress 指定的连续 commit、offset/shape/大小/数量和链式摘要。它只读 JSON 与文件大小，不打开或重哈希 PCM。初次授权要求精确边界 **322 分片 / 164,864 条 / 682,205 总条数**，且旧前缀全部为 `ok`。若实际值不同，立即停止交接并核对，不修改预期参数来强行通过。

授权只原子添加 `clipping_compatibility.json`，记录原/新代码与 Git 身份、源与参数绑定、原始身份指纹、迁移边界和摘要。原 `identity.json`、`progress.json`、`commit_*.json` 和 `.bin` 在授权阶段不变。正常 `--resume` 随后保留旧 identity 与原链锚点，从第 164,865 条继续；已提交 322 个 `.bin` / commit 不重写，正常进度只追加。最终 Reader 继续验证审计边界，训练运行身份绑定审计摘要。

若授权发布前中断，没有有效审计，重跑即可；原子发布后中断，重跑返回 `already_authorized`，不覆盖审计。续建再次中断后也可重复授权检查，最初边界不向后漂移。未提交临时文件仍按原 v2 提交点规则重算。后端环境、设备/线程或源身份不同仍拒绝，不能静默混续建。

以下是后续 AutoDL Cursor 的交接命令，**本次本机不执行**。应先停止旧构建进程，使用原环境与同一配置文件/数据路径；推荐继续 **CPU 4 + 单个 CUDA EnCodec、512 条分片、seed 42**。`--cache-dir` 传最终目录，不带 `.building`：

```bash
git pull --ff-only

# 先只检查；不初始化 Codec、不写授权
python scripts/recover_codec_cache.py --config configs/speechfake.yaml \
  --cache-dir /root/autodl-tmp/fs_lprf_codec_cache \
  --expected-shards 322 --expected-records 164864 --expected-total 682205 \
  --encodec-device cuda:0 --worker-threads 1 --dry-run

# 同样检查后，原子写入一次兼容审计
python scripts/recover_codec_cache.py --config configs/speechfake.yaml \
  --cache-dir /root/autodl-tmp/fs_lprf_codec_cache \
  --expected-shards 322 --expected-records 164864 --expected-total 682205 \
  --encodec-device cuda:0 --worker-threads 1

# 保持原配置与固定分配；不要加 --limit、不要换成新的目录
python scripts/build_codec_cache.py --config configs/speechfake.yaml \
  --output-dir /root/autodl-tmp/fs_lprf_codec_cache --resume \
  --codec-workers 4 --worker-threads 1 --codec-shard-size 512 \
  --encodec-device cuda:0 --max-inflight 10

# 全部完成后快速检查并输出成功/回退统计，无 PCM 全量扫描
python scripts/build_codec_cache.py --config configs/speechfake.yaml \
  --output-dir /root/autodl-tmp/fs_lprf_codec_cache --verify
```

授权输出中的 `audit.boundary` 应为上述分片数/记录数，并保存原 `index_fingerprint` 和字节数；`--resume` 每次都会重算小型提交记录链并核对该边界及旧分片尺寸。最终目录的 `identity.json` 仍为旧 5f53805 身份，`clipping_compatibility.json` 记录本次目标身份；`shard_00000.bin` 至 `shard_00321.bin` 和对应 commit 保留，最终 report 的 `compatibility_boundary` 保留原边界。检查 `codec_success_count + fallback_original_count == total_samples == 682205`。这些检查不需要计算旧 PCM 的 SHA。当前未访问 AutoDL，未迁移真实 322 分片，未验证真实失败语音或 GPU 性能。

## 默认取消大文件扫描

`configs/speechfake.yaml` 和缺省配置均为 `cache.verify_checksums=false`。正常读取、构建、训练和 Resume 不计算 PCM 分片或完整 index 文件的 SHA-256，不扫描全部波形。旧 Train/Dev 缓存中已有的 SHA/fingerprint 直接作为历史身份复用，无需重建原缓存。

保留文件存在/字节大小、uttid 覆盖与顺序、dtype、shape、offset、记录数、完成状态、配置和源缓存身份检查。写入前逐条检查原长与有限值，训练按需读到某条语音时检查其有限值。完成构建只做结构检查，不再多次读取全部波形。

旧格式可显式用 `--set cache.verify_checksums=true` 核对已经记录的哈希；所有正常入口遵循该开关，没有强制 True。新 Codec v2 不记录 PCM/完整索引文件 SHA，该选项不会为 v2 补扫。代码、清单和小型元数据摘要保留。默认快速检查无法发现同尺寸 PCM 的任意逐字节修改，这是取消内容扫描后的校验范围。

## Codec 专用磁盘格式 v2

只给 Codec 引入独立格式，原训练/验证缓存系统不作整体重构。`BUILDER_VERSION=1` 和 choose_spec 不变；磁盘采用独立 `format_version=2`，公共 identity 的 schema_version 仍为 1。

- `identity.json`：不变身份及完整 ID 清单，只保存一次。
- `shard_NNNNN.bin`：PCM 临时文件写完、fsync 后原子改名，不重读计算 SHA。
- `commit_NNNNN.json`：仅该分片的描述与新增索引记录，原子发布。
- `progress.json`：分片数、记录数、已提交字节数、累计时间、增量摘要等标量，原子更新，是提交点。
- `index.json` / `metadata.json`：全部完成后统一生成一次，再做结构检查和目录原子发布。metadata 引用 identity.json，不再重复整个身份。

每次 flush 不重写历史完整索引、身份或恢复快照。增量 `index_fingerprint` 是小型分片记录在内存中的链式摘要，不是扫描大型索引/PCM 文件的哈希；`content_fingerprint` 绑定身份与索引/分片结构，不宣称逐字节校验 PCM。

Resume 仅重放 progress 指定的连续提交，确认其等于原 Manifest 前缀。提交点以外的临时/已改名 PCM 或 commit 不算完成，仅重算未提交部分；缺失、重复、顺序错乱、大小/范围错误直接报错。写锁保持 OS 进程锁。空间预估只读轻量 progress，实际恢复验证一次增量记录；末尾完成发布中断也可恢复。不覆盖旧缓存。

兼容旧 Train/Dev v1 读取，复用历史指纹。已发布 `d775aa5` 的官方整段 CPU Codec v1 按精确实现身份兼容读取，不重新算旧 SHA；更早的 10 秒分段 EnCodec 缓存仍拒绝。旧 `.building/resume.json` 不迁移到新格式，保留并选新目录。新格式 Resume 允许改 CPU Worker 数、任务窗口与分片大小；设备模式、编码线程、seed、Codec 参数和源身份必须一致。

## CPU 多进程与可选 EnCodec GPU

主进程只发送 uttid、spec 和 seed，Worker 自行只读源缓存、执行真实编解码，每进程初始化一次后端。Worker 可以乱序完成，主进程按原 Manifest 顺序写入；未完成任务和乱序结果 Future 同受 `--max-inflight` 限制，另有当前写入的一条波形，不会生成全量 Future。

CPU Worker 数支持 1/2/4/8 等正整数。采用 spawn，默认每 Worker 的 PyTorch/OpenMP/ffmpeg 线程为 1，避免过度竞争。`cache.codec_shard_size=512` 独立于原 `cache.shard_size=64`，可在 CLI 选 1024 等值，不影响固定分配。

`--encodec-device cuda:0` 增加一个专用 spawn EnCodec Worker，CPU Worker 仅处理 MP3/Opus/AMR，不在同一 GPU 重复加载模型。模型与完整输入放到指定 CUDA；解码结果返回 CPU FP32，并恢复 16 kHz/原长。不引入混合精度、分段替代或静默 CPU 回退。设备模式写入构建身份，防止 CPU/GPU 混续建。

没有 CUDA 的环境使用 `--encodec-device cpu`。本机只测试 CPU 合成任务、GPU 路由/设备请求 mock；这不是实际 EnCodec 或 GPU 吞吐、显存与长音频验证。

## 后续 AutoDL 命令（本次不执行）

先在仓库根目录激活原环境并拉取交付 commit，检查剩余空间和四家族依赖。以下均限制 2000 条，使用独立目录，保留已有缓存和结果：

```bash
git pull --ff-only
df -h /root/autodl-tmp

# 一个编码 Worker + 主进程写盘
python scripts/build_codec_cache.py --limit 2000 \
  --output-dir /root/autodl-tmp/codec_v2_smoke_cpu_w1 \
  --codec-workers 1 --worker-threads 1 --codec-shard-size 512 --encodec-device cpu

# CPU 多进程；也可选 2/4
python scripts/build_codec_cache.py --limit 2000 \
  --output-dir /root/autodl-tmp/codec_v2_smoke_cpu_w8 \
  --codec-workers 8 --worker-threads 1 --codec-shard-size 1024 --max-inflight 16 --encodec-device cpu

# 8 个其他 Codec CPU Worker + 1 个 EnCodec GPU Worker
python scripts/build_codec_cache.py --limit 2000 \
  --output-dir /root/autodl-tmp/codec_v2_smoke_cuda_w8 \
  --codec-workers 8 --worker-threads 1 --codec-shard-size 1024 --max-inflight 18 --encodec-device cuda:0

# 仅中断后执行：保持相同设备、线程、seed、Codec 参数及源身份
python scripts/build_codec_cache.py --limit 2000 --resume \
  --output-dir /root/autodl-tmp/codec_v2_smoke_cuda_w8 \
  --codec-workers 8 --worker-threads 1 --codec-shard-size 1024 --max-inflight 18 --encodec-device cuda:0

# 快速结构验证；不初始化编码器、不全量扫描 PCM
python scripts/build_codec_cache.py --output-dir /root/autodl-tmp/codec_v2_smoke_cuda_w8 --verify
```

任一构建命令追加 `--dry-run` 做空间与真实后端/预设的小样本预检，不生成缓存。全量构建需以后在 AutoDL 独立授权后去掉 --limit；不在本次本机任务执行。--limit 的 smoke 标记仍禁止用于完整训练。

后续真实 EnCodec 验收先对原失败样本 `47954871497225c828cc7fdf41e5ddf3f405c382bb275347040f16a714433188` 强制五带宽 apply_array 回归（不要依赖其离线固定家族，seed42 下该 ID 固定选 AMR）。再检查短/静音、9.9/10/19.8 秒附近、非 320 整数倍长度和真实最长 Train 完整波形，记录耗时/内存。真实边界单测：`python -m pytest -q tests/test_codec_real_boundaries.py -k real_encodec_official_whole_wave`。缺依赖明确 skip，编码/权重/参数失败明确 fail，不跳过样本或改第三方算法。

如获后续测速授权，使用同一新 smoke 覆盖范围比较三模式 × workers=4/8，保持 5 预热 + 30 计时 Step：

```bash
python scripts/benchmark_pipeline.py \
  --modes waveform,waveform_codec,waveform_codec_offline \
  --workers 4,8 --warmup 5 --steps 30 \
  --smoke-codec-cache /root/autodl-tmp/codec_v2_smoke_cuda_w8 \
  --output /root/autodl-tmp/codec_v2_pipeline_benchmark
```

这不是本机执行命令；旧两组 Baseline 不与新版混成同一次公平实测。开发 Full Dev `[42]` / 官方五种子仍可隔离使用，Test 与 ASVspoof5 协议保持原样。

## 性能统计口径

- worker_initialization_seconds：进程启动、后端加载和预设探测。
- preflight_seconds：源缓存最短/中位/最长小样本实际编解码预检。
- stats.build_seconds：累计已提交构建时间及最后结构检查，包含编码等待、IPC 与写盘，不含上述启动/预检。中断前未提交部分的耗时可能未进入累计值。
- invocation_seconds：本次 build 调用墙钟时间；Resume 时与累计 build_seconds 区分。
- index.codec.seconds：单条 Worker 读取和编解码时间，不含主进程排队和写盘；并行时不能简单累加作为墙钟时间。
- stats.bytes：实际 PCM 分片字节；estimated_full_pcm_bytes 依据当前源索引长度求和；空间预留默认 5 GiB，另计索引/分片余量，Resume 扣除已提交 PCM。
- peak_inflight / max_inflight：实际峰值 Future 窗口/上限；报告同时记录 Worker 数、线程、设备、分片大小及 spawn 启动方式。
- projected_full_build_seconds：只按本次小样本累计速度线性外推。Resume 更改过 Worker 数时属于混合运行速度，不能标成单一配置实测。

此前约 42 小时仅为旧串行实现的历史推算，不作为新版构建时间。新版 CPU/GPU 速度、显存峰值、内存和长音频兼容性须以后在 AutoDL 实测；本机不声称完成这些验证。

## 前次性能优化验收（2026-11-10）

基线为 `d775aa5b0b8459a365bca0cfeb92252f53bead7f`。完整 `python -m pytest -q`：**474 passed, 44 skipped, 0 failed**，432.88 秒；`git diff --check` 通过。

合成小缓存测试覆盖普通读取/构建/Resume 不哈希 PCM 或完整 index、每分片只写新增记录、首次身份/进度及分片/最终发布中断、损坏增量索引拒绝、NaN/Inf/错误长度拒绝、1/2/4/8 个真实 CPU 子进程的固定分配与顺序、跨 Worker 数续建、CPU/GPU 身份切换拒绝，以及旧 Waveform/官方整段 CPU Codec v1 的兼容读取。GPU 任务路由和模型/输入设备请求仅用 mock 验证。

44 个跳过项涉及缺少真实 Codec 依赖或语音 fixture，不计为真实 EnCodec/GPU 通过。未在本机执行真实大规模数据处理、长时测速、AutoDL 任务、正式训练或完整 Dev/Test，也未删除或覆盖已有用户缓存、权重与实验输出。

## 本次严重削波回退与恢复验收（2026-11-10）

本次基线为 `5f538056ec070b2e1ad6938bcf9a21af5de4ffa0`。完整 `.venv/Scripts/python.exe -m pytest -q --tb=short`：**509 passed、44 skipped、0 failed**，492.45 秒；`git diff --check` 通过。

新增 35 个合成测试覆盖与精确旧版成功输出/记录对照、2.782/.000399 异常的逐样本原音回退、单进程/真实 CPU spawn 多进程一致性、全 ID 顺序与统计、非法回退记录拒绝、其他异常不回退、失败后从有效提交点继续、精确旧 v2 身份授权、授权前后中断、授权后新增回退分片再中断、重跑幂等、原分片/commit/identity 字节保持、恢复不读/哈希 PCM，以及恢复 CLI。CPU/CUDA 共同入口仅作 mock 验证；真实 GPU 性能、真实失败 Opus 样本及 AutoDL 322 分片尚未验证或迁移。此前旧 Waveform/Codec 缓存、乱序 Worker 和研究协议的原有测试继续通过。

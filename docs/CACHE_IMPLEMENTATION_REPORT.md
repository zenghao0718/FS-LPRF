# 数据缓存与 DataLoader 优化实施报告

日期：2026-11-05。基线为 `5023b6efb3d1e3aeb99d261cf590ca3ee96325ff`，实施依据为用户提供的《FS-LPRF 数据缓存与 DataLoader 加速 Codex 实施规范 v1.0》。本机只运行合成数据测试与极小真实权重 Smoke，没有连接 AutoDL，没有真实训练、完整真实 Dev/Test 或 0-shot 构建。

## 基线差异与保留范围

- 当前 main 基线为 **149 passed**，不是更早布局重构时的 145；本轮全部保留。
- 当前 `train.audio_microbatch_size` 已为 **32**，README/AutoDL 仍写旧值 8；本轮同步文档，未改变该配置或 Episode 预算。
- 规范引用方法 v2.5；仓库现有 Checkpoint 的方法/实现标签仍为 2.4/1.3。本轮不迁移这些标签，不改后端参数键或 Optimizer 状态结构，缓存 schema 与新输入管线 Run Identity 单独版本化。
- 通用音频读取的 `st/dur` 分支实际包含随机文件片段读取，基础预处理抽取时原样保留；完整 Waveform Cache 拒绝这类记录，避免冻结该随机分支。当前 SpeechFake 的标准五字段 Manifest 不含它们。
- XLS-R 架构/权重、Attention/Bottleneck 初始化与原前向、RawBoost/RIR 实现、Episode/原型/α/Loss、验证划分、指标、Full-only Best、保存节点及恢复事务保持原有计算。

## 模式与实现

`none`、Train `waveform`、Train/Dev/Test `xlsr_feature` 均已实现。默认 Train waveform，Dev/Test feature，FP32。Waveform 保存完整预处理变长 `[1,L]`，不提前裁剪或增强；Train 每次仍在线调用原裁剪及增强。Train feature 每条只保留一个固定 View，属于牺牲随机性换速度的对照。

前端缓存完整 `[25,T,1024]`，第 0 组为 post_features，随后是原 24 个 layer_results 的顺序；T 由实际输出确定。读取后仍调用原 tnd_forward 与共享 Linear，Attention/Bottleneck 有梯度，XLS-R 无梯度且保持 eval。Checkpoint 可训练参数仍为 **364800**，只有原 3 个后端参数键。

训练启动前检查完整 Train 和六个 Dev 缓存身份，缺失/失效时报显式构建命令，不静默生成、不回退 WAV。训练只初始化 Train/Dev；Test 只能在最终 evaluate.py 确认并加载已选 Best 后规划/创建/读取。缓存生成不计算 Test 指标。最终 Test 元数据将缓存阶段与评估耗时分开记录；首次建缓存仍执行 XLS-R，不宣称首次 Test 必然更快。

训练 Loader 在真实训练入口创建一次，动态 Sampler 每步接收展平索引与逐样本种子，完整消费后更新；持续进程跨 Step 存活，异常/结束时关闭。0/2/4/8 Worker 的合成输入逐位一致，进程 PID 复用和退出均测试通过。样本种子绑定全局 seed、Step、位置、uttid、RNG schema，局部保存/恢复 Python/NumPy/CPU Torch 状态，不消耗 EpisodeSampler RNG。Dev/Test 种子额外绑定 split/mode，保持 preserved_rng。

缓存使用按需 memmap 分片与索引，绑定数据 Manifest SHA、实际音频内容、预处理、seed、前端权重（仅 feature）、增强参数/资源（仅 Train feature）、层序、dtype 与 schema。每个 shard 校验大小/记录数/SHA，索引校验顺序/去重/offset/shape/dtype，记录拒绝 NaN/Inf。构建使用 OS 锁、临时目录、原子恢复快照及发布；force 保留旧副本。输入身份和实际内容 fingerprint 写入 Run Identity 与输出元数据，纯 Worker/预取/pin memory/设备 ordinal 不影响科学身份。旧输入管线的 Run 不混合续训。

## 修改文件与职责

- `.gitignore`：排除默认 `/cache/`，保留原资源与输出规则。
- `configs/speechfake.yaml`：增加三种模式、缓存默认项和 Train/Eval Worker 设置；既有模型/实验超参数不变。
- `utils/config.py`：旧配置补默认值、新字段校验、旧 Worker 字段兼容及项目内缓存目录边界。
- `utils/numerics.py`：最大/平均绝对误差、相对 L2、余弦相似度诊断。
- `data/audio.py`：仅拆出原基础预处理读取入口，裁剪函数与原处理语义保留。
- `data/randomness.py`：版本化 SHA-256 逐样本种子与 CPU RNG 隔离。
- `data/cache.py`：身份、文件哈希、分片读写、memmap、完整性、锁、续建与原子发布。
- `data/cache_pipeline.py`：Train/Dev 缓存准备/绑定、固定 View、估算、运行追溯与最终 Test 缓存准备。
- `data/speechfake_dataset.py`：三种输入模式、严格波形/特征形状、缓存 collate 与 Worker 序列化。
- `data/loaders.py`：可更新 Sampler、可复用 Loader、独立 Worker/预取/pin memory/线程配置；线程限制覆盖子进程导入阶段，父环境自动恢复。
- `models/fs_lprf_model.py`：复用原 extract_features 提取完整前端状态，增加缓存后端入口和共享输入分派；不新增模型参数。
- `engine/trainer.py`：预检/绑定缓存、写追溯元数据、跨 Step 复用 Loader；原 Episode Loss/一次 backward/一次 update 保留。
- `engine/evaluator.py`：统一 waveform/feature 分批入口，保留 Real Support/Query 共享、指标与报告。
- `engine/validation_schedule.py`：缓存内容/RNG 纳入新 Run Identity，排除已验证的纯性能项；Best 逻辑不变。
- `engine/pipeline_benchmark.py`：相同 Episode/初值的隔离基准、CUDA 分段计时、Linux RAM/CPU 观测、Dev 条件身份及总时间投影。
- `methods/prototype_bank.py`：拒绝训练增强 Feature View 构建无增强原型库；原配额与计算不变。
- `scripts/build_cache.py`：显式 Train/Dev 构建、dry-run/limit/resume/force/check、空间确认、进度与统计。
- `scripts/benchmark_pipeline.py`：三模式 × Worker 基准，独立 JSON/CSV，可显式测 Dev；不碰 Test 或正式 Checkpoint。
- `scripts/check_data.py`：保留全部原检查，增加显式选择 Train/Dev 的入口，默认仍检查三 split。
- `scripts/check_model.py`：现有单 WAV Smoke 增加真实前端 FP32/FP16 缓存后端误差。
- `scripts/evaluate.py`：Best 后最终 Test 缓存、空间规划、阈值确认和缓存/评估独立耗时。
- `scripts/build_prototype_bank.py`：Best 后使用未增强原始 Train，禁止混用训练固定增强 View。
- `tests/conftest.py`：原有单元测试显式使用 none/零 Worker；新测试另行检查正式默认配置。
- `tests/test_cache_storage.py`：19 项，波形等价、边界、完整性、损坏拒绝、续建、保留旧缓存与身份。
- `tests/test_cached_features.py`：9 项，25 层顺序、FP32/FP16、梯度/参数键、单 View 与库构建限制。
- `tests/test_pipeline_workers.py`：10 项，0/2/4/8 一致性、进程复用/退出、RNG/恢复、配置、运行身份与启动环境恢复。
- `tests/test_cache_integration.py`：5 项，缓存评估分数等价与 Real 单次读取、训练更新、Test 隔离和基准。
- `tests/test_cache_tools.py`：15 项，工具实际流程、空间估算/确认、配置拒绝、Train/Dev 数据检查、基准条件身份和跨场景校验和复用。
- `README.md`：默认缓存模式、入口、随机数及当前微批次说明。
- `docs/AUTODL.md`：复用环境、保留资源路径、使用新输出目录、显式缓存与正式运行流程。
- `docs/VALIDATION_SCHEDULE.md`：补充缓存输入/身份与恢复约束，不改验证日程。
- `docs/DATA_CACHE.md`：完整缓存、检查、Worker、恢复、基准、Baseline/两方案及最终 Test 命令。
- `docs/LOCAL_VALIDATION.md`：追加本轮验收范围与结果。
- `docs/CACHE_IMPLEMENTATION_REPORT.md`：本实施报告。

## 本机测试结果

沿用已有 `.venv`：Python **3.11.18**，torch/torchaudio **2.7.1+cu128**，CUDA runtime **12.8**；未创建环境、重复安装 PyTorch 或新增第三方依赖。

- 改造前：`python -m pytest -q` → **149 passed，78 warnings，52.43 秒**。
- 最终完整回归：`python -m pytest -q --junitxml=outputs/cache_validation/pytest.xml` → **207 passed、0 failed、0 skipped，78 warnings，74.68 秒**。新增 **58** 项，全部原有用例保留。告警来自既有 librosa、RawBoost NumPy 转换与 weight_norm。
- `python -m pip check` → **No broken requirements found.**
- 原六个 + 新两个 CLI `--help` → **8/8 正常退出**，子进程移除自定义 PYTHONPATH。
- 合成五种子/32 Fake 类评估的在线/缓存预测 CSV 与指标完全一致，逐 seed 检查所有 Real Support/Query 只读取一次。
- 新增合成测试覆盖缓存训练的 8 Episode / 240 样本 / 一次 optimizer 更新、可训练后端梯度、原后端参数键，原有测试继续验证一次 backward、Checkpoint/RNG/Optimizer/Scheduler/Best 与报告行为。

曾复现 Windows 八 Worker 在数值库并行导入时异常退出；仅在 worker_init_fn 设置线程数时限制生效过晚。将 OMP/MKL/OpenBLAS 上限提前传给启动的子进程、随后恢复父环境后，常规命令下全部 Worker 与完整回归通过；没有跳过测试、减少 Worker 或重试隐藏异常。跨 Worker 测试使用八个微批次，确认每个配置的 Worker 都实际处理样本。

机器记录均留在 Git 忽略的 `outputs/cache_validation/`：`pytest.xml`、`cli_help.json`、`real_model_report.json`、`fp16_synthetic.json`、`source_audit.json` 和导入诊断；没有缓存 tensor、WAV、模型权重或运行输出进入提交。源码 AST/基线对照确认原处理函数、模型初始化/既有前向、受保护计算文件与实验参数不变。

## FP32 / FP16 实测

**合成前端 + 30 条合成音频 Episode**，使用原后端计算实现、FS-LPRF Loss 与 K-shot 分数函数；后端为固定种子随机初始化，不是正式训练模型：

- Hidden States FP16：max abs **0.0009765625**，mean abs **9.5405863e-5**，relative L2 **1.7342679e-4**，cosine **0.999999984962**。
- 后端 FP16：max abs **0.0013298094**，mean abs **3.4227730e-5**，relative L2 **1.8018244e-4**，cosine **0.999999983767**。
- Loss：FP32 **1.1075861454**，FP16 **1.1075887680**，绝对差 **2.6226044e-6**。
- K-shot 分数：max abs **3.8430095e-5**，mean abs **1.5253449e-5**，relative L2 **1.4062669e-4**，cosine **0.999999990134**。

**真实官方 XLS-R 权重 + 单条合成 4 秒 WAV**，RTX 5060，未执行优化器更新：

- 完整前端状态实测 `[1,25,199,1024]`，后端 `[1,25,256]`；这里的 199 是此输入的观测值，不是代码常数。
- FP32 在线/缓存后端 max/mean/relative 误差均 **0**，cosine 约 **1**。
- FP16 Hidden States：max abs **0.1240539551**，mean abs **0.0004885288**，relative L2 **0.0002088276**，cosine **0.999999978201**。
- FP16 后端：max abs **0.0030708313**，mean abs **8.4757073e-5**，relative L2 **8.6016903e-5**，cosine **0.999999996332**。
- 单 WAV 平方均值诊断 Loss：FP32 **9.6719579697**，FP16 **9.6721124649**；这是梯度诊断，不是正式 Episode Loss 或真实任务指标。
- 25 层 Attention 与共享 Bottleneck 均有有限非零梯度；前端冻结/eval。可训练参数 **364800**，后端 Checkpoint **1,507,477 字节**，峰值 CUDA 张量分配 **1,443,674,624 字节**。

FP16 不是无损缓存。上述范围通过不代表真实 SpeechFake 指标等价，正式默认保持 FP32。

## AutoDL 验收与未验证项

完整复制命令见 [DATA_CACHE.md](DATA_CACHE.md) 和 [AUTODL.md](AUTODL.md)：检查 base 环境；少量 `--limit` + `--dry-run`；完整 Train waveform 和六个 Dev feature 缓存；`benchmark_pipeline.py` 比较三模式、Worker 0/2/4/8、5 warmup + 20 timed；显式 `--dev-only` 或 `--include-dev` 测真实两档 Dev；新目录正式训练；Best 后最终 Test。

**未验证**：AutoDL Linux 的真实吞吐量/CPU/RAM、真实数据全量缓存构建时间与空间、完整 Light/Full Dev 指标与速度、完整 4000 Step 稳定性、真实 Test/0-shot。没有将外部“Full Dev 约三小时”写成测量值；缺少 Dev 测量时总耗时输出 null。FP16 对真实数据指标的影响也未验证。

报告总时间使用实测组件投影 `T_cache + 4000*T_step + 15*T_light + 5*T_full`，计入一次性构建，Test 首次费用单列。默认先使用 Waveform；是否切换 Train feature，仍由用户根据 AutoDL 总时间、空间、指标与稳定性决定。

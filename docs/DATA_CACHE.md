# Train/Dev Waveform Cache 与原始 Test

当前正式默认值为 Train=waveform、Dev=waveform、Test=none。历史报告保留，历史默认值不作为当前配置依据。模型、25 层在线提取、Episode、优化器、4000 Steps、200/800 验证、五种子、指标和 Best 均不变。

## 数据流

- Train：原始 WAV → 原基础处理 → 完整 FP32 Waveform Cache → 原动态 64000 点裁剪 → RIR p=1/3 → RawBoost Algorithm 5 p=1/5 → 在线 XLS-R → FS-LPRF。
- Dev：原始 WAV → 同一基础处理 → 完整 FP32 Waveform Cache → 原 mode/seed/uttid 确定性裁剪 → 在线 XLS-R → 原验证；不增强。
- Test：保留原始 WAV → 运行时同一基础处理 → 原确定性裁剪 → 在线 XLS-R → 原指标；不使用任何 Cache。
- 未来 ASVspoof2019 LA、2021 DF、ASVspoof5 Track-1 Evaluation：原始音频在线处理，Phase-2 数据与评分适配另行实现。

基础处理仍调用 data/audio.py 的 get_audio 和 FADTrim.load_preprocessed_waveform；读取/解码、16 kHz 重采样、首声道、静音裁剪、-20 dB RMS/限幅的顺序与参数未改。固定裁剪、增强和 XLS-R 隐藏状态不进入 Cache。

## 配置与构建

唯一配置是 `configs/speechfake.yaml`。缓存模式、Worker 和资源路径都写在这一份文件里。模型、Episode 和优化协议保持不变。

~~~yaml
data:
  train_cache_mode: waveform
  dev_cache_mode: waveform
  test_cache_mode: none
runtime:
  train_num_workers: 4
  eval_num_workers: 2
  persistent_train_workers: true
  persistent_eval_workers: true
  prefetch_factor: 2
  pin_memory: true
  worker_torch_num_threads: 1
~~~

正式配置/构建 CLI 禁止 xlsr_feature；历史低层类型与合成测试保留。Waveform 始终 FP32，不受 feature_dtype 影响。显式 none 只用于源文件仍在时的输入对照，不是缓存错误的回退。

~~~bash
python scripts/build_cache.py --split train --mode waveform --dry-run
python scripts/build_cache.py --split dev --mode waveform --dry-run
python scripts/build_cache.py --split train --mode waveform
python scripts/build_cache.py --split dev --mode waveform
python scripts/build_cache.py --split train --mode waveform --check
python scripts/build_cache.py --split dev --mode waveform --check
~~~

构建/估算需要源 WAV，检查和正式读取不需要。Dev 基础波形只存一份，不按 Light/Full 或 seed 复制；不要传 --seeds。--limit 是 smoke 身份，不能用于正式运行或全量删除依据。--resume 续建，--force 保留旧缓存并占额外空间。

Worker 沿用 eval_num_workers/prefetch/线程配置，只预处理；主进程统一顺序写原 CacheWriter，保留分片、SHA、完整性和续建。CacheReader 按需映射有限分片并返回 Tensor 副本，close()/上下文管理释放当前映射，便于 Windows 临时目录清理。

## 无原始 WAV 的身份验证

构建记录 split、Manifest SHA、全部有序 uttid、标签/元数据指纹、来源 SHA、原预处理参数和 audio.py SHA、Cache/RNG Schema、dtype 与分片校验。完整验证后发布 cache.root/<split>/waveform/references/ 下的绑定引用，只指向已完成缓存。

运行时从当前 Manifest/处理实现生成不读 WAV 的绑定，读取 Metadata 中历史来源 SHA，再验证内容/身份指纹、ID/标签、Schema、索引、shape/dtype/offset、分片大小和配置要求的 SHA；逐条拒绝 NaN/Inf。不调用 source_digest 读取已删除的原始 Train/Dev。

原 Manifest 与路径字符串必须保留，不应把 TSV 改成缓存路径。缓存缺失、损坏、清单或预处理变化直接报错，绝不回退 WAV。Train/Dev Cache、references、Manifest、标签、RIR 和权重均必须保留。

旧 Train Cache 的格式、schema 和 audio.py 未变，匹配来源时无需重做基础预处理；显式构建会复用并发布引用。旧缓存无引用时只接受唯一匹配版本，多版本拒绝猜测。Dev 需新建 Waveform Cache。切换 Dev 缓存模式应使用新 output_dir，不混续旧正式运行；Worker 性能调整仍不改变 Run Identity。

## 只读删除安全审计

~~~bash
python scripts/audit_waveform_sources.py --output-dir outputs/source_audit_run1
~~~

输出 audit.json、deletable_wav.csv、retained_files.csv，拒绝覆盖已有报告。实现不包含删除或移动原始数据的操作。

删除建议的前提是两个完整缓存全量校验、数量/ID/标签一致，当前原始 Train/Dev SHA 与构建历史一致，Test 源文件存在且非空。仅有 smoke/subset Cache 不够。任何校验失败、缺失、来源变化会阻止建议删除。

必须保留全部 Test 源文件、Test 共享的物理文件/别名，六份 Manifest、标签、缓存/引用、RIR、权重。硬链接/符号链接、非 WAV 和特殊 st/dur 随机文件片段逻辑均保留并报告。原完整波形缓存拒绝特殊随机读取，不跳过样本或冻结随机片段。

只输出 Manifest 引用的逐个 WAV，不建议删除整个数据集根目录。空间是唯一文件逻辑大小估计，不保证等于实际文件系统回收量。

## 删除后的检查

~~~bash
python scripts/build_cache.py --split train --mode waveform --check
python scripts/build_cache.py --split dev --mode waveform --check
python scripts/check_data.py --cache-only --splits train dev --read-samples 8
python scripts/check_cache_runtime.py --splits train dev --batches 2 --with-model
python scripts/train.py --config configs/speechfake.yaml
~~~

check_cache_runtime 只检验输入/可选在线前向，无优化器更新、不保存状态、不开 Test。Train 动态增强仍读必须保留的 RIR WAV。原型库脚本也可读无增强 Train Cache，不依赖已删除的 Train WAV；原型、配额与决策未改。

原始音频对照基准、构建及删除前来源审计仍明确需要原始文件；删除后不能做 Raw 对照。check_data 的 --cache-only 明确检查 Train/Dev；普通模式仍用于删除前原始数据检查。

最终 Dev 选定 Best 后：

~~~bash
python scripts/evaluate.py --checkpoint best --shot 10
~~~

Test 不构建/读取/估算缓存，旧 --cache-dry-run/--confirm-large-cache 参数会拒绝。Test 原始 WAV 缺失/解码失败按原逻辑报 split、uttid、path，不替换样本，不跳过基础处理，不增强。

## Loader、RNG 与外部接口

保留训练跨 Step 复用、每种子一个评估 Loader、330 Support 合并并按 ID 恢复、Real Query 共享和 Fake Query 分批。原 train/step/position/uttid 与 view/split/mode/seed/uttid 种子不变。完整消费才允许换索引，提前退出和异常关闭 Worker。内存/显存不因 Query 一次性堆积而增加。

get_audio 不依赖 SpeechFake Manifest/32 类，FADTrim 和通用 Loader 可复用。32 Fake 限制保留在 SpeechFakeDataset。SpeechFake evaluate 仍是其 33 类 K-shot 适配器，外部 ASVspoof 需要自己的数据/评分入口，本次不更改模型推理与 Phase-2 协议。上游非 16k 多声道的 NotImplemented 分支保持原样，不宣称覆盖所有格式。

## 本机验收

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/benchmark_waveform_paths.py --audio-root D:/SpeechFake_Data/Real --limit 512 --rounds 3 --workers 0,2 --build-workers 0,2 --output-dir outputs/waveform_paths_run1
~~~

Real-only 临时清单 Fixture 不创建 Fake 类/正式划分，不加载模型、不报告分类指标；移走验证只操作自有副本。旧 benchmark_audio.py 保留。每组 1 轮预热、2 轮计时；输入吞吐不能等同正式训练吞吐。Windows spawn、实际磁盘、增强和完整模型需在 AutoDL 重测。

## AutoDL 执行顺序

本机确认并另行提交/推送后才可拉取本次代码；本任务不执行 push/AutoDL 操作。
1. 拉取已经确认的代码，保留旧输出。
2. 下载并解压完整 SpeechFake-BD，保持原 split。
3. 配置唯一 YAML，检查 Train/Dev/Test TSV/TXT、标签、类别、容量、路径。
4. 单独检查 Test 原始文件：python scripts/check_data.py --splits test --read-samples 8。
5. 分别 Train/Dev Waveform --dry-run。
6. 用 df -h /root/autodl-tmp 检查 1TB 数据盘实际余量，确保源数据、两个 Cache、Test、资源与构建余量可同时容纳。
7. 构建完整 Train Cache，不带 --limit。
8. 构建完整 Dev Cache，不带 --limit。
9. 分别 --check 全量完整性。
10. 运行只读删除审计并核查 JSON、两份 CSV。
11. 用户确认后手工处理合格的逐个 Train/Dev WAV，不删根目录。
12. 保留全部 Test WAV、Manifest、标签、Cache/references、RIR、权重和运行历史。
13. 执行 cache-only 数据检查及 --with-model 在线前向冒烟。
14. 开始原正式训练；仅 Full Dev 选 Best，最终 Test 仍读原始 WAV。

完整数据、完整指标和长时资源表现均留待正式环境，不能用本机 Real 输入结果代替。

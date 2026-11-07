# 通用外部 10-shot 评估

所有外部数据集只使用 `scripts/evaluate_external.py`。数据集差异由输入配置描述，不需要新的 Dataset、Evaluator 或专用转换脚本。标准 CSV/TSV 可直接输入；其他矩形文本表格可配置分隔符、表头、列、路径和标签。

## 1. 环境、检查点与输出目录

在仓库根目录运行已安装依赖的 Python。真实 XLS-R 前向仍要求 CUDA。显式传入已经确认的旧训练 Best 节点文件；不会从当前默认训练目录隐式查找模型，也不修改 Checkpoint、执行训练或优化器更新。

以下为 Bash 示例。已有 Best 索引时，可使用原有解析函数取得明确节点：

```bash
TRAIN_RUN=/root/autodl-tmp/fs_lprf_runs/speechfake_fs_lprf
CKPT="$(python -c 'import sys; from engine.checkpoint import resolve_checkpoint; print(resolve_checkpoint("best", sys.argv[1]))' "$TRAIN_RUN")"
FRONTEND=/root/autodl-tmp/pretrained/xlsr2_300m.pt
RESULTS=/root/autodl-tmp/fs_lprf_external_results
mkdir -p "$RESULTS"
```

旧目录没有 `checkpoints/best_checkpoint.json` 时，直接把 `CKPT` 设为已确认的旧 Best 文件。外部入口拒绝 `--checkpoint best`，避免加载到错误的默认训练目录。PowerShell 使用同样的 CLI 参数，按本机语法设置变量和换行。

`--output-dir` 必须是尚不存在的新目录，并位于训练/Checkpoint 目录之外。重跑换新目录，已有结果不会覆盖。只在成功结束且 `external_run.json.status=complete` 时，将该目录视为完整评估。

## 2. 完全陌生的数据集：零 Python 修改示例

假设新数据集位于 `/data/new_corpus`，含音频与 `index.table`。下面是清单片段；实际每类必须满足第 5 节的 Support/Query 容量要求：

```text
sequence;file_key;truth;generator;voice
r001;recordings/r001.wav;HUMAN;;speaker_a
f001;recordings/f001.flac;SYNTH;vocoder_x;speaker_b
f002;recordings/f002.wav;SYNTH;vocoder_y;speaker_c
```

只需新建 `/data/new_corpus/mapping.yaml`：

```yaml
format:
  delimiter: ";"
  header: true
columns:
  uttid: sequence
  audio_path: file_key
  label: truth
  attack_id: generator
paths:
  root: "."
  template: "{value}"
labels:
  real: ["HUMAN"]
  fake: ["SYNTH"]
metadata:
  speaker: voice
```

直接运行：

```bash
python scripts/evaluate_external.py \
  --mode labeled --manifest /data/new_corpus/index.table \
  --mapping /data/new_corpus/mapping.yaml --group-by none \
  --checkpoint "$CKPT" --set "model.frontend_path=$FRONTEND" \
  --output-dir "$RESULTS/new_corpus_binary_run1"
```

无需指定数据集名称，也不修改任何 Python。若要分别评估 vocoder_x、vocoder_y，只把 `--group-by none` 改为 `--group-by attack_id`。另一套数据即使改用数字标签、其他列顺序或音频 ID，也只需修改 YAML 或准备标准清单。

## 3. 字段、格式与路径配置

内部记录使用 `uttid`、`audio_path`、`label`、`raw_label`、`attack_id` 和 `metadata`。原始全部列保存在 metadata；显式 metadata 映射可以给列起统一别名。规范化记录的 metadata 写入 CSV 时序列化为 JSON，便于复现。

配置选项：

- `format.delimiter`：一个分隔字符，例如 `","`、`"\t"`、`";"`；或 `whitespace`，表示一个或多个空格/Tab 等空白。
- `format.header`：布尔值，默认 true。
- `format.encoding`：默认 UTF-8（可带 BOM，即 utf-8-sig）；其他文本编码必须显式指定。
- `format.column_count`：可选正整数，用于校验协议总列数。不指定时以表头或第一条数据的列数为准，后续行必须一致。
- `columns`：将统一字段映射到原列名或**零基列号**，可混用；无表头时必须使用整数。uttid、audio_path 必需，有标签输入还需 label；attack_id 可省略。
- `metadata`：可选名称到列的映射，例如 `speaker: 0`。名称必须是简单标识符，不能覆盖保留的 value/uttid，也不能用不同值覆盖同名原始列。
- `paths.root`：相对音频路径的根目录。相对 root 基于**输入清单所在目录**，不是 YAML 或当前工作目录；默认就是清单目录。
- `paths.template`：默认 `"{value}"`。value 是原 audio_path 列，uttid 是映射后的 ID，也可使用命名元数据，例如 `"{speaker}/{value}.flac"`。不允许属性访问、下标、格式化转换或可执行表达式。绝对路径保持绝对路径，不拼接 root。

未传入 format.delimiter 时，仅 .csv/.tsv 使用对应标准分隔符；其他后缀必须显式配置。没有 columns 配置时，只接受标准列名，不猜测 file、target 等字段的含义。无表头示例：

```yaml
format:
  delimiter: whitespace
  header: false
  column_count: 4
columns:
  uttid: 0
  audio_path: 1
  label: 3
  attack_id: 2
labels:
  real: ["genuine"]
  fake: ["synthetic"]
```

标准清单仍兼容：

```csv
uttid,audio_path,label,attack_id
001,audio/001.wav,bonafide,
002,audio/002.flac,spoof,generator_a
```

可用 `--manifest data.csv --label-map bonafide-spoof` 直接读取标准清单，或使用 `--mapping`。不需要中间转换文件；每次运行会自动保存规范化的 inputs.csv。

校验包括唯一 ID、唯一解析音频路径、文件存在、必需字段非空、无首尾空格、矩形行宽、有效表头/列号、标签完整和显式映射。所有配置未知键、重复键、错误类型、缺列、越界索引和未解析路径模板均报错，不跳过坏样本或替换其他音频。

## 4. 标签定义

内部固定 **Real=0，Fake=1**。建议在输入 YAML 中使用 labels.real / labels.fake 字符串列表，可包含多个同义标签。数字标签必须引用并明确方向：

```yaml
labels:
  real: ["1"]
  fake: ["0"]
```

同样支持正向数字、bonafide/spoof、real/fake、genuine/synthetic 及自定义字符串。不会因为原值是 0/1 自动推断真假。未知、缺失、重复或真假冲突的标签立即报错。

也可在输入配置中使用原有格式：

```yaml
label_mapping:
  bonafide: 0
  spoof: 1
```

单独的 `--label-map labels.yaml` 及预设 bonafide-spoof、real-fake、genuine-synthetic 继续兼容。不要同时在输入配置和 --label-map 定义标签，避免冲突。记录保留原始标签与实际解析后的完整配置。

## 5. Labeled 与固定 10-shot 协议

`--group-by none` 将全部 Fake 视作一个类别，执行一个 Real/Fake 任务，不要求知道生成器名称，也不使用 Query Attack ID 选择原型。

`--group-by attack_id` 为每个 Fake Attack 建立独立任务；Fake 必须有非空且不为保留名称 real 的 Attack ID。Real 行的原 Attack 字段保留为元数据，其 Support 在同一种子下供所有任务共享。每任务独立构建 α；Real Query 特征共享，每个 Fake Query 只在对应任务中评分。

**Attack-aware 是使用攻击标注组织的少样本适配协议，不能宣称为未知攻击盲检测或官方无适配排行榜结果。** 标签定义由用户明确提供，读取器不根据数据集名称解释攻击名称。

抽样完全调用原有 `load_or_create_manifest`。`--seeds` 决定抽哪些 Support；默认 `42`，供当前方法验证使用。正式论文实验显式传入 `17,42,71,100,123`。两种协议都保持 10-shot、Support/Query 不重叠和零阈值指标。SpeechFake 内部 Dev/Test 不读取这个参数，仍使用配置中的五种子。

- 每个传入的种子独立用该种子抽样，每类 10 条 Support。
- Query 排除的是**本次传入种子**的 Support 并集。单种子只排除 10 条；五种子排除最多 50 条。因此单种子 Query 与五种子 Query 不同，结果目录也必须分开。
- 同一输入顺序、同一 `--seeds` 会得到同一份 Support/Query。每个新输出目录都会保存自己的 `episode_manifest.json`，不同 Checkpoint 可以对照这份清单，不会改写另一份结果。
- 类顺序为 Real 在前、Fake 排序；类内保持输入行顺序。重排清单可能改变 Support。
- 每类必须有至少 10 个候选，且并集排除后还存在 Query。单种子每类至少 11 条；五种子每类至少 51 条才能保证并集达到 50 条后仍有 Query。
- 单种子报告直接写 EER/AUC/AP/ACC（`external_metrics.csv`）。只有多个种子才写 Mean ± Std，且仅当种子正好是 `17,42,71,100,123` 时记为正式五种子协议。
- 使用原有 view_tokens 的逐样本裁剪种子；相同 Query ID 不代表跨种子裁剪完全相同。

## 6. Unlabeled：已标记 Support 与未知 Query

仍须提供**恰好 10 Real + 10 Fake Support**。不自动取前十条，不把未知 Query 当 Fake Support，也不切换到 0-shot。

```bash
python scripts/evaluate_external.py \
  --mode unlabeled \
  --real-support-dir /data/MyDataset/support/real \
  --fake-support-dir /data/MyDataset/support/fake \
  --query-dir /data/MyDataset/query \
  --checkpoint "$CKPT" --set "model.frontend_path=$FRONTEND" \
  --output-dir "$RESULTS/custom_unlabeled_run1"
```

目录递归稳定排序，ID 为角色加相对路径。目录角色直接声明标签，不额外传 --mapping/--label-map。也可以用不同格式的两个清单：

```bash
python scripts/evaluate_external.py \
  --mode unlabeled \
  --support-manifest /data/custom/support.table --mapping /data/custom/support.yaml \
  --query-manifest /data/custom/query.tsv --query-mapping /data/custom/query.yaml \
  --checkpoint "$CKPT" --set "model.frontend_path=$FRONTEND" \
  --output-dir "$RESULTS/custom_manifest_unlabeled_run1"
```

--mapping 用于 Support；--query-mapping 用于 Query。Query 只需 ID 与路径，若配置了标签列，其非空标签会被拒绝；标准 label 列也不得含非空标签。未知列只作为元数据保存，不据此推断隐藏标签。Support/Query 必须具有互不重复的 ID 与解析路径。

用户指定 Support 只运行一次，裁剪种子 42，不重采样五套 Support。输出仅含分数和预测，不生成任何 EER/AUC/AP/ACC 或虚构的 Mean ± Std。

## 7. ASVspoof5：普通配置示例

官方发布版 Track 1 的列位置依据[官方 README](https://zenodo.org/records/14498691/files/README.txt?download=1)。`configs/external/asvspoof5.yaml` 是普通输入配置，描述十列混合空白表格、音频 ID 后缀和显式标签；Python 不检查或分支处理数据集名称。

直接读取原始协议，**不运行转换脚本**：

```bash
ASV_ROOT=/root/autodl-tmp/ASVspoof5_extracted
# 当前方法验证：固定单种子 42。换 Checkpoint 时更换输出目录，不要覆盖已有结果。
python scripts/evaluate_external.py \
  --mode labeled --manifest "$ASV_ROOT/ASVspoof5.eval.track_1.tsv" \
  --mapping configs/external/asvspoof5.yaml --group-by attack_id \
  --seeds 42 \
  --checkpoint "$CKPT" --set "model.frontend_path=$FRONTEND" \
  --output-dir "$RESULTS/asvspoof5_attack_seed42_run1"

# 最终论文实验：同一代码，只把种子和输出目录换成五种子正式协议。
python scripts/evaluate_external.py \
  --mode labeled --manifest "$ASV_ROOT/ASVspoof5.eval.track_1.tsv" \
  --mapping configs/external/asvspoof5.yaml --group-by attack_id \
  --seeds 17,42,71,100,123 \
  --checkpoint "$CKPT" --set "model.frontend_path=$FRONTEND" \
  --output-dir "$RESULTS/asvspoof5_attack_five_seed_run1"
```

示例 root 为相对于协议文件的 flac_E_eval。协议在额外子目录、使用 dev 的 flac_D 或早期其他列布局时，复制 YAML 并修改 root、列号、column_count 即可，不修改 Python。Track 2 的标签和列语义不同，不能拿 Track 1 配置直接读取。原始 Real 的 bonafide Attack 字段保留，Real/Fake 身份只由 label_mapping 决定。

## 8. 同一模型、预处理与结果解释

SpeechFake 内部 evaluate.py 和外部两种模式共用 `engine.evaluator.evaluate`、forward_batch、kshot_task、kshot_scores、predict 与原有指标函数。保留 SpeechFake 的 32 类限制、固定 Dev/Test 协议、输出格式、ReusableAudioLoader、Persistent Workers、合并 Support、共享 Real Query 和音频微批次。本次输入统一化不改变该核心。

原始音频复用 Layer-Wise：第一声道、16 kHz 重采样、静音裁剪、RMS 归一化、4 秒确定性裁剪。旧读取器不支持的多声道重采样由通用外部适配层取第一声道、经内存 float WAV 交回同一读取器。无 RIR/RawBoost，无外部 waveform/特征缓存；data/audio.py 不改动，现有 SpeechFake 缓存身份不因此失效。

分数仍为 `S_fake-S_real`；score >= 0 为 Fake，否则 Real。它不是 Fake 概率。有标签时原样计算 EER/AUC/AP/ACC，ACC 使用零阈值；eer_threshold 仅作 EER 报告，不替换预测阈值。先逐任务宏平均。多个种子再计算 Mean 和总体 Std（ddof=0）；单种子直接报告这一次的 EER/AUC/AP/ACC，不写成 Mean ± Std。单任务只有一类任务统计，不制造多个 Fake 类。

输出包括：

- external_run.json：状态、原始清单 SHA256、输入配置文件 SHA256、实际使用的字段/路径/标签配置、Checkpoint SHA256/step、Git Commit/状态/源码哈希、完整模型配置、CLI 参数、种子和耗时。
- inputs.csv：全部规范化输入、原始标签与 JSON 元数据；episode_manifest.json 保存完整 Support/Query ID，可联合恢复所有路径与角色。
- external_predictions.csv：全部 Query Score/Prediction；有标签时另保存真值、原始标签和 Attack ID。
- 有标签且只有一个种子时，写 `external_metrics.csv` 和 `external_per_class.csv`，直接给出 EER/AUC/AP/ACC。多个种子才写逐任务 Mean/Std 和 Macro Mean/Std。Attack-aware 的 Real Query 每个任务各有评分。
- failures.jsonl：成功时为空，失败时记录异常链；音频错误含 ID 和路径。失败不会替换样本，部分分数只供排错。

## 9. 验证与输入边界

运行 `python -m pytest -q` 做完整回归；新增映射测试位于 test_external_input_mapping.py，已有 external 测试及冻结旧版数值基线均保留。三种虚拟数据分别覆盖改名 CSV、重排列号 TSV、无表头混合空白协议，并经同一 CLI 完成 Labeled；不同格式的 Support/Query 同样共用 Unlabeled 入口。

输入范围是矩形文本表格。CSV/TSV 支持标准引号；混合空白模式按空白切词，不支持字段内带空格或引号，应改用 CSV/TSV。嵌套 JSON/XML、归档内音频、任意非表格协议及可执行解析规则不在本次范围；可先准备标准清单。路径必须在当前运行主机可访问，不自动挂载或猜测其他主机的目录。

复现需保留音频字节、原始清单顺序、输入配置、模型/官方前端权重与软件环境。清单 SHA256 不等于逐音频内容哈希；不同 CUDA/库版本不承诺逐位一致。本次不执行完整外部正式评估，无 Support 的单音频推理属于独立场景，不在当前 10-shot 方法内。

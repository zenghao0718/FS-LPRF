# 整合代码的来源与边界

FS-LPRF 将实际使用的 Layer-Wise 组件整合进自身的功能目录，不再通过 Git Submodule 或外部 `tomato` 包加载代码。来源版本为 [XIAOYixuan/tomatoDD](https://github.com/XIAOYixuan/tomatoDD/tree/88efe1376dfc5f5b9b62d016d33a7381c158d8af)，分支 `interspeech25-layer-wise`，commit `88efe1376dfc5f5b9b62d016d33a7381c158d8af`。原文件中的作者信息和已有许可声明随所用代码保留。

## 文件映射

- `src/fs_lprf/data/audio.py`：来自 `tomato/data/audio_util.py`、`fad_trim.py` 的 FADTrim/GeneralFAD 实际波形路径及 `base.py` 的必要初始化。作者 Yixuan；原 `audio_util.py` 另注明改编自 [piotrkawa/deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features)。保留 soundfile 读取、librosa 重采样/静音裁剪、第一声道、RMS、重复补足、NumPy 随机裁剪和 batch 拼接。未整合其他数据集、特征缓存或 LFCC/MFCC 变换。
- `src/fs_lprf/data/augmentation.py`：来自 `tomato/data/audio_augmentation.py` 的 RIR 卷积/功率处理及 RawBoost 调用顺序，和 `tomato/data/RawBoost/RawBoost.py`、`data_utils_rawboost.py` 中 Algorithm 5 所需函数。RawBoost 作者为 Hemlata Tak、Massimiliano Todisco；原文件注明 RIR 实现基于 ESPNet2。保留原默认参数、1/3 与 1/5 概率、随机调用顺序和全部数值步骤；不整合其他 RawBoost 组合、MUSAN 或粉红噪声路径。
- `src/fs_lprf/models/xlsr.py`：来自 `tomato/models/frontend_models/fairseq_models/wav2vec2.py` 中 Wav2Vec2Model 的特征提取路径、卷积编码器、TransformerEncoder、TransformerSentenceEncoderLayer 及位置卷积；XLSR 权重加载边界源于 `frontend_models/w2v2.py`。不包含 Conformer、Adapter 或预训练损失/采样执行路径。
- `src/fs_lprf/models/xlsr_layers.py`：来自上述 `fairseq_models/modules/` 的 LayerNorm、Fp32LayerNorm/Fp32GroupNorm、GELU、GradMultiply、SamePad、TransposeLast、GumbelVectorQuantizer，以及 `fairseq_models/utils.py` 的激活/索引辅助函数。冻结的 quantizer、project_q、mask_emb、final_proj 等参数仍保留，以严格加载原始权重的 429 个键，并保留构造时的随机数消耗顺序；不参与 FS-LPRF 后端训练。
- `src/fs_lprf/models/xlsr_attention.py`：来自 `fairseq_models/modules/multihead_attention.py` 的实际 PyTorch 自注意力路径与原初始化，以及 `fairseq_dropout.py`。保留 k/v/q/out 投影名称和创建顺序，仍调用原 `torch.nn.functional.multi_head_attention_forward`；未整合 xFormers、解码器缓存、剪枝或量化噪声路径。
- `src/fs_lprf/models/fs_lprf_model.py`：特征提取与时间 Attention 来自 `tomato/models/frontend_only.py` 的 XLSRAdapterBase/XLSRTimeAttnOnly；冻结前端和后端 state_dict 规则来自 `tomato/models/base.py` 的 ClassificationBase。保留原 logsumexp→softmax→时间加权求和、Xavier 初始化、25 层独立 4-head 参数及已有共享 Bottleneck，不包含重建头或其他前端。
- `src/fs_lprf/utils/metrics.py`：DET/EER 来自 `tomato/train_util/eer.py`；其余指标继续使用 FS-LPRF 既有实现。保留 mergesort 同分排序、deepfake=1、阈值起点和最小差位置的约定。

## 已有许可声明

Layer-Wise 固定版本中的 Fairseq 文件注明版权属于 Facebook, Inc. and its affiliates，并使用 MIT 许可；同时注明由 Yixuan 适配。其 README 标记所依据的 Fairseq commit 为 [`920a548ca770fb1a951f7f4289b4d3a0c1bc226f`](https://github.com/facebookresearch/fairseq/tree/920a548ca770fb1a951f7f4289b4d3a0c1bc226f)。对应 [原始 MIT 许可](https://github.com/facebookresearch/fairseq/blob/920a548ca770fb1a951f7f4289b4d3a0c1bc226f/LICENSE) 全文保存在 [LICENSES/Fairseq-MIT.txt](../LICENSES/Fairseq-MIT.txt)，适用于保留相关声明的整合文件。

Layer-Wise 该固定版本未提供整体 LICENSE 文件；本次保留其文件级来源和作者信息，不新增对其整体许可的声明。

## 兼容性与验证材料

`src/fs_lprf/provenance.py` 保留 `UPSTREAM_COMMIT` 原值与官方 XLS-R URL/大小/SHA-256；这些是历史来源元数据，不要求外部仓库存在。训练 Checkpoint 与原型库中的 `upstream_commit` 字段及既有 schema 不变。

`tests/fixtures/xlsr_small_reference.json` 是在整合前由上述固定版本生成的小型合成 Transformer 输出、配置与参数形状，用于独立数值回归；其中不包含任何预训练权重。真实 XLS-R 的整合前后逐元素对照、旧 Checkpoint 和 Smoke 报告位于被 Git 忽略的 `outputs/integration_validation/`，验收结论记录在 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

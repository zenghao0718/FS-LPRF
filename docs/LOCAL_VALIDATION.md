# 本地轻量验收记录

日期：2026-11-02。范围：合成向量、临时 WAV、真实官方 XLS-R 前端的小批量验证。未下载 SpeechFake-BD、未运行正式训练或完整 Dev/Test、未构建真实 0-shot 原型库。本记录不是实验结果。

## 最终环境

- 解释器：`D:\FS-LPRF\.venv\Scripts\python.exe`
- `sys.prefix`：`D:\FS-LPRF\.venv`；`include-system-site-packages=false`
- Python：3.11.18
- PyTorch：2.7.1+cu128；torchaudio：2.7.1+cu128
- NumPy：1.26.4；librosa：0.10.2.post1
- GPU：NVIDIA GeForce RTX 5060，Compute Capability `(12,0)`；PyTorch 编译架构包含 `sm_120`
- CUDA runtime：12.8
- `python -m pip check`：`No broken requirements found.`

最终测试和以下真实模型验收均在项目 `.venv` 中完成。前期曾借用既有环境做代码排查，未将其作为最终验收环境。

## 官方文件与加载兼容性

文件来自 `https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt`。

- 文件大小：3,808,868,242 bytes（3,632.42 MiB）
- SHA-256：`b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9`
- 完整文件保存在 `pretrained/xlsr2_300m.pt`；来源记录为 `pretrained/xlsr2_300m.pt.source.json`
- 通过官方 HTTP Range 分段续传完成，未采用转换版/Hugging Face 权重。

实际测试首先调用默认 `torch.load(path, map_location="cpu")`：**通过**。PyTorch 2.7.1 本次不需要 `weights_only=False`。受 SHA-256 限制的最小 fallback 已保留，来源不匹配的文件会在反序列化前被拒绝。

原始 Fairseq 快照保存 `cfg.model` 和 `model`，而固定版本 Layer-Wise 的加载器期望 `cfg: Namespace` 与 `state_dict`。项目只在内存读取边界适配字段和配置对象；未改变张量、前端类或 Hidden States 提取。原文件缺失的六个字段沿用 Fairseq 官方历史默认值：`crop_seq_to_multiple=1`、`layer_type=transformer`、`mask_channel_before=False`、`quantizer_depth=1`、`quantizer_factor=3`、`required_seq_len_multiple=2`。

PyTorch CUDA 安装包也经过官方 wheel 索引 SHA-256 校验：`138c66dcd0ed2f07aafba3ed8b7958e2bed893694990e0b4b55b6b2b4a336aa6`。该安装包及缓存均未纳入 Git。

## 单元与接口测试

执行命令：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/local_smoke/pytest.xml
```

结果：**36 passed，66.36 秒**。输出的 72 条 warning 来自官方 RawBoost 在 NumPy 1.26 下的标量转换弃用提示及 librosa 的资源接口弃用提示；固定依赖下相关测试正常通过，上游文件保持原样。

覆盖内容包括：

- 原型均值、逐层 logits、Real–Fake 类间距离、类内离散求和、float32 可靠性和 α detach。
- Query 平均 CE、层可靠性加权、独立 Episode 聚合、Support 原型的梯度路径。
- 每 Step 仅一次 backward 和一次 optimizer update；240 个合成实例按 8 个音频一组编码，并正确还原 Episode。
- Episode 类别独立采样、类内 Support/Query 不重叠。
- 官方音频读取、重采样、第一声道、重复补足、64000 samples；坏音频不替换类别。
- RIR 官方卷积和功率处理、两增强概率与顺序，以及官方 RawBoost Algorithm 5 的实际执行。
- 五种子固定 Manifest、共享 Query、Support 并集排除、完整性校验与复用。
- `num_workers=0/2` 下同种子跨 Checkpoint 的官方裁剪序列一致，不同种子允许不同；评估恢复训练 RNG。
- 五种子评估引擎、逐类宏平均、连续分数方向与固定零阈值、逐条预测导出。
- 合成 33 类 0-shot 原型库、逐层最大 Fake 相似度、Real 来源配额、不放回与容量不足报错。
- Checkpoint 中模型后端、Optimizer、Scheduler 与 RNG 往返恢复。
- 下载断点 Range 复用、SHA-256 校验、保留已有下载数据，以及完整文件重复调用不访问网络。
- 官方 Submodule 固定 commit，且源码无修改。

六个 CLI 的 `--help` 均通过：`check_data.py`、`check_model.py`、`train.py`、`evaluate.py`、`build_prototype_bank.py`、`download_frontend.py`。

## 真实 XLS-R CUDA Smoke Test

执行命令：

```powershell
& .\.venv\Scripts\python.exe scripts/check_model.py `
  --config configs/speechfake.yaml `
  --report outputs/local_smoke/model_report.json
```

结果：**passed**。使用正式 `FSLPRF` 类、真实官方权重、一条临时合成 4 秒/16 kHz/mono WAV 和官方音频链路，batch size 为 1。

- 完整 25 组表示：第 0 组 `[1,199,1024]`，其余 24 组 `[199,1,1024]`。
- 官方 Attention 输出：`[1,25,1024]`。
- 共享 Bottleneck 输出：`[1,25,256]`。
- 前端所有参数冻结，`model.train()` 后前端仍为 eval，前端无梯度。
- 25 个 Attention 参数切片独立，且每层均获得有限、非零梯度。
- 仅 Attention 和共享 Linear 可训练，共 364,800 个参数。
- 诊断损失下梯度范数：Attention 187.5170；Bottleneck weight 41.3539；bias 0.1767。
- CUDA 张量峰值分配：1,380,057,600 bytes，约 1.29 GiB（不包含其他进程和驱动占用）。
- 后端 Checkpoint：1,506,813 bytes，排除冻结 XLS-R，保存/加载后权重一致。
- 未执行 optimizer update，未进行正式训练。

Smoke 入口曾发现 CUDA 显存统计在设备初始化前调用的问题，已修正初始化顺序并以上述命令重跑通过。可选 FFmpeg 扩展在本机未加载，音频使用官方 soundfile 流程，所有音频检查通过；不为此额外安装或替换预处理组件。

详细机器可读报告与控制台记录位于 `outputs/local_smoke/`，按要求被 Git 忽略。AutoDL A100 40GB 的正式环境、数据检查、训练、恢复和 Test 10-shot 步骤见 [AUTODL.md](AUTODL.md)；本地未执行这些正式实验。

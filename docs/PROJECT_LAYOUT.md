# 平级科研代码布局与迁移清单

本次只迁移模块位置、导入和启动配置。功能目录直接位于项目根目录，保留 FS-LPRF 的模型、算法、数据处理、训练/评估协议及 Checkpoint 行为。目录组织参考 [DDFSD](https://github.com/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector)，具体目录命名以本项目约定为准。

以下目录树完整列出受 Git 管理的项目文件；本地数据、权重、缓存和运行输出继续由忽略规则排除。旧目录和旧路径仅用于迁移对照，不是当前运行要求。

## 迁移前

基准提交：`8f3fbe0406eda2536310283e105112d66440eafa`。

```text
FS-LPRF/
├── configs/
│   └── speechfake.yaml
├── docs/
│   ├── AUTODL.md
│   ├── CODE_PROVENANCE.md
│   ├── LOCAL_VALIDATION.md
│   └── VALIDATION_SCHEDULE.md
├── LICENSES/
│   └── Fairseq-MIT.txt
├── scripts/
│   ├── _bootstrap.py
│   ├── build_prototype_bank.py
│   ├── check_data.py
│   ├── check_model.py
│   ├── download_frontend.py
│   ├── evaluate.py
│   └── train.py
├── src/
│   └── fs_lprf/
│       ├── data/
│       │   ├── __init__.py
│       │   ├── audio.py
│       │   ├── augmentation.py
│       │   ├── episode_sampler.py
│       │   ├── evaluation_manifest.py
│       │   ├── loaders.py
│       │   └── speechfake_dataset.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── checkpoint.py
│       │   ├── evaluator.py
│       │   ├── trainer.py
│       │   └── validation_schedule.py
│       ├── methods/
│       │   ├── __init__.py
│       │   ├── inference.py
│       │   ├── loss.py
│       │   ├── prototype_bank.py
│       │   ├── prototypes.py
│       │   └── reliability.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── frontend_compat.py
│       │   ├── fs_lprf_model.py
│       │   ├── xlsr.py
│       │   ├── xlsr_attention.py
│       │   └── xlsr_layers.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   ├── metrics.py
│       │   └── seed.py
│       ├── __init__.py
│       ├── config.py
│       └── provenance.py
├── tests/
│   ├── fixtures/
│   │   └── xlsr_small_reference.json
│   ├── conftest.py
│   ├── test_checkpoint_refactor.py
│   ├── test_data_protocol.py
│   ├── test_download.py
│   ├── test_engine.py
│   ├── test_evaluation.py
│   ├── test_integrated_components.py
│   ├── test_light_manifest.py
│   ├── test_methods.py
│   ├── test_rir_candidates.py
│   ├── test_validation_reports.py
│   └── test_validation_schedule.py
├── .gitattributes
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 迁移后

```text
FS-LPRF/
├── configs/
│   └── speechfake.yaml
├── data/
│   ├── __init__.py
│   ├── audio.py
│   ├── augmentation.py
│   ├── episode_sampler.py
│   ├── evaluation_manifest.py
│   ├── loaders.py
│   └── speechfake_dataset.py
├── docs/
│   ├── AUTODL.md
│   ├── CODE_PROVENANCE.md
│   ├── LOCAL_VALIDATION.md
│   ├── PROJECT_LAYOUT.md
│   └── VALIDATION_SCHEDULE.md
├── engine/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── evaluator.py
│   ├── trainer.py
│   └── validation_schedule.py
├── LICENSES/
│   └── Fairseq-MIT.txt
├── methods/
│   ├── __init__.py
│   ├── inference.py
│   ├── loss.py
│   ├── prototype_bank.py
│   ├── prototypes.py
│   └── reliability.py
├── models/
│   ├── __init__.py
│   ├── frontend_compat.py
│   ├── fs_lprf_model.py
│   ├── xlsr.py
│   ├── xlsr_attention.py
│   └── xlsr_layers.py
├── scripts/
│   ├── _bootstrap.py
│   ├── build_prototype_bank.py
│   ├── check_data.py
│   ├── check_model.py
│   ├── download_frontend.py
│   ├── evaluate.py
│   └── train.py
├── tests/
│   ├── fixtures/
│   │   └── xlsr_small_reference.json
│   ├── conftest.py
│   ├── test_checkpoint_refactor.py
│   ├── test_data_protocol.py
│   ├── test_download.py
│   ├── test_engine.py
│   ├── test_evaluation.py
│   ├── test_integrated_components.py
│   ├── test_light_manifest.py
│   ├── test_methods.py
│   ├── test_rir_candidates.py
│   ├── test_validation_reports.py
│   └── test_validation_schedule.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   ├── metrics.py
│   ├── provenance.py
│   └── seed.py
├── .gitattributes
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 全部源码文件迁移

以下 30 个文件按原文件名迁移；配置和来源元数据移到 `utils/`，其余功能目录直接提升到根目录。

- `src/fs_lprf/config.py` → `utils/config.py`
- `src/fs_lprf/data/__init__.py` → `data/__init__.py`
- `src/fs_lprf/data/audio.py` → `data/audio.py`
- `src/fs_lprf/data/augmentation.py` → `data/augmentation.py`
- `src/fs_lprf/data/episode_sampler.py` → `data/episode_sampler.py`
- `src/fs_lprf/data/evaluation_manifest.py` → `data/evaluation_manifest.py`
- `src/fs_lprf/data/loaders.py` → `data/loaders.py`
- `src/fs_lprf/data/speechfake_dataset.py` → `data/speechfake_dataset.py`
- `src/fs_lprf/engine/__init__.py` → `engine/__init__.py`
- `src/fs_lprf/engine/checkpoint.py` → `engine/checkpoint.py`
- `src/fs_lprf/engine/evaluator.py` → `engine/evaluator.py`
- `src/fs_lprf/engine/trainer.py` → `engine/trainer.py`
- `src/fs_lprf/engine/validation_schedule.py` → `engine/validation_schedule.py`
- `src/fs_lprf/methods/__init__.py` → `methods/__init__.py`
- `src/fs_lprf/methods/inference.py` → `methods/inference.py`
- `src/fs_lprf/methods/loss.py` → `methods/loss.py`
- `src/fs_lprf/methods/prototype_bank.py` → `methods/prototype_bank.py`
- `src/fs_lprf/methods/prototypes.py` → `methods/prototypes.py`
- `src/fs_lprf/methods/reliability.py` → `methods/reliability.py`
- `src/fs_lprf/models/__init__.py` → `models/__init__.py`
- `src/fs_lprf/models/frontend_compat.py` → `models/frontend_compat.py`
- `src/fs_lprf/models/fs_lprf_model.py` → `models/fs_lprf_model.py`
- `src/fs_lprf/models/xlsr.py` → `models/xlsr.py`
- `src/fs_lprf/models/xlsr_attention.py` → `models/xlsr_attention.py`
- `src/fs_lprf/models/xlsr_layers.py` → `models/xlsr_layers.py`
- `src/fs_lprf/provenance.py` → `utils/provenance.py`
- `src/fs_lprf/utils/__init__.py` → `utils/__init__.py`
- `src/fs_lprf/utils/logging.py` → `utils/logging.py`
- `src/fs_lprf/utils/metrics.py` → `utils/metrics.py`
- `src/fs_lprf/utils/seed.py` → `utils/seed.py`

另将 `src/fs_lprf/__init__.py` 中的模块说明和 `__version__ = "0.1.0"` 保留到 `utils/__init__.py`，扩充原本为空的工具包初始化文件。旧顶层包初始化文件随旧包骨架移除，没有删除功能实现。

## 导入、入口与配置

- 导入直接使用 `data.*`、`models.*`、`methods.*`、`engine.*` 和 `utils.*`；配置工具为 `utils.config`，来源常量为 `utils.provenance`。功能目录内部的相对导入保持原样。
- `scripts/_bootstrap.py` 仅在需要时将项目根目录加入 Python 路径；六个原 CLI 保留原文件名和参数，从项目根目录直接运行。
- `pyproject.toml` 移除旧包构建和 setuptools 包发现配置，保留项目名称、版本、描述和 Python 版本要求；pytest 保留 `testpaths = ["tests"]`，使用 `pythonpath = ["."]`。
- 依赖通过 `python -m pip install -r requirements.txt` 安装；在克隆目录直接运行，不需要安装本项目为 editable 包或设置额外 Python 路径。
- `configs/speechfake.yaml`、所有默认字段和 `--set section.key=value` 覆盖逻辑保持不变；数据、权重、输出及 AutoDL 配置路径不改。
- `data/` 现在是源码目录。忽略规则只放行其中的 Python 源文件，本地数据文件和子目录仍被排除，权重/音频/Checkpoint/日志的原排除规则保留。

## 验证方式

在现有项目 `.venv` 中执行完整 pytest、pip check、六个 CLI 的 `--help`，并在阻止旧包导入的隔离进程中加载全部模块。迁移前后配置进行逐字段比较；旧布局生成的合成后端 Checkpoint 和 PrototypeBank 在新布局中加载。核心源码逐文件比较，除导入路径和保留的包版本元数据外没有变化。

本次验收记录见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。不运行正式训练，不读取真实数据集，不下载模型或其他资源。

# RAN-LLM-xApp

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/Research-Prototype-2F855A?style=flat)](#)
[![Paper](https://img.shields.io/badge/INFOCOM%20Workshop-2026%20Accepted-B31B1B?style=flat)](#paper-status)

## Paper Status

**English.** This repository is associated with the paper **"LLM-xApp: A Large Language Model Empowered Radio Resource Management xApp for 5G O-RAN"**, which has been **accepted by INFOCOM Workshop 2026**.

**中文。** 本仓库关联论文 **《LLM-xApp: A Large Language Model Empowered Radio Resource Management xApp for 5G O-RAN》**。该论文已被 **INFOCOM Workshop 2026** 接收。

## Overview

**English.** RAN-LLM-xApp is a research-grade, pure-Python simulation framework for studying LLM-assisted radio resource management in 5G O-RAN slicing scenarios. It implements the key metric definitions and decision pipeline inspired by the LLM-xApp paper, including utility functions, sliding-window reliability, action-to-PRB mapping, local evaluation, and LLM-assisted candidate generation.

This repository is designed for controlled algorithmic research, ablation studies, and reproducible experimentation. It does **not** depend on OAIC, srsRAN, or a live O-RAN testbed, and it is **not** intended to be a bit-level reproduction of the original experimental testbed. Instead, it provides a compact and inspectable platform for validating method behavior, extending the controller, and comparing LLM-backed policies under transparent synthetic dynamics.

**中文。** RAN-LLM-xApp 是一个面向研究的纯 Python 仿真框架，用于探索大语言模型辅助的 5G O-RAN 切片无线资源管理。项目实现了 LLM-xApp 论文中的关键指标定义与决策流程，包括 utility、滑动窗口 reliability、动作到 PRB 的映射、本地评价函数，以及 LLM 辅助候选动作生成。

本仓库的定位是可控算法研究、消融实验与可复现实验平台。它**不依赖** OAIC、srsRAN 或真实 O-RAN testbed，也**不追求**与原论文 testbed 的逐点数值复现。相反，它提供了一个结构清晰、易于审计和扩展的研究原型，用于验证方法趋势、扩展控制器逻辑，并在透明的合成环境中比较多种 LLM-backed 策略。

## What Is Implemented

**English.**

- Four allocation policies: `equal`, `random`, `proportional`, and `tnas`.
- Paper-inspired metrics:
  - Utility functions for two slices.
  - Sliding-window outage/reliability.
  - Eq.(7)-style action-to-PRB mapping with budget correction.
  - Eq.(8)-style local evaluation with `g(x) = -x^2`.
- TNAS: Top-N Action Sampling with **LLM proposal + local reranking**.
- RealScore-driven local critic for online scoring of LLM-generated candidates.
- Multi-model experiment support for OpenAI, DeepSeek, Google-compatible endpoints, and a local deterministic `stub`.
- CSV and publication-style figure generation for time-series and aggregate metrics.

**中文。**

- 实现 4 类资源分配策略：`equal`、`random`、`proportional`、`tnas`。
- 实现论文启发的关键指标：
  - 双切片 utility 函数。
  - 滑动窗口 outage/reliability。
  - 类 Eq.(7) 的动作到 PRB 映射，并包含超预算修正。
  - 类 Eq.(8) 的本地评价函数，默认 `g(x) = -x^2`。
- TNAS：Top-N Action Sampling，即**大模型生成候选动作 + 本地可控重排序**。
- RealScore-driven 本地 critic，用真实反馈在线学习候选动作评分。
- 支持 OpenAI、DeepSeek、Google-compatible endpoint，以及纯本地确定性 `stub`。
- 自动输出 CSV、时序图与聚合指标图，便于论文实验和结果分析。

## Repository Structure

```text
ran_llm_xapp/
  config.py              # Experiment configuration and default research scenario
  env.py                 # Synthetic RAN slicing environment
  metrics.py             # Utility, reliability, score, and PRB mapping functions
  plotting.py            # Figure generation
  prompts.py             # Prompt templates for LLM-assisted control
  policies/
    equal.py             # Equal allocation baseline
    random.py            # Random allocation baseline
    proportional.py      # Demand-proportional baseline
    tnas.py              # Top-N Action Sampling policy
  llm_clients/
    openai_client.py     # OpenAI-compatible client
    deepseek_client.py   # DeepSeek-compatible client
    google_client.py     # Google-compatible client
    stub_client.py       # Local deterministic LLM stub
scripts/
  run_experiments.py             # Main experiment runner
  run_seed_sweep_sys_metrics.py  # Multi-seed system-metric sweep
tests/                           # Unit tests for mapping and metrics
```

## Installation

**English.** A virtual environment is recommended.

**中文。** 建议使用虚拟环境安装。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

## Quick Start

**English.** Run all baselines and one local TNAS variant with the deterministic `stub` client:

**中文。** 使用纯本地 `stub` 客户端运行所有 baseline 与一个 TNAS 变体：

```bash
python3 -m scripts.run_experiments --methods all2 --seed 0 --out outputs/all2_stub \
  --llm-runs stub:stub
```

**English.** Run baselines plus multiple LLM-backed TNAS variants:

**中文。** 运行 baseline，并对多个真实大模型的 TNAS 变体做对比：

```bash
python3 -m scripts.run_experiments --methods all --seed 0 --out outputs/multi_llm \
  --llm-runs openai:gpt-4o deepseek:deepseek-v3.2 google:gemini-3-pro
```

When `--llm-runs` is provided, each model is treated as an independent `tnas_*` variant and is included in the same comparison pipeline.

当提供 `--llm-runs` 时，每个模型都会被视为一个独立的 `tnas_*` 变体，并进入同一套对比流程。

## Common Commands

```bash
# TNAS only, using the local deterministic stub
python3 -m scripts.run_experiments --methods tnas --seed 0 --out outputs/tnas_only \
  --llm-runs stub:stub

# Equal + proportional baselines
python3 -m scripts.run_experiments --methods equal proportional --seed 0 --out outputs/eq_prop

# Baselines plus TNAS
python3 -m scripts.run_experiments --methods all2 --seed 0 --out outputs/all2_stub \
  --llm-runs stub:stub

# Multi-seed system utility/reliability sweep
python3 -m scripts.run_seed_sweep_sys_metrics \
  --llm-runs openai:gpt-4o-mini deepseek:deepseek-v3.2 google:gemini-3-pro
```

## TNAS: Top-N Action Sampling

**English.** TNAS follows a deliberately conservative control architecture: the LLM proposes a diverse set of candidate actions, while the final decision is made by a local evaluator. This avoids handing full control authority to the LLM and makes the optimization process auditable.

The control loop at each reconfiguration slot is:

1. Build an observation from recent measured throughput, current demand, current PRB allocation, and soft-shortfall information.
2. Ask the LLM to return Top-N candidate actions `(a1, a2)` in strict JSON format.
3. Map each action to `(prb1, prb2)` using Eq.(7)-style proportional mapping with budget correction.
4. Locally rerank candidates with either RealScore-driven scoring or deterministic proxy scoring.
5. Execute the selected PRB allocation and record feedback for future slots.

**中文。** TNAS 采用一种审慎的控制架构：LLM 只负责提出多样化候选动作，最终决策由本地评价器完成。这样可以避免将控制权完全交给 LLM，同时保证优化过程可审计、可复现、可调试。

每个重配置 slot 的流程如下：

1. 根据最近实测吞吐、当前需求、当前 PRB 分配和 soft-shortfall 构造观测。
2. 要求 LLM 以严格 JSON 格式返回 Top-N 个候选动作 `(a1, a2)`。
3. 使用类 Eq.(7) 的比例映射将动作转换为 `(prb1, prb2)`，并做预算修正。
4. 使用 RealScore-driven scoring 或确定性 proxy scoring 对候选动作本地重排序。
5. 执行最终 PRB 分配，并记录反馈用于后续 slot 的在线学习。

Relevant files:

- `ran_llm_xapp/policies/tnas.py`
- `ran_llm_xapp/prompts.py`
- `ran_llm_xapp/metrics.py`

## Experiment Timeline and Metrics

**English.** The default configuration is designed to produce paper-like RAN slicing dynamics:

- `t in [0, 100)`: pre-slicing fixed allocation, default `prb1=96`, `prb2=32`.
- `t in [100, 200)`: slice initialization with equal allocation, `prb1=64`, `prb2=64`.
- `t >= 200`: policy-controlled allocation with a demand schedule.
  - `sigma1=35`, `sigma2=10` from `t=200`.
  - `sigma1=45`, `sigma2=10` from `t=400`.

System-level metrics include weighted system utility, outage fraction, reliability, and severity-weighted reliability.

**中文。** 默认配置用于构造接近论文叙述的 RAN slicing 动态：

- `t in [0, 100)`：未启用 slicing 控制的固定分配阶段，默认 `prb1=96`、`prb2=32`。
- `t in [100, 200)`：slice 初始化阶段，使用均分分配 `prb1=64`、`prb2=64`。
- `t >= 200`：策略生效，并启用需求变化。
  - 从 `t=200` 开始，`sigma1=35`、`sigma2=10`。
  - 从 `t=400` 开始，`sigma1=45`、`sigma2=10`。

系统级指标包括加权 system utility、outage fraction、reliability，以及 severity-weighted reliability。

## LLM Provider Configuration

**English.** API keys can be provided through `.env`. This repository intentionally requires explicit `*_BASE_URL` values so that experiments are reproducible across official endpoints, gateways, and local proxies.

**中文。** 可以通过 `.env` 配置 API key。本项目要求显式设置 `*_BASE_URL`，这样可以在官方 endpoint、网关或本地代理之间保持实验配置可追踪。

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...

DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=...

GOOGLE_API_KEY=...
GOOGLE_BASE_URL=...
```

If a selected online provider is missing either key or base URL, the runner will fall back to the local `stub` client so that the experiment pipeline remains executable.

如果选择在线 provider 但缺少 key 或 base URL，程序会自动退化到本地 `stub` 客户端，保证实验流程仍可跑通。

## Outputs

Each experiment writes results to the directory specified by `--out`.

每次实验会将结果写入 `--out` 指定目录。

Typical outputs include:

- `timeseries_<method>.csv`: per-time-step PRB, throughput, utility, outage, reliability, and soft-score traces.
- `timeseries_tnas_<provider>_<model>.csv`: additional files for multi-model TNAS runs.
- `fig4_<method>.png` and `.pdf`: per-method time-series figures.
- `fig4.png` and `.pdf`: combined comparison figure.
- `fig5a_sys_utility.png`: smoothed system utility.
- `fig5b_sys_reliability_severity.png`: severity-weighted system reliability.
- `fig5c_avg_utility.png`: time-averaged utility.
- `fig5d_avg_reliability_severity.png`: time-averaged severity-weighted reliability.
- `config_used.yaml` or `config_used.json`: the exact configuration used for the run.

## Testing

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Notes on Reproducibility

**English.** This codebase is intended to make the research logic explicit: random seeds are controlled, LLM responses can be cached, and the local `stub` client enables zero-cost deterministic smoke tests. For real LLM experiments, use `--cache-dir` to avoid repeated API calls for identical prompts.

**中文。** 本项目强调实验逻辑透明：随机种子可控，LLM response 可缓存，本地 `stub` 客户端可用于零成本确定性 smoke test。使用真实 LLM 时，建议通过 `--cache-dir` 缓存相同 prompt 的结果，避免重复调用 API。

## Citation

If you use this repository in academic work, please cite the paper:

```bibtex
@inproceedings{llm_xapp_2026,
  title     = {LLM-xApp: A Large Language Model Empowered Radio Resource Management xApp for 5G O-RAN},
  booktitle = {IEEE INFOCOM Workshop},
  year      = {2026},
  note      = {Accepted}
}
```

正式出版信息公布后，建议将 BibTeX 条目更新为会议官方版本。

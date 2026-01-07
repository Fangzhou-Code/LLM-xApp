# RAN-LLM-xApp

本项目是基于论文《LLM-xApp: A Large Language Model Empowered Radio Resource Management xApp for 5G O-RAN》的思想与指标定义，构建的**纯本地 Python 合成仿真研究原型**，用于在可控环境下验证/扩展论文方法并开展创新实验；**不是对原论文 testbed 的严格复现**，也不追求数值逐点对齐（输出曲线主要呈现同类信息量与相近趋势形态）。

关键点：
- **不依赖** OAIC testbed / srsRAN / O-RAN 组件
- 实现并扩展论文关键定义：Utility(式(1)(2))、Reliability(滑窗 Tw)、动作到 PRB 映射(式(7) + 超预算修正)、评价函数(式(8), 默认 `g(x)=-x^2`)
- 在论文方法基础上加入工程化/算法改动（如 RealScore-driven TNAS、多模型对比与更稳定的度量/惩罚项），便于研究与创新验证
- 实现 4 个方法：`equal` / `random` / `proportional` / `tnas`（Top-N Action Sampling）

## 安装

建议使用虚拟环境：
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

## 运行（CLI）

命令入口（推荐）：  
```bash
python3 -m scripts.run_experiments --methods all --seed 0 --out outputs/multi_llm \
  --llm-runs openai:gpt-4o deepseek:deepseek-v3.2 google:gemini
```
该指令会把 `equal/random/proportional` 与 `tnas` 全部跑一遍，而各个 `--llm-runs` 模型会被当成独立的 `tnas_*` 变体一起对比，便于对齐论文中的 multi-LLM 实验设置并开展对比/创新分析。

其他常见案例：
```bash
# 只跑 tnas（用纯本地 stub）
python3 -m scripts.run_experiments --methods tnas --seed 0 --out outputs/tnas_only \
  --llm-runs stub:stub

# 跑 equal + proportional
python3 -m scripts.run_experiments --methods equal proportional --seed 0 --out outputs/eq_prop

# 跑全部 base + tnas
python3 -m scripts.run_experiments --methods all --seed 0 --out outputs/all_stub \
  --llm-runs stub:stub

# 跑全部方法（包含 tnas；tnas 需要 --llm-runs）
python3 -m scripts.run_experiments --methods all2 --seed 0 --out outputs/all2_stub \
  --llm-runs stub:stub
```

在线 LLM（可选）：
```bash
python3 -m scripts.run_experiments --methods tnas --seed 0 --out outputs/tnas_openai \
  --llm-runs openai:gpt-4o
python3 -m scripts.run_experiments --methods tnas --seed 0 --out outputs/tnas_deepseek \
  --llm-runs deepseek:deepseek-v3.2
python3 -m scripts.run_experiments --methods tnas --seed 0 --out outputs/tnas_google \
  --llm-runs google:gemini-3-pro
```

说明：
- `--model` 不填时会按 `--provider` 自动选择默认：OpenAI→`gpt-4o-mini`，DeepSeek→`deepseek-chat`，Google→`gemini-3-pro`，stub→`stub`。
- `--llm-runs` 的每项支持两种写法：
  - `provider:model`（显式指定 provider）
  - `model`（使用 `--provider` 作为默认 provider）
- 当提供 `--llm-runs` 且 `--methods` 未包含 `tnas` 时，脚本会自动追加一次 `tnas`（改进版），并为每个 `--llm-runs` 变体生成独立的 `tnas_*` 结果。
- `--methods all` 仅跑 equal/random/proportional；不过如果你同时传 `--llm-runs`，脚本会隐式再加一次 `tnas`（改进版）并为每个 LLM 生成独立 `tnas_*` 结果。
- 若启用 `--llm-runs`，TNAS 变体会产出 `timeseries_tnas_<provider>_<model>.csv`（文件名会自动做安全化）
- 当前的 `tnas` 默认使用 RealScore-driven 版本（即上面提到的“改进后的 TNAS”）；如需切回 proxy score 排序，可将 `ExperimentConfig.tnas_use_real_score` 设为 `false`（目前需通过修改代码中的配置默认值实现）。
- 为减少干扰，critic 在置信度不足时会回退到近似比例分配（见 `tnas_confidence_threshold`）；这可避免某些 slot 中 abrupt 的 UE1/UE2 变化。

## TNAS 方法（Top-N Action Sampling）

`tnas` 的核心思路是“**大模型提案 + 本地可控重排序**”：让 LLM 先给出一组多样候选动作，再用本地评分器选择要执行的动作，避免把最终决策完全交给 LLM。

流程（每个重配置 slot）：
- **观测构造**：用最近窗口的实测吞吐（`mean/last hat_sigma`）、当前需求 `sigma1/2`、当前 PRB 等构造 prompt（`ran_llm_xapp/policies/tnas.py`、`ran_llm_xapp/prompts.py`）。
- **LLM 生成候选**：输出 Top‑N 个 `(a1,a2)`（动作范围 `[1,128]`），解析失败会触发一次修复提示，仍失败则回退到近似比例分配（`ran_llm_xapp/policies/tnas.py`）。
- **动作映射与选择**：将 `(a1,a2)` 按论文式(7)映射到 `(prb1,prb2)` 并做超预算修正（`ran_llm_xapp/metrics.py:action_to_prbs`），再用本地评分选择最终动作。

本地评分/重排序有两种模式（默认启用 RealScore）：
- **RealScore-driven（默认）**：用轻量 critic 在线学习 `V_k_soft` 的真实反馈并对候选打分（`RealScoreCritic`），兼顾探索（`tnas_real_score_explore`）与回退（`tnas_confidence_threshold`）。
- **Proxy score（可选）**：用确定性 proxy 评分函数直接对候选打分（`ran_llm_xapp/metrics.py:score_allocation_proxy`）。

常用配置项（见 `ran_llm_xapp/config.py`）：
- `tnas_top_n`：每次重配置参与比较的候选数量
- `llm_max_tokens` / `llm_parse_retry`：减少 JSON 截断与解析失败
- `tnas_use_real_score` / `tnas_real_score_lr` / `tnas_real_score_explore` / `tnas_confidence_threshold`：RealScore 模式相关

LLM 缓存目录（避免重复计费）可通过 `--cache-dir` 指定；同 prompt 命中缓存会直接复用 response。

## 配置 OpenAI / DeepSeek / Google Key

推荐使用 `.env`（更方便且不会污染全局 shell 环境）：
1) 在项目根目录编辑 `.env`，填写：
   - `OPENAI_API_KEY=...`
   - `OPENAI_BASE_URL=...`（必填，本项目不会使用默认 URL）
   - `DEEPSEEK_API_KEY=...`
   - `DEEPSEEK_BASE_URL=...`（必填，本项目不会使用默认 URL）
   - `GOOGLE_API_KEY=...`
   - `GOOGLE_BASE_URL=...`（必填，本项目不会使用默认 URL）

程序启动时会自动加载项目根目录或当前目录下的 `.env`（不会覆盖已存在的系统环境变量）。

例如：
```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GOOGLE_API_KEY="..."
```

无 key 或 base_url 时仍可运行：若你选择 `--provider openai|deepseek|google` 但未提供对应的 key/base_url，程序会提示并自动退化到 `stub`（启发式 LLM），保证可跑通并出图。

若你在 macOS 上看到类似 `SSL: UNEXPECTED_EOF_WHILE_READING` 的报错，常见原因是你当前的 `python3` 来自 CommandLineTools（SSL 后端是 LibreSSL，兼容性较差）。建议改用 Homebrew 或 python.org 的 Python（SSL 后端是 OpenSSL）再运行。

如果你的 Python SSL 后端已经是 OpenSSL 仍然报 `UNEXPECTED_EOF_WHILE_READING`，通常是网关/代理在 TLS 1.3 上不兼容导致连接被提前断开。可尝试在 `.env` 里加一行强制 TLS 1.2：
- DeepSeek：`DEEPSEEK_TLS_FORCE_TLS12=1`
- OpenAI：`OPENAI_TLS_FORCE_TLS12=1`
- Google：`GOOGLE_TLS_FORCE_TLS12=1`
- 或全局：`LLM_TLS_FORCE_TLS12=1`

## 输出文件说明

每次运行输出到 `--out` 指定目录，至少包含：
- `timeseries_<method>.csv`：包含 `t, method, prb1, prb2, sigma1, sigma2, eff_cap1, eff_cap2, shortfall1, shortfall2, prb2_min_est, waste, penalty, V_k_soft, hat_sigma1, hat_sigma2, u1, u2`
  - legacy：`theta1, theta2, sys_theta`（这里的 `θ` 是 **outage fraction**，越小越好；保留用于兼容）
  - 推荐使用：`outage_theta1, outage_theta2, system_outage_theta, reliability1, reliability2, system_reliability`，且满足 `reliability = 1 - outage_theta`
  - 当使用 `--llm-runs` 跑多个 TNAS 变体时，会额外生成 `timeseries_tnas_<provider>_<model>.csv`
- 图4：
  - 每个方法/变体都会单独输出 `fig4_<method>.png`
  - 额外输出 `fig4.png`：把本次运行产生的所有方法/变体放到**同一张组合图**（支持多个 TNAS 变体，子图数量会随之增加）
- 图5：
  - `fig5a_sys_utility.png`
  - `fig5b_sys_reliability.png`
  - `fig5b_outage_theta.png`（debug：outage θ，越低越好）
  - `fig5c_avg_utility.png`
  - `fig5d_avg_reliability.png`
  - `fig5d_outage_theta.png`（debug：outage θ，越低越好）
- `config_used.yaml`（若未安装 PyYAML 则为 `config_used.json`）

## 时间线与 system 指标

- **时间线**（默认配置对齐论文图4叙述）：
  - `t∈[0,100)`：未启用 slicing 控制的默认阶段，固定 PRB 分配（默认 `prb1=96, prb2=32`）→ UE1≈30 Mbps、UE2≈10 Mbps
  - `t∈[100,200)`：slice init + 初始化均分阶段，固定 `prb1=prb2=64` → UE1≈20 Mbps、UE2≈10 Mbps
  - `t≥200`：方法策略生效并进入对比阶段；同时启用 **demand schedule**（默认：`sigma1` 在 `t=200` 变为 30、在 `t=400` 变为 45；`sigma2` 保持 10），用于制造 “可行→不可行” 的切换与 trade-off
- **system 指标聚合方式**：
  - `system_utility(t)`：按 β 权重加权平均，默认 `w1 = beta1 / (beta1 + beta2)`
  - `system_outage_theta(t) = mean(outage_theta1(t), outage_theta2(t))`（简单平均）
  - `system_reliability(t) = 1 - system_outage_theta(t)`
- **soft score（V_k_soft）**：在论文 Eq.(8) 的 `V_k` 基础上加入短缺惩罚（默认 `t>=200` 启用）：
  - `eff_cap_s = min(sigma_s, cap_s_hard)`（`cap_s_hard=None` 视为 +inf）
  - `shortfall_s = max(0, eff_cap_s - mean_hat_sigma_s)`
  - `V_k_soft = V_k - lambda1*shortfall1^p - lambda2*shortfall2^p`（默认 `p=2, lambda1=6, lambda2=1`）
- **outage θ（违约占比）**：按论文定义为滑窗内 `u_s^τ <= u_th_s` 的比例（范围[0,1]，越小越好；详见 `ran_llm_xapp/metrics.py`）。
- **reliability（对外展示）**：`reliability = 1 - outage_theta`（范围[0,1]，越大越好）。图5b/5d 默认画的是 reliability。
- 图5 的 time-averaged 统计默认在 **t ∈ [baseline_start_time, T_end]** 上计算（强调 t=200 后策略差异）。

## 运行测试

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

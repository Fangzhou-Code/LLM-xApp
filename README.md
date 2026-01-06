# ran-llm-xApp (Pure Python, Synthetic Repro)

本项目用**纯本地 Python 合成仿真**复现论文《LLM-xApp: A Large Language Model Empowered Radio Resource Management xApp for 5G O-RAN》的核心流程、指标定义与对比曲线（图4/图5同类信息量与相近趋势形态）。

关键点：
- **不依赖** OAIC testbed / srsRAN / O-RAN 组件
- 严格实现论文关键定义：Utility(式(1)(2))、Reliability(滑窗 Tw)、动作到 PRB 映射(式(7) + 超预算修正)、评价函数(式(8), 默认 `g(x)=-x^2`)
- 实现 5 个方法：`equal` / `random` / `proportional` / `tnas`（Top-N Action Sampling）/ `cem`（Budgeted CEM）

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
该指令会把 `equal/random/proportional` 与 `tnas` 全部跑一遍，而各个 `--llm-runs` 模型会被当成独立的 `tnas_*` 变体一起对比，便于复现 paper 里的 multi-LLM 结果。

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
```

在线 LLM（可选）：
```bash
python3 -m scripts.run_experiments --methods tnas --seed 0 --out outputs/tnas_openai \
  --llm-runs openai:gpt-4o
python3 -m scripts.run_experiments --methods tnas --seed 0 --out outputs/tnas_deepseek \
  --llm-runs deepseek:deepseek-v3.2
```

说明：
- `--model` 不填时会按 `--provider` 自动选择默认：OpenAI→`gpt-4o-mini`，DeepSeek→`deepseek-chat`，stub→`stub`。
- `--llm-runs` 的每项支持两种写法：
  - `provider:model`（显式指定 provider）
  - `model`（使用 `--provider` 作为默认 provider）
- 当同时提供 `--llm-runs` 且未显式选择 `tnas`（如仅 `--methods all`），脚本会自动把 `tnas` 加回去，这样 `gpt` 和 `deepseek` 变体仍会跑一次改进后的 TNAS。
- `--methods all` 仅跑 equal/random/proportional；不过如果你同时传 `--llm-runs`，脚本会隐式再加一次 `tnas`（改进版）并为每个 LLM 生成独立 `tnas_*` 结果。
- 若启用 `--llm-runs`，请确保 `--methods` 中包含 `tnas` 或 `all`（因为我们会自动补上）；TNAS 变体仍会产出 `timeseries_tnas_<provider>_<model>.csv`。（文件名会自动做安全化）
- 当前的 `tnas` 已切换为 RealScore-driven 版本（即上面提到的“改进后的 TNAS”），不会再使用旧的 proxy score 排序。
- 为减少干扰，critic 只在信号（特征范数）足够强时输出、否则会回退到近似比例分配；这可避免某些 slot 中 abrupt 的 UE1/UE2 变化。

LLM 缓存目录（避免重复计费）可通过 `--cache-dir` 指定；同 prompt 命中缓存会直接复用 response。

## 配置 OPENAI / DeepSeek Key

推荐使用 `.env`（更方便且不会污染全局 shell 环境）：
1) 在项目根目录编辑 `.env`，填写：
   - `OPENAI_API_KEY=...`
   - `OPENAI_BASE_URL=...`（必填，本项目不会使用默认 URL）
   - `DEEPSEEK_API_KEY=...`
   - `DEEPSEEK_BASE_URL=...`（必填，本项目不会使用默认 URL）

程序启动时会自动加载项目根目录或当前目录下的 `.env`（不会覆盖已存在的系统环境变量）。

例如：
```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

无 key 或 base_url 时仍可运行：若你选择 `--provider openai|deepseek` 但未提供对应的 key/base_url，程序会提示并自动退化到 `stub`（启发式 LLM），保证可跑通并出图。

若你在 macOS 上看到类似 `SSL: UNEXPECTED_EOF_WHILE_READING` 的报错，常见原因是你当前的 `python3` 来自 CommandLineTools（SSL 后端是 LibreSSL，兼容性较差）。建议改用 Homebrew 或 python.org 的 Python（SSL 后端是 OpenSSL）再运行。

如果你的 Python SSL 后端已经是 OpenSSL 仍然报 `UNEXPECTED_EOF_WHILE_READING`，通常是网关/代理在 TLS 1.3 上不兼容导致连接被提前断开。可尝试在 `.env` 里加一行强制 TLS 1.2：
- DeepSeek：`DEEPSEEK_TLS_FORCE_TLS12=1`
- OpenAI：`OPENAI_TLS_FORCE_TLS12=1`
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
- **system 指标聚合方式**：从 `t=0` 起对 UE1/UE2 两个 slice 做**简单平均**（未加权）
  - `system_utility(t) = w1*u1(t) + (1-w1)*u2(t)`，其中默认 `w1 = beta1 / (beta1 + beta2)`
  - `system_outage_theta(t) = mean(outage_theta1(t), outage_theta2(t))`（仍为简单平均）
  - `system_reliability(t) = 1 - system_outage_theta(t)`
- **soft score（V_k_soft）**：在论文 Eq.(8) 的 `V_k` 基础上加入短缺惩罚（默认 `t>=200` 启用）：
  - `eff_cap_s = min(sigma_s, cap_s_hard)`（`cap_s_hard=None` 视为 +inf）
  - `shortfall_s = max(0, eff_cap_s - mean_hat_sigma_s)`
  - `V_k_soft = V_k - lambda1*shortfall1^p - lambda2*shortfall2^p`（默认 `p=2, lambda1=0.1, lambda2=6.0`）
- **outage θ（违约占比）**：按论文定义为滑窗内 `u_s^τ <= u_th_s` 的比例（范围[0,1]，越小越好；详见 `ran_llm_xapp/metrics.py`）。
- **reliability（对外展示）**：`reliability = 1 - outage_theta`（范围[0,1]，越大越好）。图5b/5d 默认画的是 reliability。
- 图5 的 time-averaged 统计默认在 **t ∈ [baseline_start_time, T_end]** 上计算（强调 t=200 后策略差异）。

## 运行测试

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

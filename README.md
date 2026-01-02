# ran-llm-xApp (Pure Python, Synthetic Repro)

本项目用**纯本地 Python 合成仿真**复现论文《LLM-xApp: A Large Language Model Empowered Radio Resource Management xApp for 5G O-RAN》的核心流程、指标定义与对比曲线（图4/图5同类信息量与相近趋势形态）。

关键点：
- **不依赖** OAIC testbed / srsRAN / O-RAN 组件
- 严格实现论文关键定义：Utility(式(1)(2))、Reliability(滑窗 Tw)、动作到 PRB 映射(式(7) + 超预算修正)、评价函数(式(8), 默认 `g(x)=-x^2`)、以及 LLM-OPRO（Algorithm 1 的历史排序 + 温度衰减 + in-context）
- 实现 4 个方法：`random` / `equal` / `proportional` / `llm`（LLM-OPRO）

## 安装

建议使用虚拟环境：
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

## 运行（CLI）

命令入口：
```bash
python3 -m scripts.run_experiments --methods all --seed 0 --out outputs/run1
```

至少 3 个 methods 子集示例：
```bash
# 只跑 llm
python3 -m scripts.run_experiments --methods llm --seed 0 --out outputs/llm_only --provider stub

# 跑 equal + proportional
python3 -m scripts.run_experiments --methods equal proportional --seed 0 --out outputs/eq_prop

# 跑全部（默认推荐：stub，无需 key 也能出图）
python3 -m scripts.run_experiments --methods all --seed 0 --out outputs/all_stub --provider stub
```

在线 LLM（可选）：
```bash
python3 -m scripts.run_experiments --methods llm --seed 0 --out outputs/llm_openai --provider openai --model gpt-4o-mini
python3 -m scripts.run_experiments --methods llm --seed 0 --out outputs/llm_deepseek --provider deepseek --model deepseek-chat
```

一条指令同时跑多个模型（会把每个模型视为一个独立的 `llm_*` 变体一起对比）：
```bash
python3 -m scripts.run_experiments --methods all --seed 0 --out outputs/multi_llm \
  --llm-runs openai:gpt-4o-mini deepseek:deepseek-chat stub:stub
```

说明：
- `--model` 不填时会按 `--provider` 自动选择默认：OpenAI→`gpt-4o-mini`，DeepSeek→`deepseek-chat`，stub→`stub`。
- `--llm-runs` 的每项支持两种写法：
  - `provider:model`（显式指定 provider）
  - `model`（使用 `--provider` 作为默认 provider）
- 若启用 `--llm-runs`，则 `llm` 方法会按列表逐个运行，并输出 `timeseries_llm_<provider>_<model>.csv`（文件名会自动做安全化）。

LLM 缓存目录（避免重复计费）可通过 `--cache-dir` 指定；同 prompt 命中缓存会直接复用 response。

## 配置 OPENAI / DeepSeek Key

推荐使用 `.env`（更方便且不会污染全局 shell 环境）：
1) 在项目根目录编辑 `.env`，填写：
   - `OPENAI_API_KEY=...`
   - `DEEPSEEK_API_KEY=...`
   - （可选）`OPENAI_BASE_URL=...` / `DEEPSEEK_BASE_URL=...`

程序启动时会自动加载项目根目录或当前目录下的 `.env`（不会覆盖已存在的系统环境变量）。

例如：
```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

无 key 时仍可运行：若你选择 `--provider openai|deepseek` 但未提供 key，程序会提示并自动退化到 `stub`（启发式 LLM），保证可跑通并出图。

## 输出文件说明

每次运行输出到 `--out` 指定目录，至少包含：
- `timeseries_<method>.csv`：包含 `t, method, prb1, prb2, hat_sigma1, hat_sigma2, sigma1, sigma2, u1, u2, theta1, theta2, sys_u, sys_theta`
  - 当使用 `--llm-runs` 跑多个 LLM 时，会额外生成 `timeseries_llm_<provider>_<model>.csv`
- 图4：
  - 每个方法/变体都会单独输出 `fig4_<method>.png`
  - 若同时具备 `random/equal/proportional` 且至少跑了 1 个 `llm` 变体：输出 `fig4.png`（2x2：random/equal/proportional/第一个 llm 变体）
  - 若跑了多个 llm 变体：额外输出 `fig4_grid_<llm_variant>.png`（每个 llm 变体各一张 2x2 grid）
- 图5：
  - `fig5a_sys_utility.png`
  - `fig5b_sys_reliability.png`
  - `fig5c_avg_utility.png`
  - `fig5d_avg_reliability.png`
- `config_used.yaml`（若未安装 PyYAML 则为 `config_used.json`）

## 关于 “t<100 只有 S1 存在” 与 system 指标的处理选择

- **t < 100s**：仅 S1(UE1) 存在；UE2 的 `hat_sigma2` 记为 0（CSV 中如此），`u2/theta2` 记为 `NaN`，并且 system 指标只对**已存在的 slice**做简单平均（因此 t<100 时 `sys_u = u1`，`sys_theta = theta1`）。为了图形观感更接近论文，图4 会在 `t<100` 段落不绘制 UE2 曲线（避免出现一条长时间的 0 平线）。
- **system utility / system reliability**：对当前时刻已存在 slice 的 `u_s^t` / `θ_s^t` 做**简单平均**（未加权）。
- **Reliability θ**：按论文定义为滑窗内 `u_s^τ <= u_th_s` 的比例（本实现将窗口越界部分截断，并除以有效样本数；详见 `ran_llm_xapp/metrics.py`）。
- 图5 的 time-averaged 统计默认在 **t ∈ [baseline_start_time, T_end]** 上计算（强调 t=200 后策略差异）。

## 运行测试

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

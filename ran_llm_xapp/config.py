from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration (defaults are 'out-of-the-box' runnable).

    Notes on defaults (chosen to match the paper-like shapes):
    - UE1/UE2 exist for the whole time horizon t∈[0,T_end].
    - The three-stage allocation timeline is implemented in the runner:
        (i)  0~slice_init_time: fixed PRB split (defaults 96/32) → ~30/~10 Mbps
        (ii) slice_init_time~baseline_start_time: evenly split (64/64) → ~20/~10 Mbps
        (iii) baseline_start_time~end: method policy takes effect (equal/random/proportional/tnas/oracle)
    - UE2 has a tight application-layer cap (cap2≈10 Mbps) and high per-PRB efficiency,
      so that TNAS/proportional can allocate fewer PRBs to UE2 while still keeping it near 10 Mbps.
    """

    # Time
    T_end: int = 800
    dt: int = 1
    slice_init_time: int = 100
    baseline_start_time: int = 200

    # PRB budget
    R_total: int = 128
    R_eff_pre: int = 128  # kept for backward-compat; effective budget is `R_total` by default.

    # Pre-slicing stage fixed PRB split (0~slice_init_time)
    pre_slice_prb1: int = 96
    pre_slice_prb2: int = 32

    # Requested rates (Mbps)
    sigma1: float = 40.0
    sigma2: float = 10.0
    # Demand schedule for the allocation phase (time-varying σ_s^t).
    # Each entry is: (start_time_s, sigma1_mbps, sigma2_mbps), piecewise-constant.
    #
    # Default schedule requested by the experiment spec:
    # - t < 200         : sigma1=40, sigma2=10
    # - 200 <= t < 400  : sigma1=30, sigma2=10   (feasible)
    # - t >= 400        : sigma1=45, sigma2=10   (infeasible)
    demand_schedule: List[Tuple[int, float, float]] = dataclasses.field(
        default_factory=lambda: [(0, 40.0, 10.0), (200, 30.0, 10.0), (400, 45.0, 10.0)]
    )

    # Utility params (Table I)
    a: float = 0.9
    b: float = 6.5
    c: float = 5.0
    u_th1: float = 0.7
    u_th2: float = 0.3

    # Reliability window (seconds/samples)
    Tw: int = 20

    # Control interval (seconds)
    reconfig_interval: int = 5

    # System curve smoothing (seconds/samples)
    smooth_window: int = 10

    # Evaluation function params (Table I)
    beta1: float = 2.5
    beta2: float = 1.0
    gamma1: float = 2000.0
    gamma2: float = 2000.0
    g_name: str = "neg_square"  # default g(x) = -x^2

    # LLM-OPRO params (Table I + reasonable defaults)
    Tem_max: float = 1.0
    Tem_min: float = 0.3
    Tem_delta: float = 0.05
    in_context_top_n: int = 5
    in_context_n_examples: int = 3
    # TNAS outputs a JSON object with Top-N candidates (each with a short "reason").
    # 150 tokens is often insufficient and can lead to truncated / invalid JSON.
    llm_max_tokens: int = 400
    llm_timeout_s: int = 60
    llm_parse_retry: int = 1  # retry once with repair prompt
    tnas_top_n: int = 8

    # Budgeted CEM baseline (black-box elite sampling)
    cem_iters: int = 1
    cem_samples: int = 8
    cem_elite_k: int = 2
    cem_step: int = 8
    cem_alpha: float = 0.5

    # Synthetic environment model (tunable)
    eff1_mbps_per_prb: float = 0.305  # ~64 PRB -> 19.5 Mbps; ~96 PRB -> 29.3 Mbps; 128 PRB -> 39.0 Mbps
    eff2_mbps_per_prb: float = 0.5
    # Hard caps (Mbps). Effective target is min(demand, hard_cap); None means +inf.
    cap1_hard_mbps: Optional[float] = 45.0
    cap2_hard_mbps: Optional[float] = 10.0
    ar_rho: float = 0.9
    ar_eps_std1: float = 0.55
    ar_eps_std2: float = 0.22
    meas_std1: float = 0.15
    meas_std2: float = 0.10

    # "Soft two-stage" waste penalty (only enabled for t >= baseline_start_time)
    lambda_waste: float = 1.0
    waste_eps: float = 1e-6
    # Soft-score (shortfall) penalty settings.
    use_soft_score: bool = True
    soft_p: int = 2
    lambda1: float = 6
    lambda2: float = 1
    soft_enable_time: int = 200
    schedule_margin_prb: int = 8
    # System reliability (severity-weighted) settings
    # mode: 'legacy' | 'severity' | 'both' (default 'both' writes both legacy and new cols)
    sys_r_mode: str = "both"
    # severity power p (shortfall^p)
    sys_reliability_p: int = 2
    # numerical stability epsilon for denominator
    sys_reliability_eps: float = 1e-6
    # optional override lambdas for severity formula; if None, fall back to lambda1/lambda2
    sys_lambda1: Optional[float] = None
    sys_lambda2: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        known = {field.name for field in dataclasses.fields(cls)}
        filtered = {k: v for k, v in dict(data).items() if k in known}
        return cls(**filtered)

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        return dataclasses.replace(self, **kwargs)

    def sigma_at(self, t: int) -> Tuple[float, float]:
        """Return (sigma1(t), sigma2(t)) from demand_schedule (piecewise-constant)."""

        tt = int(t)
        schedule: Sequence[Sequence[object]] = self.demand_schedule or [(0, self.sigma1, self.sigma2)]
        best_t0 = None
        best_s1 = float(self.sigma1)
        best_s2 = float(self.sigma2)
        for entry in schedule:
            if len(entry) < 3:
                continue
            try:
                t0 = int(entry[0])
                s1 = float(entry[1])
                s2 = float(entry[2])
            except Exception:
                continue
            if t0 <= tt and (best_t0 is None or t0 >= best_t0):
                best_t0 = t0
                best_s1 = s1
                best_s2 = s2
        return best_s1, best_s2


def _try_import_yaml():
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    return yaml


def load_config(path: str | Path) -> ExperimentConfig:
    """Load config from YAML/JSON; unknown keys are ignored."""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        yaml = _try_import_yaml()
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; cannot read YAML config.")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML config must be a mapping/object.")
        return ExperimentConfig.from_dict(data)

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON config must be an object.")
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, out_path: str | Path) -> Path:
    """Save config to YAML if suffix is .yaml/.yml (and PyYAML exists), else JSON."""

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".yaml", ".yml"}:
        yaml = _try_import_yaml()
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; cannot write YAML config.")
        p.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
        return p

    p.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
    return p


def save_config_prefer_yaml(config: ExperimentConfig, out_dir: str | Path) -> Path:
    """Write `config_used.yaml` if possible, otherwise `config_used.json`."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml = _try_import_yaml()
    if yaml is not None:
        p = out_dir / "config_used.yaml"
        p.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
        return p
    p = out_dir / "config_used.json"
    p.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
    return p

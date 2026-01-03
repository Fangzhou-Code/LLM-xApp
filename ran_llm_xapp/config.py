from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration (defaults are 'out-of-the-box' runnable).

    Notes on defaults (chosen to match the paper-like shapes):
    - UE1/UE2 exist for the whole time horizon t∈[0,T_end].
    - The three-stage allocation timeline is implemented in the runner:
        (i)  0~slice_init_time: fixed PRB split (defaults 96/32) → ~30/~10 Mbps
        (ii) slice_init_time~baseline_start_time: evenly split (64/64) → ~20/~10 Mbps
        (iii) baseline_start_time~end: method policy takes effect (equal/proportional/random/llm)
    - UE2 has a tight application-layer cap (cap2≈10 Mbps) and high per-PRB efficiency,
      so that llm/proportional can allocate fewer PRBs to UE2 while still keeping it near 10 Mbps.
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

    # Utility params (Table I)
    a: float = 0.9
    b: float = 6.5
    c: float = 5.0
    u_th1: float = 0.6
    u_th2: float = 0.96

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
    llm_max_tokens: int = 150
    llm_timeout_s: int = 60
    llm_parse_retry: int = 1  # retry once with repair prompt

    # Synthetic environment model (tunable)
    eff1_mbps_per_prb: float = 0.3125  # 64 PRB -> 20 Mbps; 96 PRB -> 30 Mbps; 128 PRB -> 40 Mbps
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

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        known = {field.name for field in dataclasses.fields(cls)}
        filtered = {k: v for k, v in dict(data).items() if k in known}
        return cls(**filtered)

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        return dataclasses.replace(self, **kwargs)


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

"""Harness configuration: the independent variables of the study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .baselines import BaselineRules, SearchBaseline
from .controllers import MockController

TOKEN_DIVISOR = 4  # chars/token ESTIMATE; the metric is labelled as such

# Control-layer placement. The three levels differ ONLY in the two properties
# that define an O-RAN control loop: how often the loop may act (cadence) and how
# stale the telemetry it decides on is. No per-layer tools, no privileged access.
LAYERS = {
    "near_rt":    {"cadence": 1, "staleness": 0, "label": "near-RT RIC (xApp)"},
    "non_rt":     {"cadence": 2, "staleness": 1, "label": "non-RT RIC (rApp)"},
    "smo_intent": {"cadence": 3, "staleness": 2, "label": "SMO / intent layer"},
}
CONTROLLERS = ("mock", "baseline_rules", "baseline_search", "llm")


@dataclass
class HarnessConfig:
    tool_granularity: str = "fine"
    state_management: str = "scratchpad"
    control: str = "reactive"
    guardrails: str = "strict"
    retry: str = "retry_rollback"
    controller: str = "mock"
    layer: str = "near_rt"
    # 多层部署：((cadence, staleness), ...)。为 None 时走原有单层路径，
    # 行为与改动前**逐位一致**。设定后 harness 会让每个子智能体各按自己的
    # 节奏与新鲜度行动（见 aixnet/multilayer.py）。
    layer_specs: tuple | None = None
    # 多层的**角色分工**：每层允许的操作类别（frozenset 或 None=不限）。
    # 用于区分"冗余副本"（各层做同样的事）与"分工"（慢层只设策略、快层只动物理）。
    layer_op_roles: tuple | None = None
    # 每步最多应用几个操作（仅 mock 控制器）。默认 3 = MockController 的原默认值，
    # 改动前行为不变。用于**分离"动作更多"与"协作"**的对照实验。
    mock_batch: int | None = None
    # 遥测预测器：'none'（默认，用陈旧观测）| 'oracle'（用当前真值）| 'delta'（线性外推）
    # 用途：分离"慢层的代价来自信息陈旧"与"来自行动机会少"。
    predictor: str = "none"
    # 预测噪声幅度（相对值）。0 = 完美外推；0.1 = 每个字段 ±10% 均匀噪声。
    # 用途：把"确定性孪生上外推几乎精确"这一局限变成**量化曲线**——
    # 预测要多准才够用？
    predictor_noise: float = 0.0

    def cid(self) -> str:
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer {self.layer!r}")
        return (f"{self.tool_granularity[:2]}|{self.state_management[:4]}|"
                f"{self.control[:4]}|{self.guardrails[:4]}|{self.retry}|"
                f"{self.layer[:7]}")

    @property
    def cadence(self) -> int:
        return LAYERS[self.layer]["cadence"]

    @property
    def staleness(self) -> int:
        return LAYERS[self.layer]["staleness"]

    def as_dict(self) -> dict:
        return asdict(self)


CONTROLLERS = ("mock", "baseline_rules", "baseline_search", "llm")


def make_controller(cfg: "HarnessConfig", seed: int):
    # 多层：包一个复合控制器，每个子智能体对应一组 (cadence, staleness)
    if cfg.layer_specs and len(cfg.layer_specs) > 1:
        from .multilayer import CompositeController
        batch = None
        if cfg.controller == "mock":
            import inspect as _i
            if "batch" in _i.signature(MockController.__init__).parameters:
                pass
        return CompositeController(cfg.controller, cfg.control, seed,
                                   len(cfg.layer_specs), batch)
    if cfg.controller == "mock":
        if getattr(cfg, "mock_batch", None):
            return MockController(cfg.control, seed, batch=cfg.mock_batch)
        return MockController(cfg.control, seed)
    if cfg.controller == "baseline_rules":
        return BaselineRules()
    if cfg.controller == "baseline_search":
        return SearchBaseline()
    if cfg.controller == "llm":
        from .llm_client import LLMController, OpenAICompatClient
        return LLMController(OpenAICompatClient(), cfg.tool_granularity)
    if cfg.controller == "rl":
        from .controllers import NullController
        return NullController()
    raise ValueError(f"unknown controller {cfg.controller!r}")



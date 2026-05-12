from faro.agents.base import (
    Agent,
    Condition,
    IntraExperimentAgent,
    InterPhaseAgent,
    PreExperimentAgent,
)
from faro.agents.bo_dose_response import DoseResponseBO
from faro.agents.bo_oscillation import OscillationBO
from faro.agents.composed import ComposedAgent
from faro.agents.condition_monitor import ConditionMonitorAgent
from faro.agents.fov_finder import FOVCondition, FOVFinderAgent

# BoTorch-based agents (lazy import — only available when botorch is installed)
try:
    from faro.agents.bo_botorch import BOptBoTorch
    from faro.agents.bo_botorch_oscillation import OscillationBOBoTorch
except ImportError:
    pass

__all__ = [
    "Agent",
    "BOptBoTorch",
    "ComposedAgent",
    "Condition",
    "ConditionMonitorAgent",
    "DoseResponseBO",
    "FOVCondition",
    "FOVFinderAgent",
    "IntraExperimentAgent",
    "InterPhaseAgent",
    "OscillationBO",
    "OscillationBOBoTorch",
    "PreExperimentAgent",
]

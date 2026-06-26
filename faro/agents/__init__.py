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
from faro.agents.fov_condition_monitor import FOVConditionMonitorAgent
from faro.agents.fov_density import (
    FovDensityScorer,
    build_stage_montage,
    find_fov_windows,
)
from faro.agents.fov_finder import FOVCondition, FOVFinderAgent
from faro.agents.grid_fov_finder import GridFOVFinderAgent
from faro.agents.well_pattern import (
    FOVFinder,
    ResolvedWellPatterns,
    WellPattern,
    resolve_well_patterns,
    run_well_patterns,
    run_well_patterns_async,
)

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
    "FOVConditionMonitorAgent",
    "FOVFinder",
    "FOVFinderAgent",
    "FovDensityScorer",
    "GridFOVFinderAgent",
    "IntraExperimentAgent",
    "InterPhaseAgent",
    "build_stage_montage",
    "find_fov_windows",
    "OscillationBO",
    "OscillationBOBoTorch",
    "PreExperimentAgent",
    "ResolvedWellPatterns",
    "WellPattern",
    "resolve_well_patterns",
    "run_well_patterns",
    "run_well_patterns_async",
]

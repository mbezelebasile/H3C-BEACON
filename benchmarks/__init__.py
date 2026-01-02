"""
H3C-BEACON Benchmark Environments
=================================
Professional-grade multi-agent environments for MARL evaluation.

Supported Benchmarks:
- MPE (PettingZoo): simple_spread, simple_world_comm
- SMAC (StarCraft): 27m_vs_30m
- GRF (Football): academy_3_vs_1_with_keeper

Author: H3C-BEACON Research Team
"""

from .base_env import BaseEnv, PettingZooWrapper, SMACWrapper, GRFWrapper
from .simple_spread import SimpleSpreadEnv, make_simple_spread
from .simple_world_comm import SimpleWorldCommEnv, make_simple_world_comm
from .smac_scenarios import SMAC27mVs30mEnv, make_27m_vs_30m
from .grf_academy import Academy3vs1WithKeeperEnv, make_academy_3_vs_1_with_keeper

__all__ = [
    # Base classes
    'BaseEnv',
    'PettingZooWrapper',
    'SMACWrapper', 
    'GRFWrapper',
    # MPE environments
    'SimpleSpreadEnv',
    'SimpleWorldCommEnv',
    'make_simple_spread',
    'make_simple_world_comm',
    # SMAC environments
    'SMAC27mVs30mEnv',
    'make_27m_vs_30m',
    # GRF environments
    'Academy3vs1WithKeeperEnv',
    'make_academy_3_vs_1_with_keeper',
    # Registry
    'BENCHMARKS',
    'make_env',
]

# Environment registry
BENCHMARKS = {
    # MPE - Multi-Particle Environments
    'simple_spread': make_simple_spread,
    'simple_world_comm': make_simple_world_comm,
    # SMAC - StarCraft Multi-Agent Challenge
    '27m_vs_30m': make_27m_vs_30m,
    # GRF - Google Research Football
    'academy_3_vs_1_with_keeper': make_academy_3_vs_1_with_keeper,
}


def make_env(env_name: str, **kwargs) -> BaseEnv:
    """
    Create an environment by name.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional arguments passed to environment constructor
    
    Returns:
        Environment instance
    
    Available environments:
        MPE (PettingZoo):
        - simple_spread: 3 agents cover 3 landmarks (cooperative)
        - simple_world_comm: 2 good agents vs 4 adversaries (mixed)
        
        SMAC (StarCraft):
        - 27m_vs_30m: 27 Marines vs 30 Marines (large-scale)
        
        GRF (Football):
        - academy_3_vs_1_with_keeper: 3v1+GK football scenario
    """
    if env_name not in BENCHMARKS:
        available = list(BENCHMARKS.keys())
        raise ValueError(f"Unknown environment: {env_name}. Available: {available}")
    
    return BENCHMARKS[env_name](**kwargs)
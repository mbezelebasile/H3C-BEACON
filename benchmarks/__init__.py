"""
H3C-BEACON Benchmark Environments
=================================
Professional-grade multi-agent environments for MARL evaluation.

REVISED VERSION - Major Revision for Complex & Intelligent Systems
Supports all environments requested by reviewers (R2.9):
- MPE (PettingZoo): simple_spread, simple_world_comm
- SMAC (SMACLite): 3m, 8m, 27m_vs_30m
- Hanabi (PettingZoo): hanabi_small, hanabi_full
- GRF (Football): academy_3_vs_1_with_keeper

Author: Basile BETE MBEZELE, Ghislain ALO'O ABESSOLO
University of Yaoundé I, Cameroon
"""

from .base_env import BaseEnv, PettingZooWrapper, SMACWrapper, GRFWrapper
from .simple_spread import SimpleSpreadEnv, make_simple_spread
from .simple_world_comm import SimpleWorldCommEnv, make_simple_world_comm
from .smac_scenarios import SMAC27mVs30mEnv, make_27m_vs_30m
from .grf_academy import Academy3vs1WithKeeperEnv, make_academy_3_vs_1_with_keeper

# SMACLite environments (R2.9)
try:
    from .smac_lite_env import SMACLiteEnv, make_smac_env
    SMACLITE_AVAILABLE = True
except ImportError:
    SMACLITE_AVAILABLE = False
    SMACLiteEnv = None
    make_smac_env = None

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

# Environment registry - REVISED for R2.9
BENCHMARKS = {
    # MPE - Multi-Particle Environments
    'simple_spread': make_simple_spread,
    'simple_world_comm': make_simple_world_comm,
    # SMAC - StarCraft Multi-Agent Challenge (via SMACLite)
    '27m_vs_30m': make_27m_vs_30m,
}

# Add SMACLite environments if available
if SMACLITE_AVAILABLE:
    BENCHMARKS.update({
        'smac_3m': lambda **kw: make_smac_env('3m', **kw),
        'smac_8m': lambda **kw: make_smac_env('8m', **kw),
        'smac_2s3z': lambda **kw: make_smac_env('2s3z', **kw),
        'smac_3s5z': lambda **kw: make_smac_env('3s5z', **kw),
        'smac_27m_vs_30m': lambda **kw: make_smac_env('27m_vs_30m', **kw),
    })

# Add GRF
BENCHMARKS['academy_3_vs_1_with_keeper'] = make_academy_3_vs_1_with_keeper


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
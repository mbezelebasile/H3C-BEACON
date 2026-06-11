"""
H3C-BEACON Modules
==================

Core components:
- H3CTrainer: Main trainer (original)
- H3CTrainer_Revised: Updated trainer with reviewer modifications
- AAH: Adaptive Attention Hierarchy
- PCC: Probabilistic Coalition Coordinator
- CDEGA: Coalition-Driven Entropy-Guided Actor
- AUTO_HP: Automatic Hyperparameter Tuning
"""

from modules.AAH import AAH
from modules.PCC import PCC
from modules.CDEGA import CDEGA
from modules.AUTO_HP import AUTO_HP

# Original trainer
try:
    from modules.H3CTrainer import H3CTrainer
except ImportError:
    H3CTrainer = None
    print("Note: H3CTrainer.py not found")

# Revised trainer (with reviewer modifications)
try:
    from modules.H3CTrainer_Revised import (
        H3CTrainerRevised, 
        create_h3c_trainer,
        DynamicGraphAttention,
        BayesianBeliefFusion,
        AdaptiveCoalitionFormation,
        DualCritic,
        RTDPlusPlusElite,
    )
except ImportError:
    H3CTrainerRevised = None
    print("Note: H3CTrainer_Revised.py not found")

__version__ = '4.0-revision'
__all__ = [
    'AAH', 'PCC', 'CDEGA', 'AUTO_HP', 
    'H3CTrainer', 'H3CTrainerRevised', 
    'create_h3c_trainer',
    'DynamicGraphAttention',
    'BayesianBeliefFusion',
    'AdaptiveCoalitionFormation',
    'DualCritic',
    'RTDPlusPlusElite',
]
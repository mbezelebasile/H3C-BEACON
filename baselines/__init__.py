"""
H3C-BEACON Baselines
State-of-the-art MARL algorithms for comparison.
"""

from .base_trainer import BaseTrainer
from .mappo import MAPPOTrainer
from .ippo import IPPOTrainer
from .qmix import QMIXTrainer
from .vdn import VDNTrainer
from .coma import COMATrainer
from .facmac import FACMACTrainer

__all__ = [
    'BaseTrainer',
    'MAPPOTrainer',
    'IPPOTrainer',
    'QMIXTrainer',
    'VDNTrainer',
    'COMATrainer',
    'FACMACTrainer'
]

BASELINES = {
    'MAPPO': MAPPOTrainer,
    'IPPO': IPPOTrainer,
    'QMIX': QMIXTrainer,
    'VDN': VDNTrainer,
    'COMA': COMATrainer,
    'FACMAC': FACMACTrainer
}
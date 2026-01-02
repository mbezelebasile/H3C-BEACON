
try:
    from .H3CTrainer import *
except ImportError:
    print("Note: H3CTrainer.py not found in modules/. Please add your H3CTrainer.py file.")
from modules.AAH import AAH
from modules.PCC import PCC
from modules.CDEGA import CDEGA
from modules.AUTO_HP import AUTO_HP
from modules.H3CTrainer import H3CTrainer

__version__ = '3.1-stable'
__all__ = ['AAH', 'PCC', 'CDEGA', 'AUTO_HP', 'H3CTrainer']
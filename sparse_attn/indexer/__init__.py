"""Sparse indexer package.

Kept surface: TIA coarse/scoring primitives + NFA near/far pipeline.
"""
from .base import Indexer
from .tia_indexer import TIAIndexer
from .nfa_indexer import NFAIndexer

indexer_type_dict = {
    'tia': TIAIndexer,
    'nfa': NFAIndexer,
}

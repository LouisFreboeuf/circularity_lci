# src/circularity_lci/__init__.py

"""
src: A Python package for circularity and LCA analysis.
This package provides classes for analyzing burden-free activities,
duplicating a list of technosphere flows to biosphere flows, calculating circularity indicators,
to case studies and entire LCA database.
For long calculating process, it also contain a tracking progress in LCA projects.
"""

from .burden_free_analyzer import BurdenFreeAnalyzer
from .biosphere_flow_manager import BiosphereFlowManager
from .circularity_calculator import CircularityCalculator
from .multi_lca_calculator import MultiLCACalculator
from .circularity_database_analyzer import CircularityDatabaseAnalyzer
from .lcia import LCIAMethodBuilder, create_circularity_lcia_methods
from .progress_tracker import ProgressTracker
#  import_bafu_from_sacchi is a top-level function in functions_bafu_from_sacchi.py
from .functions_bafu_from_sacchi import import_bafu_from_sacchi

# Lowercase aliases for desired usage pattern
burdenfreeanalyzer = BurdenFreeAnalyzer
biosphereflowmanager = BiosphereFlowManager
circularitycalculator = CircularityCalculator
circularitydatabaseanalyzer = CircularityDatabaseAnalyzer
multilcacalculator = MultiLCACalculator
lciamethodbuilder = LCIAMethodBuilder
progressTracker = ProgressTracker

__all__ = [
    "BurdenFreeAnalyzer",
    "BiosphereFlowManager",
    "CircularityCalculator",
    "MultiLCACalculator",
    "CircularityDatabaseAnalyzer",
    "LCIAMethodBuilder",
    "create_circularity_lcia_methods",
    "ProgressTracker",
    "import_bafu_from_sacchi",
    "burdenfreeanalyzer",
    "biosphereflowmanager", 
    "circularitycalculator",
    "circularitydatabaseanalyzer",
    "multilcacalculator",
    "lciamethodbuilder",
    "progressTracker",
]

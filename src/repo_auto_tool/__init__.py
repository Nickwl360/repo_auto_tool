"""
repo-improver: Continuously improve a codebase toward a goal using Claude Code CLI.
"""

from .config import ImproverConfig
from .improver import RepoImprover
from .state import ImprovementState

__version__ = "0.1.0"
__all__ = ["RepoImprover", "ImproverConfig", "ImprovementState"]

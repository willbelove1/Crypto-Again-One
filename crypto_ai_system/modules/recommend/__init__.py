# This file makes Python treat the 'recommend' directory as a subpackage.

from .engine import RecommenderCore, parse_json_from_ai

__all__ = [
    "RecommenderCore",
    "parse_json_from_ai" # Exposing helper
]

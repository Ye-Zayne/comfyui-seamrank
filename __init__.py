"""ComfyUI SeamRank: model-independent continuation candidate comparison."""

from .nodes import SeamRankCandidates

NODE_CLASS_MAPPINGS = {"SeamRankCandidates": SeamRankCandidates}
NODE_DISPLAY_NAME_MAPPINGS = {"SeamRankCandidates": "SeamRank · Rank Continuations"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

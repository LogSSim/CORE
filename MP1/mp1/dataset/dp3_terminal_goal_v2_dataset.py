"""V2 dataset entry for DP3 terminal-goal training.

This is intentionally a separate Hydra target so the v2 method can evolve
without changing the original dp3_terminal_goal dataset entry.
"""

from mp1.dataset.dp3_terminal_goal_dataset import DP3TerminalGoalDataset


class DP3TerminalGoalV2Dataset(DP3TerminalGoalDataset):
    pass

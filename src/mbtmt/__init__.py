"""
Compatibility alias for a common typo.

The actual package name is `mtbmt` (Mentor-Trajectory Based Model Trainer).
If some environment/code imports `mbtmt`, we forward it to `mtbmt`.
"""

from mtbmt import *  # noqa: F401,F403


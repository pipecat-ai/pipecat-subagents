#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapters for bus message serialization.

Provides ready-made `TypeAdapter` implementations for common Pipecat types
(frames, aggregator messages, etc.) used in bus messages.
"""

from pipecat_subagents.bus.adapters.frame_adapter import FrameAdapter

__all__ = [
    "FrameAdapter",
]

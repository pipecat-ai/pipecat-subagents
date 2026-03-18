#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapters for bus message serialization.

Provides ready-made ``TypeAdapter`` implementations for common Pipecat types
(``LLMContext``, ``ToolsSchema``) used in bus messages.
"""

from pipecat_subagents.bus.adapters.base import TypeAdapter
from pipecat_subagents.bus.adapters.llm_context_adapter import LLMContextAdapter
from pipecat_subagents.bus.adapters.tools_schema_adapter import ToolsSchemaAdapter

__all__ = [
    "LLMContextAdapter",
    "ToolsSchemaAdapter",
    "TypeAdapter",
]

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent base classes for the multi-agent framework.

This package provides the core agent hierarchy:

- `BaseAgent`: Abstract base with bus integration and lifecycle management.
- `LLMAgent`: Agent with an LLM pipeline (BusInput -> LLM -> BusOutput).
- `LLMContextAgent`: LLM agent that owns its own context with system messages.
- `FlowsAgent`: Agent that uses Pipecat Flows for structured conversation.
"""

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.agents.flows_agent import FlowsAgent
from pipecat_agents.agents.llm_agent import LLMAgent
from pipecat_agents.agents.llm_context_agent import LLMContextAgent

__all__ = [
    "BaseAgent",
    "FlowsAgent",
    "LLMAgent",
    "LLMContextAgent",
]

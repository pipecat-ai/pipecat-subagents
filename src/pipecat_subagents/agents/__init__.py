#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent base classes for the multi-agent framework.

This package provides the core agent hierarchy:

- `BaseAgent`: Abstract base with bus integration and lifecycle management.
- `DetachedAgent`: Agent with a pipeline detached from transport, connected
  via bus frame routing and handoff semantics.
- `LLMAgent`: Agent with an LLM pipeline (BusInput -> LLM -> BusOutput).
- `FlowsAgent`: Agent that uses Pipecat Flows for structured conversation.
"""

from pipecat_subagents.agents.base_agent import ActivationArgs, BaseAgent
from pipecat_subagents.agents.detached_agent import DetachedAgent
from pipecat_subagents.agents.flows_agent import FlowsAgent
from pipecat_subagents.agents.llm_agent import LLMActivationArgs, LLMAgent
from pipecat_subagents.agents.tool import tool

__all__ = [
    "ActivationArgs",
    "BaseAgent",
    "DetachedAgent",
    "FlowsAgent",
    "LLMActivationArgs",
    "LLMAgent",
    "tool",
]

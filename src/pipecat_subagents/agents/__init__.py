#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent base classes for the multi-agent framework.

This package provides the core agent hierarchy:

- `BaseAgent`: Base with bus integration, lifecycle, and optional bridged mode.
- `LLMAgent`: Agent with an LLM pipeline and tool registration.
- `FlowsAgent`: Agent that uses Pipecat Flows for structured conversation.
"""

from pipecat_subagents.agents.base_agent import AgentActivationArgs, BaseAgent
from pipecat_subagents.agents.flows_agent import FlowsAgent
from pipecat_subagents.agents.llm_agent import LLMAgent, LLMAgentActivationArgs
from pipecat_subagents.agents.task_group import (
    TaskGroupContext,
    TaskGroupError,
    TaskGroupEvent,
    TaskGroupResponse,
    TaskStatus,
)
from pipecat_subagents.agents.tool import tool

__all__ = [
    "AgentActivationArgs",
    "BaseAgent",
    "FlowsAgent",
    "LLMAgentActivationArgs",
    "LLMAgent",
    "TaskGroupContext",
    "TaskGroupError",
    "TaskGroupEvent",
    "TaskGroupResponse",
    "TaskStatus",
    "tool",
]

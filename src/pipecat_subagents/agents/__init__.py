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
  Import from ``pipecat_subagents.agents.flows_agent`` (requires ``pipecat-ai-subagents[flows]``).
"""

from pipecat_subagents.agents.base_agent import AgentActivationArgs, BaseAgent
from pipecat_subagents.agents.llm_agent import LLMAgent, LLMAgentActivationArgs
from pipecat_subagents.agents.task_context import (
    TaskContext,
    TaskError,
    TaskGroupContext,
    TaskGroupError,
    TaskGroupEvent,
    TaskGroupResponse,
    TaskStatus,
)
from pipecat_subagents.agents.task_decorator import task
from pipecat_subagents.agents.tool_decorator import tool

__all__ = [
    "AgentActivationArgs",
    "BaseAgent",
    "LLMAgentActivationArgs",
    "LLMAgent",
    "TaskContext",
    "TaskError",
    "TaskGroupContext",
    "TaskGroupError",
    "TaskGroupEvent",
    "TaskGroupResponse",
    "TaskStatus",
    "task",
    "tool",
]

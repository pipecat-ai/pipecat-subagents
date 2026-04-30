#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent base classes for the multi-agent framework.

This package provides the core agent hierarchy:

- `BaseAgent`: Base with bus integration, lifecycle, and optional bridged mode.
- `LLMAgent`: Agent with an LLM pipeline and tool registration.

`FlowsAgent` is provided in the ``pipecat_subagents.agents.flows`` subpackage
and requires the ``pipecat-ai-subagents[flows]`` optional dependency.
"""

from pipecat_subagents.agents.base_agent import AgentActivationArgs, BaseAgent
from pipecat_subagents.agents.llm import LLMAgent, LLMAgentActivationArgs, LLMContextAgent, tool
from pipecat_subagents.agents.task_context import (
    TaskContext,
    TaskError,
    TaskEvent,
    TaskGroupContext,
    TaskGroupError,
    TaskGroupEvent,
    TaskGroupResponse,
    TaskStatus,
)
from pipecat_subagents.agents.task_decorator import task
from pipecat_subagents.agents.watch_decorator import agent_ready

__all__ = [
    "AgentActivationArgs",
    "BaseAgent",
    "LLMAgentActivationArgs",
    "LLMAgent",
    "TaskContext",
    "TaskError",
    "TaskEvent",
    "TaskGroupContext",
    "TaskGroupError",
    "TaskGroupEvent",
    "TaskGroupResponse",
    "TaskStatus",
    "agent_ready",
    "task",
    "tool",
]

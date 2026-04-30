#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent base classes for the multi-agent framework.

This package provides the core agent hierarchy:

- `BaseAgent`: Base with bus integration, lifecycle, and optional bridged mode.
- `LLMAgent`: Agent with an LLM pipeline and tool registration.
- `FlowsAgent` is provided in the ``pipecat_subagents.agents.flows`` subpackage
and requires the ``pipecat-ai-subagents[flows]`` optional dependency.
- `UIAgent`: ``LLMAgent`` that dispatches UI events from the client.
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
from pipecat_subagents.agents.ui import (
    UI_STATE_PROMPT_GUIDE,
    Click,
    Focus,
    Highlight,
    Navigate,
    ReplyToolMixin,
    ScrollTo,
    SelectText,
    SetInputValue,
    Toast,
    UIAgent,
    attach_ui_bridge,
    on_ui_event,
)
from pipecat_subagents.agents.watch_decorator import agent_ready

__all__ = [
    "AgentActivationArgs",
    "BaseAgent",
    "Click",
    "Focus",
    "Highlight",
    "LLMAgent",
    "LLMAgentActivationArgs",
    "Navigate",
    "ReplyToolMixin",
    "ScrollTo",
    "SelectText",
    "SetInputValue",
    "TaskContext",
    "TaskError",
    "TaskEvent",
    "TaskGroupContext",
    "TaskGroupError",
    "TaskGroupEvent",
    "TaskGroupResponse",
    "TaskStatus",
    "Toast",
    "UIAgent",
    "UI_STATE_PROMPT_GUIDE",
    "agent_ready",
    "attach_ui_bridge",
    "on_ui_event",
    "task",
    "tool",
]

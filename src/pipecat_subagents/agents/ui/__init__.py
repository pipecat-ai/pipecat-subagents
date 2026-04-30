#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UI agent: LLM agent that observes and drives a GUI app.

Composes a wire protocol for client UI events, accessibility-tree
snapshots, and server-emitted UI commands, plus an opt-in
``ReplyToolMixin`` exposing the canonical bundled reply tool
(``answer`` + optional ``scroll_to`` / ``highlight``).
"""

from pipecat_subagents.agents.ui.ui_agent import UIAgent
from pipecat_subagents.agents.ui.ui_bridge import attach_ui_bridge
from pipecat_subagents.agents.ui.ui_commands import (
    Click,
    Focus,
    Highlight,
    Navigate,
    ScrollTo,
    SelectText,
    SetInputValue,
    Toast,
)
from pipecat_subagents.agents.ui.ui_event_decorator import on_ui_event
from pipecat_subagents.agents.ui.ui_prompts import UI_STATE_PROMPT_GUIDE
from pipecat_subagents.agents.ui.ui_tools import ReplyToolMixin

__all__ = [
    "Click",
    "Focus",
    "Highlight",
    "Navigate",
    "ReplyToolMixin",
    "ScrollTo",
    "SelectText",
    "SetInputValue",
    "Toast",
    "UIAgent",
    "UI_STATE_PROMPT_GUIDE",
    "attach_ui_bridge",
    "on_ui_event",
]

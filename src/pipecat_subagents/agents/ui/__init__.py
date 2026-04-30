#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UI agent: LLM agent that observes and drives a GUI app.

Composes a wire protocol for client UI events, accessibility-tree
snapshots, and server-emitted UI commands, plus opt-in tool mixins
for the standard action verbs (scroll_to, highlight, select_text,
set_input_value, click) and an answer terminator.
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
from pipecat_subagents.agents.ui.ui_tools import (
    AnswerToolMixin,
    ClickToolMixin,
    HighlightToolMixin,
    ScrollToToolMixin,
    SelectTextToolMixin,
    SetInputValueToolMixin,
)

__all__ = [
    "AnswerToolMixin",
    "Click",
    "ClickToolMixin",
    "Focus",
    "Highlight",
    "HighlightToolMixin",
    "Navigate",
    "ScrollTo",
    "ScrollToToolMixin",
    "SelectText",
    "SelectTextToolMixin",
    "SetInputValue",
    "SetInputValueToolMixin",
    "Toast",
    "UIAgent",
    "UI_STATE_PROMPT_GUIDE",
    "attach_ui_bridge",
    "on_ui_event",
]

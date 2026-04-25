#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Canonical prompt fragments describing the UI agent wire format.

Apps concatenate these constants into their system prompt so the
LLM understands the ``<ui_state>`` and ``<ui_event>`` developer
messages the SDK injects on its behalf.

Example::

    system_prompt = f'''
    You are a voice-driven music player agent.
    ...app-specific tool and behavior instructions...

    {UI_STATE_PROMPT_GUIDE}
    '''

The SDK updates the guide alongside the wire format, so apps get
new tags and semantics automatically on the next release.
"""

UI_STATE_PROMPT_GUIDE: str = """\
## UI context

Your developer context includes two kinds of SDK-managed messages:

- ``<ui_event name="..." >payload</ui_event>``: an event the user just \
triggered on the client (click, tab switch, navigation, etc.). The \
payload is JSON for that event.
- ``<ui_state>...</ui_state>``: an accessibility snapshot of the \
current screen, injected at the start of every task request. \
Indented tree in Playwright-MCP style. Each line is \
``- role "name" [state] [ref=eN]`` with children nested one level \
deeper.

State tags include ``[focused]``, ``[selected]``, ``[disabled]``, and \
``[offscreen]``. A node tagged ``[offscreen]`` exists on the page \
but is not currently in the user's viewport; only visible \
(non-offscreen) nodes count for position-based references.

Grids carry a ``[cols=N]`` tag. Their cells are listed in reading \
order (left-to-right, top-to-bottom); with N columns, cell K sits \
at row ``ceil(K/N)``, column ``((K-1) mod N) + 1``. Example with \
``[cols=8]`` and 16 children: "top right" is cell 8, "bottom left" \
is cell 9.

Resolve position references ("top right", "the first one", "the \
third new release") against the most recent ``<ui_state>`` tree. \
Sibling order matches reading order on screen (top-to-bottom, \
left-to-right within each region).\
"""

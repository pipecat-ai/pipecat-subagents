#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Opt-in tool mixin for ``UIAgent``.

Ships ``ReplyToolMixin``: a single ``reply(answer, scroll_to,
highlight)`` LLM tool that bundles a required spoken answer with
optional visual actions. One tool call per turn; the model cannot
drop the terminator because ``answer`` is a required argument that
the API schema enforces.

Apps that need a different bundle of fields (e.g. ``click``,
``select_text``, app-specific actions) write their own ``@tool reply``
on their ``UIAgent`` subclass instead of composing this mixin. The
helper methods on ``UIAgent`` (``scroll_to``, ``highlight``) make a
custom ``reply`` body straightforward, and ``send_command`` plus the
standard payload dataclasses (``Click``, ``SelectText``,
``SetInputValue``, etc.) cover the rest.
"""

from __future__ import annotations

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from pipecat_subagents.agents.llm.tool_decorator import tool


class ReplyToolMixin:
    """Expose a ``reply(answer, scroll_to, highlight)`` tool to the LLM.

    The canonical reply shape: a required spoken ``answer`` plus
    optional snapshot refs to scroll into view and/or visually pulse
    in the same turn. One tool call per turn, no chaining; the
    required ``answer`` argument is enforced by the API schema so
    the model cannot omit the terminator.

    Compose alongside ``UIAgent``::

        class MyUIAgent(ReplyToolMixin, UIAgent):
            ...

    For a different bundle of fields (e.g. apps that need ``click``,
    ``select_text``, app-specific actions like ``play_song``), write
    your own ``@tool reply`` on the ``UIAgent`` subclass directly. The
    helper methods on ``UIAgent`` (``scroll_to``, ``highlight``) and
    ``send_command`` plus the standard payload dataclasses cover the
    common building blocks.

    The host class must provide ``scroll_to``, ``highlight``, and
    ``respond_to_task`` (``UIAgent`` does) and must be the target of
    ``@tool`` discovery on the LLM pipeline.
    """

    @tool
    async def reply(
        self,
        params: FunctionCallParams,
        answer: str,
        scroll_to: str | None = None,
        highlight: str | None = None,
    ):
        """Reply to the user. Optionally scroll to and/or highlight an item.

        Always called exactly once per turn. ``answer`` is required;
        the visual fields are optional.

        Args:
            params: Framework-provided tool invocation context.
            answer: The spoken reply in plain language. One short
                sentence. No markdown, no symbols.
            scroll_to: Optional snapshot ref like ``"e5"``. When set,
                scrolls that element into view before speaking.
            highlight: Optional snapshot ref like ``"e5"``. When set,
                visually pulses that element while speaking.
        """
        preview = (answer or "").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        logger.info(
            f"{self}: reply(answer={preview!r}, scroll_to={scroll_to!r}, highlight={highlight!r})"
        )
        if scroll_to:
            await self.scroll_to(scroll_to)  # type: ignore[attr-defined]
        if highlight:
            await self.highlight(highlight)  # type: ignore[attr-defined]
        await self.respond_to_task(speak=answer)  # type: ignore[attr-defined]
        await params.result_callback(None)

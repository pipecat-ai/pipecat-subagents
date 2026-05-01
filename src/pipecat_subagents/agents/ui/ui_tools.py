#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Opt-in tool mixin for ``UIAgent``.

Ships ``ReplyToolMixin``: a single ``reply(answer, scroll_to,
highlight, select_text)`` LLM tool that bundles a required spoken
answer with optional attention-pointing visual actions. One tool
call per turn; the model cannot drop the terminator because
``answer`` is a required argument that the API schema enforces.

The bundled mixin covers the canonical "reply with optional
pointers" pattern, which fits pointing-style apps (lists, grids,
phone catalog) AND reading-style apps (articles, documents) without
forcing apps to choose between two near-identical mixins.

Apps with state-changing actions (``click``, ``set_input_value``)
or app-specific commands (e.g. ``play_song``) write their own
``@tool reply`` on the ``UIAgent`` subclass directly. The helper
methods on ``UIAgent`` (``scroll_to``, ``highlight``,
``select_text``) plus ``send_command`` and the standard payload
dataclasses (``Click``, ``SetInputValue``, etc.) cover the
building blocks for custom replies.
"""

from __future__ import annotations

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from pipecat_subagents.agents.llm.tool_decorator import tool


class ReplyToolMixin:
    """Expose a ``reply(answer, scroll_to, highlight, select_text)`` tool.

    The canonical reply shape: a required spoken ``answer`` plus
    optional snapshot refs to scroll into view, visually pulse, and
    place a text selection in the same turn. One tool call per turn,
    no chaining; the required ``answer`` argument is enforced by the
    API schema so the model cannot omit the terminator.

    Compose alongside ``UIAgent``::

        class MyUIAgent(ReplyToolMixin, UIAgent):
            ...

    Covers both pointing-style apps (grid of items, phone catalog —
    use ``scroll_to`` + ``highlight``) and reading-style apps
    (articles, documents — use ``scroll_to`` + ``select_text``).
    Apps don't need to pick a mode up front; the LLM uses whichever
    fields make sense per turn.

    For state-changing actions (``click``, ``set_input_value``) or
    app-specific commands, write your own ``@tool reply`` on the
    ``UIAgent`` subclass directly using the helper methods on
    ``UIAgent`` plus ``send_command``.

    The host class must provide ``scroll_to``, ``highlight``,
    ``select_text``, and ``respond_to_task`` (``UIAgent`` does) and
    must be the target of ``@tool`` discovery on the LLM pipeline.
    """

    @tool
    async def reply(
        self,
        params: FunctionCallParams,
        answer: str,
        scroll_to: str | None = None,
        highlight: list[str] | None = None,
        select_text: str | None = None,
    ):
        """Reply to the user. Optionally point at content visually.

        Always called exactly once per turn. ``answer`` is required;
        the visual fields are optional and may be combined.

        Three pointing idioms with distinct semantics:

        - ``scroll_to`` brings an element into view (single ref —
          there's only one viewport position).
        - ``highlight`` flashes elements briefly (list of refs —
          multiple tiles can pulse together). Best for short
          emphasis like a button or a fact.
        - ``select_text`` puts the page's text selection on an
          element (single ref). Best for "this paragraph" / "the
          section about X" so the user sees exactly what was meant.
          Persists until the user clicks elsewhere.

        Args:
            params: Framework-provided tool invocation context.
            answer: The spoken reply in plain language. One short
                sentence. No markdown, no symbols.
            scroll_to: Optional snapshot ref like ``"e5"``. When set,
                scrolls that element into view before speaking.
            highlight: Optional list of snapshot refs like
                ``["e5", "e8", "e47"]``. When set, visually pulses
                each element while speaking.
            select_text: Optional snapshot ref like ``"e5"``. When
                set, places the page's text selection on that
                element. Useful for pointing at content the agent
                refers to in its answer.
        """
        preview = (answer or "").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        logger.info(
            f"{self}: reply(answer={preview!r}, scroll_to={scroll_to!r}, "
            f"highlight={highlight!r}, select_text={select_text!r})"
        )
        if scroll_to:
            await self.scroll_to(scroll_to)  # type: ignore[attr-defined]
        if highlight:
            for ref in highlight:
                await self.highlight(ref)  # type: ignore[attr-defined]
        if select_text:
            await self.select_text(select_text)  # type: ignore[attr-defined]
        await self.respond_to_task(speak=answer)  # type: ignore[attr-defined]
        await params.result_callback(None)

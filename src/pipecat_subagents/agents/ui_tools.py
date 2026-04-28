#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Opt-in tool mixins for ``UIAgent``.

Each mixin exposes one focused LLM tool. Apps compose them by
inheriting alongside their ``UIAgent`` subclass::

    class MyAgent(ScrollToToolMixin, UIAgent):
        ...

Keeping these as separate mixins (instead of methods on ``UIAgent``)
means apps opt in to what the LLM sees: a single-screen app with no
scrolling shouldn't have a ``scroll_to`` tool cluttering its tool list.

The shipped mixin tools are **silent fire-and-forget side effects**:
they dispatch a UI command via ``send_command``, complete the
in-flight task via ``respond_to_task()`` with no ``speak`` field,
and exit. The visual change on the client (the scroll, the highlight)
is the user-facing feedback; the voice agent stays quiet for that
turn. Apps that want spoken narration override the mixin tool and
pass a ``speak`` argument::

    class MyAgent(ScrollToToolMixin, UIAgent):
        @tool
        async def scroll_to(self, params, ref: str):
            await self.send_command("scroll_to", ScrollTo(ref=ref))
            await self.respond_to_task(speak="Scrolling.")
            await params.result_callback(None)
"""

from __future__ import annotations

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from pipecat_subagents.agents.tool_decorator import tool
from pipecat_subagents.agents.ui_commands import (
    Click,
    Highlight,
    ScrollTo,
    SelectText,
    SetInputValue,
)


class ScrollToToolMixin:
    """Expose a ``scroll_to(ref)`` tool to the LLM.

    Inherit alongside ``UIAgent``. The LLM sees a ``scroll_to`` tool
    whose docstring guides it to call it before acting on elements
    tagged ``[offscreen]`` in the current ``<ui_state>``. The tool
    issues a standard ``ScrollTo`` command via the UI command pipe;
    the client's ``useStandardScrollToHandler`` (or any custom
    handler) does the actual scrolling.

    The host class must provide ``send_command(name, payload)`` and
    ``respond_to_task(...)`` (``UIAgent`` does) and must be the
    target of ``@tool`` discovery on the LLM pipeline.
    """

    @tool
    async def scroll_to(self, params: FunctionCallParams, ref: str):
        """Scroll an element into view by its snapshot ref.

        Call this when the user wants to interact with an element
        whose state in ``<ui_state>`` includes ``[offscreen]``. The
        client scrolls the referenced element into view, after which
        a fresh snapshot arrives and the element is no longer
        offscreen. Typical pattern: issue ``scroll_to`` on this turn,
        then the follow-up action on the next turn once the user
        confirms or repeats.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``.
        """
        logger.info(f"{self}: scroll_to(ref={ref!r})")
        await self.send_command("scroll_to", ScrollTo(ref=ref))  # type: ignore[attr-defined]
        await self.respond_to_task()  # type: ignore[attr-defined]
        await params.result_callback(None)


class HighlightToolMixin:
    """Expose a ``highlight(ref)`` tool to the LLM.

    Inherit alongside ``UIAgent``. Gives the LLM a way to visually
    point at a specific element on screen, e.g. answering "which one
    is Radiohead?" by flashing the tile. The tool issues a standard
    ``Highlight`` command via the UI command pipe; the client's
    ``useStandardHighlightHandler`` (or a custom one) does the
    actual visual effect.

    The host class must provide ``send_command(name, payload)`` and
    ``respond_to_task(...)`` (``UIAgent`` does) and must be the
    target of ``@tool`` discovery on the LLM pipeline.
    """

    @tool
    async def highlight(self, params: FunctionCallParams, ref: str):
        """Briefly flash an element on screen by its snapshot ref.

        Use this when the user asks you to point at, identify, or
        call attention to a specific element they'd recognize
        visually. After the flash, the element returns to its
        normal appearance.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``.
        """
        logger.info(f"{self}: highlight(ref={ref!r})")
        await self.send_command("highlight", Highlight(ref=ref))  # type: ignore[attr-defined]
        await self.respond_to_task()  # type: ignore[attr-defined]
        await params.result_callback(None)


class SelectTextToolMixin:
    """Expose a ``select_text(ref)`` tool to the LLM.

    Inherit alongside ``UIAgent``. Lets the agent point the user's
    attention at a specific paragraph or input value by selecting it
    on the page (mirror of the read-side ``<selection>`` block in
    ``<ui_state>``). Useful when the agent has just answered a
    question about "this paragraph" or "what I selected" and wants
    to visually confirm which content it was referring to.

    The tool issues a standard ``SelectText`` command via the UI
    command pipe; the client's ``useStandardSelectTextHandler`` (or a
    custom one) does the actual selection. Whole-element select with
    no offsets is the common case; apps that want sub-range selection
    should override the tool and pass ``start_offset`` /
    ``end_offset`` themselves.

    The host class must provide ``send_command(name, payload)`` and
    ``respond_to_task(...)`` (``UIAgent`` does) and must be the
    target of ``@tool`` discovery on the LLM pipeline.
    """

    @tool
    async def select_text(self, params: FunctionCallParams, ref: str):
        """Select an element's text on the page by its snapshot ref.

        Use this to highlight the exact paragraph or input the user
        is asking about, so they can see what content you're
        referring to. The selection covers the entire element's
        text content.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``. Typically a paragraph or input.
        """
        logger.info(f"{self}: select_text(ref={ref!r})")
        await self.send_command("select_text", SelectText(ref=ref))  # type: ignore[attr-defined]
        await self.respond_to_task()  # type: ignore[attr-defined]
        await params.result_callback(None)


class SetInputValueToolMixin:
    """Expose a ``set_input_value(ref, value)`` tool to the LLM.

    Inherit alongside ``UIAgent``. Lets the agent fill in form fields
    on the user's behalf, e.g. populating a clarification answer or a
    structured form entry derived from the conversation. The tool
    issues a standard ``SetInputValue`` command via the UI command
    pipe; the client's ``useStandardSetInputValueHandler`` (or a
    custom one) writes the value and dispatches input/change events
    so React-controlled inputs update naturally.

    The default tool overwrites the field. Apps that need an "append"
    mode (e.g. continuing a long answer in a textarea) should
    override the tool and pass ``replace=False`` themselves. The
    client refuses to write into ``disabled`` / ``readonly`` /
    ``type="hidden"`` targets so the agent can't bypass UI
    affordances the user is meant to control.

    The host class must provide ``send_command(name, payload)`` and
    ``respond_to_task(...)`` (``UIAgent`` does) and must be the
    target of ``@tool`` discovery on the LLM pipeline.
    """

    @tool
    async def set_input_value(self, params: FunctionCallParams, ref: str, value: str):
        """Write ``value`` into the input or textarea identified by ``ref``.

        Overwrites whatever was in the field. Use this when the user
        has answered a question that should be reflected in a form,
        or when populating a field from data the agent just looked up.
        The client refuses on disabled / readonly / hidden inputs.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``. Should be an input or textarea.
            value: The text to write into the field.
        """
        logger.info(f"{self}: set_input_value(ref={ref!r}, len={len(value)})")
        await self.send_command(  # type: ignore[attr-defined]
            "set_input_value",
            SetInputValue(ref=ref, value=value),
        )
        await self.respond_to_task()  # type: ignore[attr-defined]
        await params.result_callback(None)


class ClickToolMixin:
    """Expose a ``click(ref)`` tool to the LLM.

    Inherit alongside ``UIAgent``. Closes the form-fill story:
    text inputs and textareas are covered by ``set_input_value``,
    but checkboxes, radios, and submit buttons need a click. Also
    serves as the general "act on this element" verb for links,
    app-specific clickable nodes, and anything else with a real
    ``click`` handler.

    The tool issues a standard ``Click`` command via the UI command
    pipe; the client's ``useStandardClickHandler`` (or a custom one)
    calls ``el.click()`` after refusing on ``disabled`` targets.

    For native ``<select>``, prefer ``set_input_value`` (clicking
    options doesn't reliably change the selection); for custom
    comboboxes, apps wire their own command matching the library's
    interaction model. Click is the verb for everything else.

    The host class must provide ``send_command(name, payload)`` and
    ``respond_to_task(...)`` (``UIAgent`` does) and must be the
    target of ``@tool`` discovery on the LLM pipeline.

    Trust note: a click can submit forms, navigate, or delete
    things. Apps that ship this mixin accept that the agent can act
    on any element with a ref. Wire confirmation patterns at the
    prompt level (have the agent describe what it's about to do
    before clicking) when the action is destructive.
    """

    @tool
    async def click(self, params: FunctionCallParams, ref: str):
        """Click an element on the page by its snapshot ref.

        Use this for checkboxes, radios, submit buttons, links, and
        any other clickable element. The client refuses on
        ``disabled`` targets.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``.
        """
        logger.info(f"{self}: click(ref={ref!r})")
        await self.send_command("click", Click(ref=ref))  # type: ignore[attr-defined]
        await self.respond_to_task()  # type: ignore[attr-defined]
        await params.result_callback(None)

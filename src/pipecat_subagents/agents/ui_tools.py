#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Opt-in tool mixins for ``UIAgent``.

Each mixin exposes one focused LLM tool. Apps compose them by
inheriting alongside their ``UIAgent`` subclass::

    class MyAgent(ScrollToToolMixin, HighlightToolMixin, AnswerToolMixin, UIAgent):
        ...

Keeping these as separate mixins (instead of methods on ``UIAgent``)
means apps opt in to what the LLM sees: a single-screen app with no
scrolling shouldn't have a ``scroll_to`` tool cluttering its tool list.

## Two kinds of tools: actions and terminators

The shipped mixins fall into two roles, distinguished by whether they
complete the in-flight task:

- **Action mixins** (``ScrollToToolMixin``, ``HighlightToolMixin``,
  ``SelectTextToolMixin``, ``SetInputValueToolMixin``,
  ``ClickToolMixin``) are pure chainable side effects. Each
  dispatches one UI command and returns. They do **not** complete the
  task, which means the LLM can call several in sequence in a single
  turn (e.g. ``scroll_to(ref)`` followed by ``highlight(ref)``)
  without prematurely unblocking the voice agent waiting on the task.

- **Terminator mixins** (``AnswerToolMixin``) close out the in-flight
  task. ``answer(text)`` calls ``respond_to_task(speak=text)``, which
  the canonical voice agent reads as "speak this verbatim" and which
  unblocks the ``task("ui", ...)`` context that triggered this
  inference.

Every UIAgent that processes voice tasks needs at least one
terminator tool wired up; otherwise the voice agent's task hangs
until it times out. Compose ``AnswerToolMixin`` for the standard
``answer(text)`` shape, or write your own terminator tool that calls
``self.respond_to_task(...)``.

## Patterns

Standard "point at it AND tell me what it is"::

    class MyAgent(ScrollToToolMixin, HighlightToolMixin, AnswerToolMixin, UIAgent):
        ...

The LLM chains: ``scroll_to(ref)`` (if offscreen) →
``highlight(ref)`` → ``answer("Here's the iPhone 17.")``. The first
two are pure side effects; the third terminates.

Fire-and-forget where the visual change *is* the response (no
spoken reply on this turn)::

    class MyAgent(ScrollToToolMixin, AnswerToolMixin, UIAgent):
        ...

The LLM calls ``scroll_to(ref)``, then ``answer("")``. The empty
string ends the turn silently. Or skip ``AnswerToolMixin`` entirely
and override the action mixin to also terminate::

    class MyAgent(UIAgent):
        @tool
        async def scroll_to(self, params, ref: str):
            await self.send_command("scroll_to", ScrollTo(ref=ref))
            await self.respond_to_task()  # silent terminator
            await params.result_callback(None)

## Extending with custom actions

Define your own action by following the same shape as the shipped
ones: declare a ``@tool``, dispatch a UI command via
``send_command``, return through ``params.result_callback(None)``.
Don't call ``respond_to_task`` in the action — leave that to the
terminator tool::

    @tool
    async def play_song(self, params, song_id: str):
        await self.send_command("play_song", {"song_id": song_id})
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

    This is a pure side effect: it dispatches the command and
    returns. It does **not** complete the in-flight task, so the LLM
    is free to chain it with ``highlight``, ``answer``, or any other
    tool in the same turn. Compose ``AnswerToolMixin`` (or write
    your own terminator) so the task actually closes.

    The host class must provide ``send_command(name, payload)``
    (``UIAgent`` does) and must be the target of ``@tool`` discovery
    on the LLM pipeline.
    """

    @tool
    async def scroll_to(self, params: FunctionCallParams, ref: str):
        """Scroll an element into view by its snapshot ref.

        Call this when the user wants to interact with an element
        whose state in ``<ui_state>`` includes ``[offscreen]``.
        Chain it with another tool in the same turn (typically
        ``highlight`` and/or ``answer``); on its own this scroll is a
        side effect and does not end the turn.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``.
        """
        logger.info(f"{self}: scroll_to(ref={ref!r})")
        await self.send_command("scroll_to", ScrollTo(ref=ref))  # type: ignore[attr-defined]
        await params.result_callback(None)


class HighlightToolMixin:
    """Expose a ``highlight(ref)`` tool to the LLM.

    Inherit alongside ``UIAgent``. Gives the LLM a way to visually
    point at a specific element on screen, e.g. answering "which one
    is Radiohead?" by flashing the tile. The tool issues a standard
    ``Highlight`` command via the UI command pipe; the client's
    ``useStandardHighlightHandler`` (or a custom one) does the
    actual visual effect.

    This is a pure side effect: it dispatches the command and
    returns. It does **not** complete the in-flight task, so the LLM
    is free to chain it with ``scroll_to``, ``answer``, or any other
    tool in the same turn. Compose ``AnswerToolMixin`` (or write
    your own terminator) so the task actually closes.

    The host class must provide ``send_command(name, payload)``
    (``UIAgent`` does) and must be the target of ``@tool`` discovery
    on the LLM pipeline.
    """

    @tool
    async def highlight(self, params: FunctionCallParams, ref: str):
        """Briefly flash an element on screen by its snapshot ref.

        Use this when the user asks you to point at, identify, or
        call attention to a specific element they'd recognize
        visually. Chain it with another tool in the same turn
        (typically ``answer``, optionally preceded by ``scroll_to``);
        on its own this flash is a side effect and does not end the
        turn.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``.
        """
        logger.info(f"{self}: highlight(ref={ref!r})")
        await self.send_command("highlight", Highlight(ref=ref))  # type: ignore[attr-defined]
        await params.result_callback(None)


class SelectTextToolMixin:
    """Expose a ``select_text(ref)`` tool to the LLM.

    Inherit alongside ``UIAgent``. Lets the agent point the user's
    attention at a specific paragraph or input value by selecting it
    on the page (mirror of the read-side ``<selection>`` block in
    ``<ui_state>``). Useful when the agent has just answered a
    question about "this paragraph" or "what I selected" and wants
    to visually confirm which content it was referring to.

    This is a pure side effect: it dispatches the command and
    returns without completing the in-flight task, so the LLM can
    chain it with ``answer`` (or other tools) in the same turn.
    Compose ``AnswerToolMixin`` (or write your own terminator) so
    the task actually closes.

    The tool issues a standard ``SelectText`` command via the UI
    command pipe; the client's ``useStandardSelectTextHandler`` (or a
    custom one) does the actual selection. Whole-element select with
    no offsets is the common case; apps that want sub-range selection
    should override the tool and pass ``start_offset`` /
    ``end_offset`` themselves.

    The host class must provide ``send_command(name, payload)``
    (``UIAgent`` does) and must be the target of ``@tool`` discovery
    on the LLM pipeline.
    """

    @tool
    async def select_text(self, params: FunctionCallParams, ref: str):
        """Select an element's text on the page by its snapshot ref.

        Use this to highlight the exact paragraph or input the user
        is asking about, so they can see what content you're
        referring to. The selection covers the entire element's
        text content. Chain with ``answer`` to finish the turn.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``. Typically a paragraph or input.
        """
        logger.info(f"{self}: select_text(ref={ref!r})")
        await self.send_command("select_text", SelectText(ref=ref))  # type: ignore[attr-defined]
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

    This is a pure side effect: it dispatches the command and
    returns without completing the in-flight task, so the LLM can
    chain several writes (e.g. filling multiple fields) and then
    speak via ``answer`` to finish the turn. Compose
    ``AnswerToolMixin`` (or write your own terminator) so the task
    actually closes.

    The default tool overwrites the field. Apps that need an "append"
    mode (e.g. continuing a long answer in a textarea) should
    override the tool and pass ``replace=False`` themselves. The
    client refuses to write into ``disabled`` / ``readonly`` /
    ``type="hidden"`` targets so the agent can't bypass UI
    affordances the user is meant to control.

    The host class must provide ``send_command(name, payload)``
    (``UIAgent`` does) and must be the target of ``@tool`` discovery
    on the LLM pipeline.
    """

    @tool
    async def set_input_value(self, params: FunctionCallParams, ref: str, value: str):
        """Write ``value`` into the input or textarea identified by ``ref``.

        Overwrites whatever was in the field. Use this when the user
        has answered a question that should be reflected in a form,
        or when populating a field from data the agent just looked
        up. The client refuses on disabled / readonly / hidden
        inputs. Chain with ``answer`` to finish the turn.

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
        await params.result_callback(None)


class ClickToolMixin:
    """Expose a ``click(ref)`` tool to the LLM.

    Inherit alongside ``UIAgent``. Closes the form-fill story:
    text inputs and textareas are covered by ``set_input_value``,
    but checkboxes, radios, and submit buttons need a click. Also
    serves as the general "act on this element" verb for links,
    app-specific clickable nodes, and anything else with a real
    ``click`` handler.

    This is a pure side effect: it dispatches the command and
    returns without completing the in-flight task, so the LLM can
    chain it with ``answer`` (or further actions) in the same turn.
    Compose ``AnswerToolMixin`` (or write your own terminator) so
    the task actually closes.

    The tool issues a standard ``Click`` command via the UI command
    pipe; the client's ``useStandardClickHandler`` (or a custom one)
    calls ``el.click()`` after refusing on ``disabled`` targets.

    For native ``<select>``, prefer ``set_input_value`` (clicking
    options doesn't reliably change the selection); for custom
    comboboxes, apps wire their own command matching the library's
    interaction model. Click is the verb for everything else.

    The host class must provide ``send_command(name, payload)``
    (``UIAgent`` does) and must be the target of ``@tool`` discovery
    on the LLM pipeline.

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
        ``disabled`` targets. Chain with ``answer`` to finish the
        turn.

        Args:
            params: Framework-provided tool invocation context.
            ref: Ref string from the most recent ``<ui_state>``,
                e.g. ``"e42"``.
        """
        logger.info(f"{self}: click(ref={ref!r})")
        await self.send_command("click", Click(ref=ref))  # type: ignore[attr-defined]
        await params.result_callback(None)


class AnswerToolMixin:
    """Expose an ``answer(text)`` tool that closes out the in-flight task.

    Inherit alongside ``UIAgent``. ``answer(text)`` is the canonical
    terminator for a UIAgent turn: it completes the in-flight task
    via ``respond_to_task(speak=text)``, which the canonical voice
    agent reads as "speak this verbatim" via TTS. The voice agent's
    ``task("ui", ...)`` context unblocks at this point and the user
    hears the reply.

    Compose this alongside one or more action mixins (or your own
    custom action tools) so every UIAgent turn has something to end
    it. Pure-action mixins (``ScrollToToolMixin`` et al.) do **not**
    complete the task on their own. Without an answer (or another
    terminator) the voice agent waits until its task timeout fires.

    Apps that want a different terminator shape (e.g. a "respond
    silently" tool, or a tool that returns structured data instead
    of a spoken reply) should write their own ``@tool`` instead of
    composing this mixin.

    The host class must provide ``respond_to_task(...)``
    (``UIAgent`` does) and must be the target of ``@tool`` discovery
    on the LLM pipeline.
    """

    @tool
    async def answer(self, params: FunctionCallParams, text: str):
        """Speak ``text`` back to the user and end the turn.

        Always call this exactly once per turn, last. After any
        visual actions (scroll, highlight, etc.) have been chained,
        ``answer`` provides the spoken reply and closes the turn.
        Pass an empty string to end the turn silently when the
        visual change *is* the user-facing feedback.

        Args:
            params: Framework-provided tool invocation context.
            text: The spoken reply in plain language. One short
                sentence. No markdown, no symbols. Empty string for
                a silent end-of-turn.
        """
        preview = (text or "").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        logger.info(f"{self}: answer({preview!r})")
        speak: str | None = text if text else None
        await self.respond_to_task(speak=speak)  # type: ignore[attr-defined]
        await params.result_callback(None)

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UIAgent: LLM agent that dispatches UI events from the client."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMMessagesUpdateFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.models import (
    Click,
    Highlight,
    ScrollTo,
    SelectText,
    SetInputValue,
)
from pydantic import BaseModel

from pipecat_subagents.agents.llm.llm_context_agent import LLMContextAgent
from pipecat_subagents.agents.task_context import TaskGroupError, TaskStatus
from pipecat_subagents.agents.ui.ui_event_decorator import _collect_ui_event_handlers
from pipecat_subagents.agents.ui.ui_messages import (
    _UI_CANCEL_TASK_BUS_EVENT_NAME,
    _UI_SNAPSHOT_BUS_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
    BusUITaskCompletedMessage,
    BusUITaskUpdateMessage,
)
from pipecat_subagents.agents.ui.ui_task_context import UserTaskGroupContext
from pipecat_subagents.bus import AgentBus
from pipecat_subagents.bus.messages import (
    BusMessage,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
    BusTaskResponseMessage,
    BusTaskUpdateMessage,
)


@dataclass
class _UserTaskGroupRegistration:
    """Metadata kept on a UIAgent for each in-flight user task group.

    The registry is the lookup the agent uses to decide which bus
    task messages to forward to the client and whether a client
    ``__cancel_task`` event should be honored.
    """

    agent_names: list[str]
    label: str | None
    cancellable: bool


class UIAgent(LLMContextAgent):
    """LLM agent that dispatches UI events from the client.

    Receives ``BusUIEventMessage`` (republished by ``attach_ui_bridge``)
    and dispatches each one to the matching ``@on_ui_event(name)``
    handler. By default every event is also appended to the LLM context
    as ``<ui_event name="...">payload</ui_event>`` so the agent can
    reason about what the user just did on the next inference.

    ## Canonical pattern

    A ``UIAgent`` is the delegate side of a voice ↔ UI split: a
    bridged ``LLMAgent`` (the voice layer) receives the user's
    transcript and delegates UI-relevant work to this agent via
    ``self.task("ui_agent_name", payload={"query": text})``. The UI
    agent's ``on_task_request`` fires, ``<ui_state>`` is auto-injected,
    the LLM picks a tool, and the task completes with a spoken reply
    the voice agent hands to TTS.

    A working subclass needs two things: an LLM and an
    ``on_task_request`` override that feeds the user's query into the
    LLM. The pipeline (LLM wrapped in an aggregator pair) comes from
    ``LLMContextAgent``; activation, snapshot injection, and tool
    registration are handled by the defaults.

    Example::

        class MyUIAgent(UIAgent):
            def build_llm(self) -> LLMService:
                return OpenAILLMService(api_key="...")

            async def on_task_request(self, message):
                # super() records the in-flight task and (if
                # auto_inject_ui_state is on) injects ``<ui_state>``.
                # Then feed the user's query into the LLM context with
                # ``run_llm=True`` so the LLM actually generates.
                await super().on_task_request(message)
                query = (message.payload or {}).get("query", "")
                await self.queue_frame(LLMMessagesAppendFrame(
                    messages=[{"role": "developer", "content": query}],
                    run_llm=True,
                ))

            @on_ui_event("nav_click")
            async def on_nav(self, message):
                view = message.payload.get("view")
                ...

            @tool
            async def answer(self, params, text: str):
                await self.respond_to_task(speak=text)
                await params.result_callback(None)

    ## Activation default

    Unlike ``LLMAgent`` and ``LLMContextAgent`` (which default to
    ``active=False`` because voice handoff patterns require
    parent-driven activation), ``UIAgent`` defaults to
    ``active=True``. UIAgents are typically always-on delegates with
    no equivalent of "the user is now talking to UIAgent" — the
    agent activates as soon as its pipeline starts, registers its
    tools, and stays online to receive tasks. Pass ``active=False``
    explicitly if you have a handoff use case.

    ## Single-flight task semantics

    A ``UIAgent`` owns a single LLM context, one running pipeline, and
    one notion of "the current screen the user is looking at."
    Concurrent task processing has no clean meaning under that model:
    two overlapping tasks would interleave their ``<ui_state>``
    injections, race on which task is "current" when a tool calls
    ``respond_to_task``, and corrupt the conversation history. To
    codify the single-flight invariant, ``on_task_request`` acquires an
    internal lock that is held for the lifetime of the in-flight task.
    Task requests that arrive while another is in flight queue and
    process in arrival order. Concurrent submission is safe; concurrent
    execution is what gets serialized.

    The lock is released along two paths: ``respond_to_task`` (the
    happy path, where a tool fires the response) and
    ``on_task_cancelled`` (the cancellation path, where ``BaseAgent``
    sends the CANCELLED response directly and bypasses
    ``respond_to_task``). A tool that forgets to call
    ``respond_to_task`` for a non-cancelled task will hold the lock
    indefinitely and subsequent task requests will block. That's
    intentional. The hang is a fast-surfacing signal that a tool is
    missing its terminator. Don't add a watchdog timeout: it would
    mask the bug instead of exposing it, and the requester's own
    task timeout already provides error-path liveness on the
    calling side.

    ## Visibility convention

    When the client captures snapshots with ``trackViewport`` enabled
    (the default), every node in the accessibility tree whose bounding
    rect sits fully outside the viewport carries ``"offscreen"`` in its
    ``state`` list. The rendered ``<ui_state>`` surfaces this as
    ``[offscreen]``. ``visible_nodes()`` returns the filtered subset
    the user is actually looking at right now. Agents should treat
    ``[offscreen]`` nodes as on-the-page-but-not-in-view, typically
    issuing a ``scroll_to`` command before acting on them.
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = True,
        bridged: tuple[str, ...] | None = None,
        defer_tool_frames: bool = True,
        context: LLMContext | None = None,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
        inject_events: bool = True,
        auto_inject_ui_state: bool = True,
        keep_history: bool = False,
    ):
        """Initialize the UIAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active. Defaults to
                ``True`` for ``UIAgent`` (vs. ``False`` on
                ``LLMAgent`` / ``LLMContextAgent``) because the
                canonical UIAgent role is an always-on delegate that
                should self-activate as soon as its pipeline starts.
                Pass ``active=False`` only if you have a handoff use
                case.
            bridged: Bridge configuration. See ``BaseAgent`` for details.
            defer_tool_frames: Forwarded to ``LLMContextAgent``. See
                ``LLMAgent`` for details.
            context: Optional pre-built ``LLMContext``. Forwarded to
                ``LLMContextAgent``. Note that any messages seeded
                here are part of the mutable task history and are
                cleared before each task when ``keep_history=False``
                (the default), since the reset replaces the entire
                message list. For durable UI / app instructions,
                pass them via the LLM service's ``system_instruction``
                instead, where they live outside the context message
                list and survive resets.
            user_params: Optional user-aggregator parameters.
                Forwarded to ``LLMContextAgent``.
            assistant_params: Optional assistant-aggregator parameters.
                Forwarded to ``LLMContextAgent``.
            inject_events: Whether to auto-append each UI event to the
                LLM context as a ``<ui_event>`` developer message.
                Defaults to True. Override ``render_ui_event`` to change
                the injected content, or set this to False to disable.
            auto_inject_ui_state: When True (the default), the latest
                ``<ui_state>`` snapshot is appended to the LLM context
                at the start of every task request, so the agent always
                reasons over the current screen. Set to False if you
                want to call ``inject_ui_state()`` yourself.
            keep_history: When False (the default), the LLM context is
                cleared at the start of every task. Each task starts
                from an empty messages list, the current ``<ui_state>``
                is injected, and the user's query follows. Best for
                the canonical stateless-delegate role where the voice
                agent owns dialog state and the UI agent's job is
                "given the current screen, do something." When True,
                the conversation history accumulates across tasks
                (queries, prior ``<ui_state>`` blocks, tool calls,
                responses) so the LLM can reason over multi-turn
                references like "show me the next one" or "tell me
                about the Pro version of that." History accumulation
                grows token usage and can confuse smaller models when
                multiple ``<ui_state>`` snapshots are present at once;
                opt in only when the dialog continuity is worth the
                cost. Apps in ``keep_history=True`` mode that
                occasionally want to clear can call
                ``await self.reset_context()``.

                Note: in ``keep_history=False`` mode any messages
                pre-seeded via ``context=`` are also cleared on the
                first task, since the reset replaces the entire
                message list. Durable UI / app instructions belong
                in the LLM service's ``system_instruction`` setting
                (which is sent at API call time and is not touched
                by the context reset). For example, include the
                ``UI_STATE_PROMPT_GUIDE`` constant there alongside
                your app's prompt so the wire-format guide is always
                available to the LLM. Use ``keep_history=True`` if
                the seeded messages genuinely need to live in the
                conversation history.

        Raises:
            ValueError: If ``bridged`` is set together with the default
                ``auto_inject_ui_state=True``. The two are
                incompatible: auto-injection fires on
                ``on_task_request``, but a bridged ``UIAgent`` receives
                user voice frames through the bridge instead of task
                messages, so the snapshot would never reach the LLM
                context. The canonical pattern is a non-bridged
                ``UIAgent`` that receives delegated tasks from a
                separate voice ``LLMAgent``. If you really want a
                bridged ``UIAgent`` (advanced cases), pass
                ``auto_inject_ui_state=False`` explicitly and call
                ``inject_ui_state()`` yourself.
        """
        if bridged is not None and auto_inject_ui_state:
            raise ValueError(
                f"UIAgent '{name}': bridged + auto_inject_ui_state=True is "
                "incompatible. Auto-injection fires on on_task_request, but a "
                "bridged UIAgent receives frames through the bridge — the "
                "snapshot would never land in the LLM context and the agent "
                "would silently hallucinate. Use the canonical pattern "
                "(non-bridged UIAgent receiving tasks from a separate "
                "LLMAgent) or pass auto_inject_ui_state=False if you really "
                "want a bridged UIAgent and will manage injection manually."
            )
        super().__init__(
            name,
            bus=bus,
            active=active,
            bridged=bridged,
            defer_tool_frames=defer_tool_frames,
            context=context,
            user_params=user_params,
            assistant_params=assistant_params,
        )
        self._inject_events = inject_events
        self._auto_inject_ui_state = auto_inject_ui_state
        self._keep_history = keep_history
        self._ui_event_handlers = _collect_ui_event_handlers(self)
        # Latest accessibility snapshot received from the client. Updated
        # in ``on_bus_message`` when a ``__ui_snapshot`` event arrives.
        # Rendered into LLM context via ``inject_ui_state``.
        self._latest_snapshot: dict[str, Any] | None = None
        # Task currently being processed by this agent. Set in
        # ``on_task_request``, cleared by ``respond_to_task``. Lets
        # ``@tool`` methods (and the mixin tools) close out the task
        # without having to thread the task id through every call.
        self._current_task: BusTaskRequestMessage | None = None
        # Single-flight serialization. ``on_task_request`` acquires
        # this lock and holds it until ``respond_to_task`` fires.
        # See the "Single-flight task semantics" section in the
        # class docstring for why.
        self._task_lock = asyncio.Lock()
        # Registry of in-flight user task groups dispatched by this
        # agent (see ``user_task_group``). Keyed by ``task_id``.
        # ``on_bus_message`` consults this to decide which task
        # update / response messages should be forwarded to the
        # client as ``ui.task`` envelopes.
        self._user_task_groups: dict[str, _UserTaskGroupRegistration] = {}

    async def send_command(self, name: str, payload: Any = None) -> None:
        """Send a named UI command to the client.

        Publishes a ``BusUICommandMessage`` which the bridge installed
        by ``attach_ui_bridge`` translates into an
        ``RTVIServerMessageFrame`` on the root agent's pipeline.
        Client-side handlers registered via ``registerCommandHandler``
        dispatch on the command name.

        Args:
            name: App-defined command name (e.g. ``"toast"``,
                ``"navigate"``, or any app-specific name).
            payload: One of:

                - A pydantic ``BaseModel`` instance (including the
                  built-in command models in
                  ``pipecat.processors.frameworks.rtvi.models``).
                  Converted to a plain dict with ``model_dump()``.
                - A dataclass instance. Converted to a plain dict with
                  ``dataclasses.asdict``.
                - A ``dict`` forwarded as-is.
                - ``None``, forwarded as an empty dict.
        """
        if payload is None:
            serialized: Any = {}
        elif isinstance(payload, BaseModel):
            serialized = payload.model_dump()
        elif is_dataclass(payload) and not isinstance(payload, type):
            serialized = asdict(payload)
        else:
            serialized = payload

        await self.bus.send(
            BusUICommandMessage(
                source=self.name,
                target=None,
                command_name=name,
                payload=serialized,
            )
        )

    async def scroll_to(self, ref: str) -> None:
        """Dispatch a standard ``scroll_to`` UI command for the given ref.

        Convenience wrapper around ``send_command("scroll_to",
        ScrollTo(ref=ref))``. Useful when composing custom ``@tool``
        methods that bundle visual actions with a spoken reply (see
        ``ReplyToolMixin`` for the canonical shape).

        Note this is a regular instance method, not an LLM tool.
        Subclasses that want to expose it to the LLM should write
        their own ``@tool`` method that calls
        ``await super().scroll_to(ref)``.

        Args:
            ref: Snapshot ref like ``"e42"`` from the most recent
                ``<ui_state>``.
        """
        await self.send_command("scroll_to", ScrollTo(ref=ref))

    async def highlight(self, ref: str) -> None:
        """Dispatch a standard ``highlight`` UI command for the given ref.

        Convenience wrapper around ``send_command("highlight",
        Highlight(ref=ref))``. Same pattern as ``scroll_to``: a plain
        instance method, not an LLM tool. Compose into a custom
        ``@tool`` body, or use ``ReplyToolMixin`` for the standard
        shape.

        Args:
            ref: Snapshot ref like ``"e42"`` from the most recent
                ``<ui_state>``.
        """
        await self.send_command("highlight", Highlight(ref=ref))

    async def select_text(
        self,
        ref: str,
        *,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ) -> None:
        """Dispatch a standard ``select_text`` UI command for the given ref.

        Convenience wrapper around ``send_command("select_text",
        SelectText(...))``. Same pattern as ``scroll_to`` and
        ``highlight``: a plain instance method, not an LLM tool.
        Compose into a custom ``@tool`` body for apps that want to
        point at content via the page's text selection (deixis).

        With ``start_offset`` / ``end_offset`` set, the client's
        standard handler selects a sub-range of the element's text.
        With both ``None`` (the default), the entire element's
        contents are selected. The offsets are character offsets over
        the element's concatenated ``textContent``.

        Args:
            ref: Snapshot ref like ``"e42"`` from the most recent
                ``<ui_state>``.
            start_offset: Optional character offset where the
                selection should start. Pass with ``end_offset`` to
                select a sub-range.
            end_offset: Optional character offset where the selection
                should end (exclusive).
        """
        await self.send_command(
            "select_text",
            SelectText(ref=ref, start_offset=start_offset, end_offset=end_offset),
        )

    async def click(self, ref: str) -> None:
        """Dispatch a standard ``click`` UI command for the given ref.

        Convenience wrapper around ``send_command("click",
        Click(ref=ref))``. Same pattern as the other helpers: a plain
        instance method, not an LLM tool. Use inside custom ``@tool``
        bodies for state-changing form actions like checkboxes,
        radio buttons, and submit buttons.

        The standard client handler silently no-ops on ``disabled``
        targets so the agent can't bypass UI affordances meant to be
        user-controlled.

        Args:
            ref: Snapshot ref like ``"e42"`` from the most recent
                ``<ui_state>``.
        """
        await self.send_command("click", Click(ref=ref))

    async def set_input_value(
        self,
        ref: str,
        value: str,
        *,
        replace: bool = True,
    ) -> None:
        """Dispatch a standard ``set_input_value`` UI command.

        Convenience wrapper around ``send_command("set_input_value",
        SetInputValue(...))``. Same pattern as the other helpers: a
        plain instance method, not an LLM tool. Use inside custom
        ``@tool`` bodies for filling text inputs and textareas.

        With ``replace=True`` (the default) the field's existing
        value is overwritten. With ``replace=False`` the value is
        appended (useful for continuing a long answer in a textarea).

        Args:
            ref: Snapshot ref like ``"e42"`` of the input/textarea.
            value: Text to write into the field.
            replace: When True, overwrite the existing value. When
                False, append to it. Defaults to True.
        """
        await self.send_command(
            "set_input_value",
            SetInputValue(ref=ref, value=value, replace=replace),
        )

    async def on_bus_message(self, message: BusMessage) -> None:
        """Dispatch UI events alongside base lifecycle handling."""
        await super().on_bus_message(message)

        # Forward task lifecycle for user-facing task groups before
        # touching anything else. This is independent of UI event
        # handling and may fire on messages targeted at this agent
        # (the requester) for groups it dispatched.
        if isinstance(message, BusTaskUpdateMessage):
            await self._maybe_forward_task_update(message)
            return
        if isinstance(message, BusTaskResponseMessage):
            await self._maybe_forward_task_completed(message)
            return

        if not isinstance(message, BusUIEventMessage):
            return
        if message.target and message.target != self.name:
            return

        # Reserved snapshot event: store and return without dispatch or
        # ``<ui_event>`` injection. Apps render via ``inject_ui_state``.
        if message.event_name == _UI_SNAPSHOT_BUS_EVENT_NAME:
            if isinstance(message.payload, dict):
                self._latest_snapshot = message.payload
            return

        # Reserved cancel event: route to ``cancel_task`` for the
        # registered user task group. Honored only when the group was
        # registered with ``cancellable=True``.
        if message.event_name == _UI_CANCEL_TASK_BUS_EVENT_NAME:
            await self._handle_cancel_task_event(message)
            return

        await self._handle_ui_event(message)

    @property
    def current_task(self) -> BusTaskRequestMessage | None:
        """The task this agent is currently processing, or ``None`` when idle.

        Set when ``on_task_request`` runs and cleared by
        ``respond_to_task``. Lets ``@tool`` methods inspect the
        in-flight task without threading the message through every
        call.
        """
        return self._current_task

    async def on_task_request(self, message: BusTaskRequestMessage) -> None:
        """Reset (when configured) and inject ``<ui_state>`` before dispatch.

        Acquires the single-flight lock, records the in-flight task
        for ``respond_to_task`` to close out, clears the LLM context
        when ``keep_history=False``, and injects the current snapshot
        when ``auto_inject_ui_state=True``. Returns to the subclass,
        which queues the LLM frame; the lock stays held until
        ``respond_to_task`` fires from inside a tool. Concurrent task
        requests block on the lock and process in arrival order.

        The pipeline processes the queued frames in order, so by the
        time the subclass appends the user's query with
        ``run_llm=True`` the context contains exactly: current
        ``<ui_state>`` + query.
        """
        await self._task_lock.acquire()
        try:
            await super().on_task_request(message)
            self._current_task = message
            if not self._keep_history:
                await self.reset_context()
            if self._auto_inject_ui_state:
                await self.inject_ui_state()
        except BaseException:
            # Setup failed before the LLM ever ran. Release so the
            # next task can proceed; respond_to_task won't fire on
            # this one.
            self._current_task = None
            if self._task_lock.locked():
                self._task_lock.release()
            raise

    async def reset_context(self) -> None:
        """Clear the LLM conversation history.

        Replaces all messages in the running context with an empty
        list. The system prompt (set on the LLM service via
        ``system_instruction``) is unaffected. Messages pre-seeded
        via ``context=`` on construction live in the same mutable
        message list and ARE cleared, so anything that needs to
        survive resets (durable instructions, prompt guides) belongs
        in ``system_instruction`` rather than in seed messages.

        Implemented via ``LLMMessagesUpdateFrame``, which the context
        aggregator processes in pipeline order alongside other
        context-modifying frames. Apps in ``keep_history=True`` mode
        call this when they want to deliberately start over (e.g.,
        the user said "start fresh"). Apps in the default
        ``keep_history=False`` mode don't need to call this; the
        per-task reset runs automatically.
        """
        await self.queue_frame(LLMMessagesUpdateFrame(messages=[], run_llm=False))

    async def respond_to_task(
        self,
        response: dict | None = None,
        *,
        speak: str | None = None,
        status: TaskStatus = TaskStatus.COMPLETED,
    ) -> None:
        """Complete the in-flight task this agent is processing.

        Convenience wrapper around ``send_task_response`` that looks up
        the current task from ``current_task``. Clears
        ``current_task`` after the response is sent so a second call
        is a no-op.

        ``speak`` is the convention the SDK demos use for "text the
        voice agent should hand verbatim to TTS". When provided it's
        merged into the response dict as ``{"speak": speak}``.
        Apps that don't follow the convention can pass a fully formed
        ``response`` dict instead and leave ``speak`` unset.

        No-op when there is no task in flight (e.g. the tool was
        invoked outside a task dispatch).

        Args:
            response: Result data dict. Merged with the ``speak`` key
                when ``speak`` is provided.
            speak: Optional short text for verbatim TTS. Omit (or pass
                ``None``) to leave the response without a ``speak``
                key, signaling to the voice agent that this turn
                completes silently.
            status: Completion status. Defaults to
                ``TaskStatus.COMPLETED``.
        """
        message = self._current_task
        if message is None:
            return
        self._current_task = None
        payload: dict = dict(response) if response else {}
        if speak is not None:
            payload["speak"] = speak
        try:
            await self.send_task_response(message.task_id, response=payload, status=status)
        finally:
            # Release the single-flight lock so the next queued task
            # can proceed. ``locked()`` guard handles the unusual
            # case where a tool calls ``respond_to_task`` outside the
            # ``on_task_request`` flow (no lock to release).
            if self._task_lock.locked():
                self._task_lock.release()

    async def on_task_cancelled(self, message: BusTaskCancelMessage) -> None:
        """Release the single-flight lock when the in-flight task is cancelled.

        ``BaseAgent._handle_task_cancel`` sends a ``CANCELLED``
        response directly via ``send_task_response``, bypassing
        ``respond_to_task`` and its lock-release. Without this hook
        the lock would stay held after a cancellation and every
        subsequent task request would block at ``on_task_request``'s
        ``_task_lock.acquire()`` forever.

        Idempotent and race-safe: if a tool happens to fire
        ``respond_to_task`` concurrently and clears the slot first,
        the ``current_task`` check below short-circuits, and the
        ``locked()`` guard makes the release a no-op when nothing
        is held.
        """
        await super().on_task_cancelled(message)
        current = self._current_task
        if current is not None and current.task_id == message.task_id:
            self._current_task = None
            if self._task_lock.locked():
                self._task_lock.release()

    def user_task_group(
        self,
        *agent_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ) -> UserTaskGroupContext:
        """Dispatch a task group whose lifecycle is forwarded to the client.

        Behaves exactly like ``task_group(...)`` but additionally emits
        ``ui.task`` envelopes that the client's task reducer consumes.
        Use this for any user-initiated work that fans out to workers
        and that the user should be able to see (and optionally cancel)
        on screen.

        Workers don't need to change. Any ``send_task_update`` they
        emit against the group's ``task_id`` is forwarded automatically
        as a ``task_update`` envelope, and their final response is
        forwarded as ``task_completed``.

        Args:
            *agent_names: Names of the agents to send the task to.
            name: Optional task name for routing to named ``@task``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and task execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            label: Optional human-readable label surfaced to the
                client. The client UI uses it to title the in-flight
                task card.
            cancellable: Whether the client may request cancellation
                of this group via the reserved ``__cancel_task``
                event. Defaults to True.

        Returns:
            A ``UserTaskGroupContext`` to use with ``async with``.

        Example::

            async with self.user_task_group(
                "researcher_a", "researcher_b",
                payload={"query": query},
                label=f"Research: {query}",
            ) as tg:
                async for event in tg:
                    ...
        """
        for agent_name in agent_names:
            if not isinstance(agent_name, str):
                raise TypeError(
                    f"{self} Expected agent name as str, got {type(agent_name).__name__}"
                )
        return UserTaskGroupContext(
            self,
            agent_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
            label=label,
            cancellable=cancellable,
        )

    async def start_user_task_group(
        self,
        *agent_names: str,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ) -> str:
        """Fire-and-forget version of ``user_task_group``.

        Dispatches a user task group in the background and returns
        immediately. The SDK manages the asyncio task that holds the
        ``user_task_group`` context open while workers run, so callers
        don't need to spawn one themselves. All lifecycle events
        (``group_started``, ``task_update``, ``task_completed``,
        ``group_completed``) are forwarded to the client as usual.

        Use this when an LLM ``@tool`` body wants to kick off a task
        group and return immediately so the voice agent unblocks.
        Use ``user_task_group`` (the context manager) when the caller
        wants to consume worker events inline (``async for event in
        tg``) or react to results before returning.

        Worker exceptions inside the dispatched context are logged
        but do not propagate; cancellation works the same way it
        does for the context-manager form (the user clicks Cancel,
        the SDK turns it into ``cancel_task``).

        Args:
            *agent_names: Names of the agents to send the task to.
            name: Optional task name for routing to named ``@task``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and task execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            label: Optional human-readable label surfaced to the
                client. The client UI uses it to title the in-flight
                task card.
            cancellable: Whether the client may request cancellation
                of this group via the reserved ``__cancel_task``
                event. Defaults to True.

        Returns:
            The ``task_id`` of the dispatched group. Useful if the
            caller wants to track it (e.g. to cancel programmatically
            via ``cancel_task(task_id)``).

        Example::

            @tool
            async def reply(self, params, answer, research_query=None):
                if research_query:
                    await self.start_user_task_group(
                        "wikipedia", "news", "scholar",
                        payload={"query": research_query},
                        label=f"Research: {research_query}",
                    )
                await self.respond_to_task(speak=answer)
                await params.result_callback(None)
        """
        ctx = self.user_task_group(
            *agent_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
            label=label,
            cancellable=cancellable,
        )
        # Enter the context now so the caller has a valid task_id and
        # the ``group_started`` envelope has fired before we return.
        # The body and exit run in a background asyncio task so the
        # caller doesn't await worker completion.
        await ctx.__aenter__()
        task_id = ctx.task_id

        async def _run_to_completion() -> None:
            # Drain the event stream so __aexit__ sees a fully
            # consumed group, matching what ``async with ... : pass``
            # does. Cancellation and worker errors are expected exits
            # for fire-and-forget groups: the client already learned
            # via the group_completed envelope, so we log at debug.
            iteration_exc: BaseException | None = None
            try:
                async for _ in ctx:
                    pass
            except Exception as e:
                iteration_exc = e

            try:
                if iteration_exc is None:
                    await ctx.__aexit__(None, None, None)
                else:
                    await ctx.__aexit__(
                        type(iteration_exc),
                        iteration_exc,
                        iteration_exc.__traceback__,
                    )
            except TaskGroupError as e:
                logger.debug(
                    f"UIAgent '{self.name}': background user task group {task_id} ended: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"UIAgent '{self.name}': background user task group {task_id} failed: {e}"
                )

        self.create_asyncio_task(
            _run_to_completion(),
            f"{self.name}::user_task_group::{task_id}",
        )
        return task_id

    def _register_user_task_group(
        self,
        *,
        task_id: str,
        agent_names: list[str],
        label: str | None,
        cancellable: bool,
    ) -> None:
        """Register an in-flight user task group for lifecycle forwarding.

        Called from ``UserTaskGroupContext.__aenter__``. Subsequent
        ``BusTaskUpdateMessage`` / ``BusTaskResponseMessage`` whose
        ``task_id`` matches this entry will be forwarded to the
        client.
        """
        if task_id in self._user_task_groups:
            logger.warning(
                f"UIAgent '{self.name}': user task group {task_id} already registered; overwriting"
            )
        self._user_task_groups[task_id] = _UserTaskGroupRegistration(
            agent_names=list(agent_names),
            label=label,
            cancellable=cancellable,
        )

    def _unregister_user_task_group(self, task_id: str) -> None:
        """Remove a user task group from the forwarding registry.

        Called from ``UserTaskGroupContext.__aexit__``. After this,
        late-arriving updates or responses for the group are not
        forwarded.
        """
        self._user_task_groups.pop(task_id, None)

    async def _maybe_forward_task_update(self, message: BusTaskUpdateMessage) -> None:
        """Forward a worker update for a registered user task group.

        No-op if the message's ``task_id`` is not registered.
        """
        if message.task_id not in self._user_task_groups:
            return
        await self.bus.send(
            BusUITaskUpdateMessage(
                source=self.name,
                target=None,
                task_id=message.task_id,
                agent_name=message.source,
                data=message.update,
                at=int(time.time() * 1000),
            )
        )

    async def _maybe_forward_task_completed(self, message: BusTaskResponseMessage) -> None:
        """Forward a worker response for a registered user task group.

        No-op if the message's ``task_id`` is not registered.
        """
        if message.task_id not in self._user_task_groups:
            return
        await self.bus.send(
            BusUITaskCompletedMessage(
                source=self.name,
                target=None,
                task_id=message.task_id,
                agent_name=message.source,
                status=str(message.status),
                response=message.response,
                at=int(time.time() * 1000),
            )
        )

    async def _handle_cancel_task_event(self, message: BusUIEventMessage) -> None:
        """Translate a client ``__cancel_task`` event into ``cancel_task``.

        Looks up the registered group and calls
        ``cancel_task(task_id, reason)``. Ignores the request silently
        if the group is unknown or was registered with
        ``cancellable=False``.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            logger.warning(
                f"UIAgent '{self.name}': received {_UI_CANCEL_TASK_BUS_EVENT_NAME} "
                "with no task_id; ignoring"
            )
            return
        registration = self._user_task_groups.get(task_id)
        if registration is None:
            logger.debug(
                f"UIAgent '{self.name}': {_UI_CANCEL_TASK_BUS_EVENT_NAME} for "
                f"unknown task_id {task_id}; ignoring"
            )
            return
        if not registration.cancellable:
            logger.debug(
                f"UIAgent '{self.name}': {_UI_CANCEL_TASK_BUS_EVENT_NAME} for "
                f"non-cancellable group {task_id}; ignoring"
            )
            return
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = None
        await self.cancel_task(task_id, reason=reason or "cancelled by user")

    def render_ui_state(self) -> str:
        """Render the latest accessibility snapshot as a ``<ui_state>`` block.

        Produces Playwright-MCP-style indented text with stable element
        refs. Apps inject the output via ``inject_ui_state()`` when they
        want the LLM to see what's on screen.

        When the snapshot carries a current text selection, a nested
        ``<selection ref="...">...</selection>`` block is appended
        inside ``<ui_state>`` so the LLM can resolve deictic references
        ("this paragraph", "what I selected") against on-page content.

        Override to customize the rendered form. Returns an empty
        string if no snapshot has been received yet.
        """
        if not self._latest_snapshot:
            return ""
        root = self._latest_snapshot.get("root")
        if not isinstance(root, dict):
            return ""
        lines = ["<ui_state>"]
        _render_node(root, depth=0, lines=lines)
        selection = self._latest_snapshot.get("selection")
        if isinstance(selection, dict):
            _render_selection(selection, lines)
        lines.append("</ui_state>")
        return "\n".join(lines)

    def visible_nodes(self) -> list[dict[str, Any]]:
        """Return the snapshot nodes the user is currently looking at.

        Walks the latest snapshot and returns a flat list of every
        node whose ``state`` does not contain ``"offscreen"``. Useful
        for apps that want to reason about visible interactables
        without scanning the rendered ``<ui_state>`` text.

        Ordering follows a depth-first traversal of the snapshot,
        matching the order in ``<ui_state>``. Returns an empty list
        when no snapshot has been received yet.
        """
        if not self._latest_snapshot:
            return []
        root = self._latest_snapshot.get("root")
        if not isinstance(root, dict):
            return []
        out: list[dict[str, Any]] = []
        _collect_visible(root, out)
        return out

    async def inject_ui_state(self) -> None:
        """Append the latest ``<ui_state>`` block to the LLM context.

        No-op when no snapshot has been received. Frame has
        ``run_llm=False`` — the snapshot is context, not a user turn.
        """
        content = self.render_ui_state()
        if not content:
            return
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": content}],
                run_llm=False,
            )
        )

    def render_ui_event(self, message: BusUIEventMessage) -> str:
        """Render a UI event as a string for LLM context injection.

        Override to customize the injected content. The default wraps
        the event in a single ``<ui_event>`` XML tag with a ``name``
        attribute and a JSON-encoded payload as inner text.

        Args:
            message: The UI event to render.

        Returns:
            A string to append to the LLM context as a developer message.
        """
        payload_repr = json.dumps(message.payload, default=str)
        return f'<ui_event name="{message.event_name}">{payload_repr}</ui_event>'

    async def _handle_ui_event(self, message: BusUIEventMessage) -> None:
        """Inject the event into LLM context, then dispatch to the handler.

        Injection runs synchronously first so the ``<ui_event>``
        developer message lands in the context before any side effects
        the handler triggers. The matching ``@on_ui_event`` handler
        then runs in its own asyncio task so the bus dispatcher isn't
        held open while the handler awaits downstream work (task
        requests, network calls). Events with no registered handler
        are a no-op after injection.
        """
        if self._inject_events:
            content = self.render_ui_event(message)
            if content:
                await self.queue_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "developer", "content": content}],
                        run_llm=False,
                    )
                )

        handler = self._ui_event_handlers.get(message.event_name)
        if handler is None:
            return

        # Handlers run in their own asyncio task so the bus dispatcher
        # is never held open while a handler awaits downstream work
        # (task requests, network calls, etc.). Same pattern as ``@task``.
        self.create_asyncio_task(
            handler(message),
            f"{self.name}::ui_event_{message.event_name}",
        )


def _collect_visible(node: dict[str, Any], out: list[dict[str, Any]]) -> None:
    """Depth-first collect nodes whose state does not include offscreen."""
    state = node.get("state")
    is_offscreen = isinstance(state, list) and "offscreen" in state
    if not is_offscreen:
        out.append(node)
    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                _collect_visible(child, out)


def _render_node(node: dict[str, Any], *, depth: int, lines: list[str]) -> None:
    """Render one A11yNode dict as Playwright-MCP-style indented text.

    Format per node::

        - role "name" [level=N] [cols=N] [rows=N] [state1] [state2] [ref=eN]:

    Trailing ``:`` when the node has children. ``name``, ``level``,
    grid dims, and state tags are emitted only when present on the
    node.
    """
    role = node.get("role", "generic")
    name = node.get("name")
    value = node.get("value")
    state = node.get("state") or []
    level = node.get("level")
    colcount = node.get("colcount")
    rowcount = node.get("rowcount")
    ref = node.get("ref", "")
    children = node.get("children") or []

    parts: list[str] = [f"- {role}"]
    if isinstance(name, str) and name:
        parts.append(f'"{name}"')
    if isinstance(value, str) and value:
        parts.append(f'= "{value}"')
    if isinstance(level, int):
        parts.append(f"[level={level}]")
    if isinstance(colcount, int):
        parts.append(f"[cols={colcount}]")
    if isinstance(rowcount, int):
        parts.append(f"[rows={rowcount}]")
    if isinstance(state, list):
        for s in state:
            if isinstance(s, str) and s:
                parts.append(f"[{s}]")
    if isinstance(ref, str) and ref:
        parts.append(f"[ref={ref}]")

    indent = "  " * depth
    line = indent + " ".join(parts)
    if children:
        line += ":"
    lines.append(line)

    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                _render_node(child, depth=depth + 1, lines=lines)


def _render_selection(selection: dict[str, Any], lines: list[str]) -> None:
    """Render an ``A11ySelection`` dict as a ``<selection>`` block.

    Emitted at the root of ``<ui_state>`` (no leading indent) so the
    LLM can spot it without parsing the tree::

        <selection ref="e42">
        the actual selected text
        </selection>

    No-op when the selection lacks a ``ref`` or ``text``.
    """
    ref = selection.get("ref")
    text = selection.get("text")
    if not isinstance(ref, str) or not ref:
        return
    if not isinstance(text, str) or not text:
        return
    lines.append(f'<selection ref="{ref}">')
    lines.append(text)
    lines.append("</selection>")

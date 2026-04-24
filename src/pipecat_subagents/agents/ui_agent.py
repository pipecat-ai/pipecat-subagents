#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UIAgent: LLM agent that dispatches UI events from the client."""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any

from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import LLMService

from pipecat_subagents.agents.llm_agent import LLMAgent
from pipecat_subagents.agents.ui_event_decorator import _collect_ui_event_handlers
from pipecat_subagents.bus import AgentBus
from pipecat_subagents.bus.messages import (
    UI_SNAPSHOT_EVENT_NAME,
    BusMessage,
    BusTaskRequestMessage,
    BusUICommandMessage,
    BusUIEventMessage,
)


class UIAgent(LLMAgent):
    """LLM agent that dispatches UI events from the client.

    Receives ``BusUIEventMessage`` (republished by ``attach_ui_bridge``)
    and dispatches each one to the matching ``@on_ui_event(name)``
    handler. By default every event is also appended to the LLM context
    as ``<ui_event name="...">payload</ui_event>`` so the agent can
    reason about what the user just did on the next inference.

    Subclasses provide an LLM via ``build_llm()`` and declare handlers
    with ``@on_ui_event(name)``:

    Example::

        class MyUIAgent(UIAgent):
            def build_llm(self) -> LLMService:
                return OpenAILLMService(api_key="...")

            @on_ui_event("nav_click")
            async def on_nav(self, message):
                view = message.payload.get("view")
                ...

    Visibility convention: when the client captures snapshots with
    ``trackViewport`` enabled (the default), every node in the
    accessibility tree whose bounding rect sits fully outside the
    viewport carries ``"offscreen"`` in its ``state`` list. The
    rendered ``<ui_state>`` surfaces this as ``[offscreen]``.
    ``visible_nodes()`` returns the filtered subset the user is
    actually looking at right now. Agents should treat ``[offscreen]``
    nodes as on-the-page-but-not-in-view, typically issuing a
    ``scroll_to`` command before acting on them.
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = False,
        bridged: tuple[str, ...] | None = None,
        inject_events: bool = True,
        auto_inject_ui_state: bool = True,
    ):
        """Initialize the UIAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active. Defaults to False.
            bridged: Bridge configuration. See ``BaseAgent`` for details.
            inject_events: Whether to auto-append each UI event to the
                LLM context as a ``<ui_event>`` developer message.
                Defaults to True. Override ``render_ui_event`` to change
                the injected content, or set this to False to disable.
            auto_inject_ui_state: When True (the default), the latest
                ``<ui_state>`` snapshot is appended to the LLM context
                at the start of every task request, so the agent always
                reasons over the current screen. Set to False if you
                want to call ``inject_ui_state()`` yourself.
        """
        super().__init__(name, bus=bus, active=active, bridged=bridged)
        self._inject_events = inject_events
        self._auto_inject_ui_state = auto_inject_ui_state
        self._ui_event_handlers = _collect_ui_event_handlers(self)
        # Latest accessibility snapshot received from the client. Updated
        # in ``on_bus_message`` when a ``__ui_snapshot`` event arrives.
        # Rendered into LLM context via ``inject_ui_state``.
        self._latest_snapshot: dict[str, Any] | None = None

    @abstractmethod
    def build_llm(self) -> LLMService:
        """Return the LLM service for this agent.

        Returns:
            An `LLMService` instance.
        """
        pass

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

                - A dataclass instance (including the built-ins in
                  ``pipecat_subagents.agents.ui_commands``). Converted
                  to a plain dict with ``dataclasses.asdict``.
                - A ``dict`` forwarded as-is.
                - ``None``, forwarded as an empty dict.
        """
        if payload is None:
            serialized: Any = {}
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

    async def on_bus_message(self, message: BusMessage) -> None:
        """Dispatch UI events alongside base lifecycle handling."""
        await super().on_bus_message(message)

        if not isinstance(message, BusUIEventMessage):
            return
        if message.target and message.target != self.name:
            return

        # Reserved snapshot event: store and return without dispatch or
        # ``<ui_event>`` injection. Apps render via ``inject_ui_state``.
        if message.event_name == UI_SNAPSHOT_EVENT_NAME:
            if isinstance(message.payload, dict):
                self._latest_snapshot = message.payload
            return

        await self._handle_ui_event(message)

    async def on_task_request(self, message: BusTaskRequestMessage) -> None:
        """Auto-inject the latest ``<ui_state>`` before task dispatch.

        Runs before the LLM sees the task payload, so the agent
        always reasons over the current screen. Disable via
        ``auto_inject_ui_state=False`` if the app wants to drive
        injection manually.
        """
        await super().on_task_request(message)
        if self._auto_inject_ui_state:
            await self.inject_ui_state()

    def render_ui_state(self) -> str:
        """Render the latest accessibility snapshot as a ``<ui_state>`` block.

        Produces Playwright-MCP-style indented text with stable element
        refs. Apps inject the output via ``inject_ui_state()`` when they
        want the LLM to see what's on screen.

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

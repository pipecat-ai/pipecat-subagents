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
    BusMessage,
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
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = False,
        bridged: tuple[str, ...] | None = None,
        inject_events: bool = True,
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
        """
        super().__init__(name, bus=bus, active=active, bridged=bridged)
        self._inject_events = inject_events
        self._ui_event_handlers = _collect_ui_event_handlers(self)

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

        await self._handle_ui_event(message)

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

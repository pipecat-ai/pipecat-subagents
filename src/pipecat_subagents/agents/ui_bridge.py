#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bridge between the RTVI client-message channel and the agent bus.

Register with ``attach_ui_bridge(agent)`` from the root agent's
``on_ready`` hook. The bridge handles both directions:

- Inbound: client-side ``UIAgentClient.sendEvent(name, payload)`` calls
  are republished onto the bus as ``BusUIEventMessage`` for ``UIAgent``
  subscribers to dispatch.
- Outbound: ``UIAgent.send_command(name, payload)`` emits a
  ``BusUICommandMessage`` which the bridge translates into an
  ``RTVIServerMessageFrame`` and pushes through the root agent's
  pipeline for RTVI to deliver to the client.
"""

from __future__ import annotations

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.bus.messages import (
    UI_COMMAND_MESSAGE_TYPE,
    UI_EVENT_MESSAGE_TYPE,
    BusUICommandMessage,
    BusUIEventMessage,
)


def attach_ui_bridge(agent: BaseAgent, *, target: str | None = None) -> None:
    """Wire the root agent's pipeline to the UI Agent SDK wire protocol.

    Must be called after the pipeline task exists, typically from the
    root agent's ``on_ready`` hook. The root agent's pipeline task must
    be built with ``enable_rtvi=True``.

    Side effects:
        - Registers an ``on_client_message`` handler on RTVI. Messages
          whose ``type`` matches ``UI_EVENT_MESSAGE_TYPE`` are
          republished as ``BusUIEventMessage``; all other client
          messages pass through untouched.
        - Registers an ``on_bus_message`` handler on ``agent``. When a
          ``BusUICommandMessage`` arrives, the bridge pushes an
          ``RTVIServerMessageFrame`` downstream through ``agent``'s
          pipeline so RTVI delivers it to the client.

    Args:
        agent: The root agent owning the pipeline task and RTVI processor.
        target: Optional target agent name for the republished
            ``BusUIEventMessage``. When ``None`` (the default), the
            message is broadcast and every ``UIAgent`` on the bus sees
            it. Set this when multiple UIAgents coexist and only one
            should handle UI events.

    Raises:
        RuntimeError: If the agent's pipeline task has no RTVI processor.
    """
    rtvi = getattr(agent.pipeline_task, "rtvi", None)
    if rtvi is None:
        raise RuntimeError(
            f"Agent '{agent.name}' has no RTVI processor. "
            "Ensure build_pipeline_task creates the task with enable_rtvi=True "
            "before calling attach_ui_bridge."
        )

    @rtvi.event_handler("on_client_message")
    async def _on_client_message(_rtvi, msg):
        if msg.type != UI_EVENT_MESSAGE_TYPE:
            return
        data = msg.data
        if not isinstance(data, dict):
            return
        event_name = data.get("name")
        if not isinstance(event_name, str):
            return
        await agent.bus.send(
            BusUIEventMessage(
                source=agent.name,
                target=target,
                event_name=event_name,
                payload=data.get("payload"),
            )
        )

    @agent.event_handler("on_bus_message")
    async def _on_bus_message(_agent, message):
        if not isinstance(message, BusUICommandMessage):
            return
        frame = RTVIServerMessageFrame(
            data={
                "type": UI_COMMAND_MESSAGE_TYPE,
                "name": message.command_name,
                "payload": message.payload,
            }
        )
        await agent.queue_frame(frame)

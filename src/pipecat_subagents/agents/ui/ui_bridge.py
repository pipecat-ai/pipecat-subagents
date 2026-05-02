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
- Outbound: bus messages in the UI Agent SDK protocol are translated
  into ``RTVIServerMessageFrame`` envelopes and pushed through the
  root agent's pipeline. Two surfaces share this path:

  - Commands published by ``UIAgent.send_command(name, payload)`` are
    forwarded as ``ui.command`` envelopes.
  - Task lifecycle messages published by
    ``UIAgent.user_task_group(...)`` and the agent's automatic
    forwarding of task updates / responses are forwarded as
    ``ui.task`` envelopes with a ``kind`` discriminator.
"""

from __future__ import annotations

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.processors.frameworks.rtvi.models import (
    UI_COMMAND_MESSAGE_TYPE,
    UI_EVENT_MESSAGE_TYPE,
    UI_TASK_COMPLETED_KIND,
    UI_TASK_GROUP_COMPLETED_KIND,
    UI_TASK_GROUP_STARTED_KIND,
    UI_TASK_MESSAGE_TYPE,
    UI_TASK_UPDATE_KIND,
)

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.bus.messages import (
    BusUICommandMessage,
    BusUIEventMessage,
    BusUITaskCompletedMessage,
    BusUITaskGroupCompletedMessage,
    BusUITaskGroupStartedMessage,
    BusUITaskUpdateMessage,
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

    Note:
        RTVI's ``on_client_ready`` event fires the moment the client's
        handshake message arrives — that can land before
        ``on_ready`` runs. Register ``on_client_ready`` handlers in
        ``build_pipeline_task`` (after creating the task) rather than
        ``on_ready`` to avoid a registration-vs-event race that drops
        the event silently.
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
        if isinstance(message, BusUICommandMessage):
            data = {
                "type": UI_COMMAND_MESSAGE_TYPE,
                "name": message.command_name,
                "payload": message.payload,
            }
        elif isinstance(message, BusUITaskGroupStartedMessage):
            data = {
                "type": UI_TASK_MESSAGE_TYPE,
                "kind": UI_TASK_GROUP_STARTED_KIND,
                "task_id": message.task_id,
                "agents": list(message.agents or []),
                "label": message.label,
                "cancellable": message.cancellable,
                "at": message.at,
            }
        elif isinstance(message, BusUITaskUpdateMessage):
            data = {
                "type": UI_TASK_MESSAGE_TYPE,
                "kind": UI_TASK_UPDATE_KIND,
                "task_id": message.task_id,
                "agent_name": message.agent_name,
                "data": message.data,
                "at": message.at,
            }
        elif isinstance(message, BusUITaskCompletedMessage):
            data = {
                "type": UI_TASK_MESSAGE_TYPE,
                "kind": UI_TASK_COMPLETED_KIND,
                "task_id": message.task_id,
                "agent_name": message.agent_name,
                "status": message.status,
                "response": message.response,
                "at": message.at,
            }
        elif isinstance(message, BusUITaskGroupCompletedMessage):
            data = {
                "type": UI_TASK_MESSAGE_TYPE,
                "kind": UI_TASK_GROUP_COMPLETED_KIND,
                "task_id": message.task_id,
                "at": message.at,
            }
        else:
            return
        await agent.queue_frame(RTVIServerMessageFrame(data=data))

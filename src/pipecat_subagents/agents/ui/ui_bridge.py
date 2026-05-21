#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bridge between the RTVI UI message channel and the agent bus.

Apply the ``@ui_agent(...)`` class decorator to the root agent that
owns the transport and RTVI processor. The decorator wires the bridge
when the agent becomes ready. The bridge handles both directions:

- Inbound: typed UI messages from the client (``ui-event``,
  ``ui-snapshot``, ``ui-cancel-task``) are republished onto the bus
  as ``BusUIEventMessage`` for ``UIAgent`` subscribers to dispatch.
  The wire-level type is collapsed onto the bus to a single carrier
  with an ``event_name`` discriminator; ``ui-snapshot`` and
  ``ui-cancel-task`` map to subagents-internal event names.
- Outbound: bus messages in the UI Agent SDK protocol are translated
  into the matching pipecat RTVI frames (``RTVIUICommandFrame``,
  ``RTVIUITaskFrame``) and pushed through the root agent's pipeline.
  The RTVI observer wraps each frame into the corresponding typed
  RTVI envelope (``UICommandMessage``, ``UITaskMessage``) and sends
  it to the client.
"""

from __future__ import annotations

import functools

from pipecat.processors.frameworks.rtvi.frames import (
    RTVIUICommandFrame,
    RTVIUITaskFrame,
)
from pipecat.processors.frameworks.rtvi.models import (
    UICancelTaskMessage,
    UIEventMessage,
    UISnapshotMessage,
    UITaskCompletedData,
    UITaskGroupCompletedData,
    UITaskGroupStartedData,
    UITaskUpdateData,
)

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.agents.ui.ui_messages import (
    _UI_CANCEL_TASK_BUS_EVENT_NAME,
    _UI_SNAPSHOT_BUS_EVENT_NAME,
    BusUICommandMessage,
    BusUIEventMessage,
    BusUITaskCompletedMessage,
    BusUITaskGroupCompletedMessage,
    BusUITaskGroupStartedMessage,
    BusUITaskUpdateMessage,
)


def _attach_ui_bridge(agent: BaseAgent, *, targets: list[str | None] | None = None) -> None:
    """Wire the root agent's pipeline to the UI Agent SDK wire protocol.

    Internal helper driven by the ``@ui_agent`` class decorator. Must
    run after the pipeline task exists; the decorator calls it from the
    agent's ``on_ready`` hook. The root agent's pipeline task must be
    built with ``enable_rtvi=True``.

    Side effects:
        - Registers an ``on_ui_message`` handler on RTVI. Typed UI
          messages from the client (``ui-event``, ``ui-snapshot``,
          ``ui-cancel-task``) are republished as ``BusUIEventMessage``,
          one per entry in ``targets``; ``ui-snapshot`` and
          ``ui-cancel-task`` collapse to subagents-internal event names
          so ``UIAgent`` can dispatch them through its existing
          bus-event path.
        - Registers an ``on_bus_message`` handler on ``agent``. When a
          ``BusUICommandMessage`` or any of the four ``BusUITask*``
          messages arrive, the bridge pushes the matching typed RTVI
          envelope downstream so RTVI delivers it to the client.

    Args:
        agent: The root agent owning the pipeline task and RTVI processor.
        targets: Target agent names for the republished
            ``BusUIEventMessage``. Each inbound UI message fans out to
            one ``BusUIEventMessage`` per entry. ``None`` (or a ``None``
            entry) broadcasts so every ``UIAgent`` on the bus sees it.
            Defaults to ``[None]`` (broadcast).

    Raises:
        RuntimeError: If the agent's pipeline task has no RTVI processor.

    Note:
        RTVI's ``on_client_ready`` event fires the moment the client's
        handshake message arrives, which can land before ``on_ready``
        runs. Register ``on_client_ready`` handlers in
        ``build_pipeline_task`` (after creating the task) rather than
        ``on_ready`` to avoid a registration-vs-event race that drops
        the event silently.
    """
    targets = targets if targets is not None else [None]
    rtvi = getattr(agent.pipeline_task, "rtvi", None)
    if rtvi is None:
        raise RuntimeError(
            f"Agent '{agent.name}' has no RTVI processor. "
            "Ensure build_pipeline_task creates the task with enable_rtvi=True "
            "before applying @ui_agent."
        )

    @rtvi.event_handler("on_ui_message")
    async def _on_ui_message(_rtvi, message):
        # All three inbound UI envelopes collapse to BusUIEventMessage on
        # the bus, with an event_name discriminator. UIAgent's bus
        # dispatch already routes on event_name; using two
        # subagents-internal names for the snapshot and cancel-task
        # cases keeps app @on_ui_event handlers from colliding.
        if isinstance(message, UIEventMessage):
            event_name = message.data.event
            payload = message.data.payload
        elif isinstance(message, UISnapshotMessage):
            event_name = _UI_SNAPSHOT_BUS_EVENT_NAME
            payload = message.data.tree.model_dump(exclude_none=True)
        elif isinstance(message, UICancelTaskMessage):
            event_name = _UI_CANCEL_TASK_BUS_EVENT_NAME
            payload = {"task_id": message.data.task_id, "reason": message.data.reason}
        else:
            return
        for target in targets:
            await agent.bus.send(
                BusUIEventMessage(
                    source=agent.name,
                    target=target,
                    event_name=event_name,
                    payload=payload,
                )
            )

    @agent.event_handler("on_bus_message")
    async def _on_bus_message(_agent, message):
        frame = None
        if isinstance(message, BusUICommandMessage):
            frame = RTVIUICommandFrame(
                command=message.command_name,
                payload=message.payload,
            )
        elif isinstance(message, BusUITaskGroupStartedMessage):
            frame = RTVIUITaskFrame(
                data=UITaskGroupStartedData(
                    task_id=message.task_id,
                    agents=list(message.agents or []),
                    label=message.label,
                    cancellable=message.cancellable,
                    at=message.at,
                )
            )
        elif isinstance(message, BusUITaskUpdateMessage):
            frame = RTVIUITaskFrame(
                data=UITaskUpdateData(
                    task_id=message.task_id,
                    agent_name=message.agent_name,
                    data=message.data,
                    at=message.at,
                )
            )
        elif isinstance(message, BusUITaskCompletedMessage):
            frame = RTVIUITaskFrame(
                data=UITaskCompletedData(
                    task_id=message.task_id,
                    agent_name=message.agent_name,
                    status=message.status,
                    response=message.response,
                    at=message.at,
                )
            )
        elif isinstance(message, BusUITaskGroupCompletedMessage):
            frame = RTVIUITaskFrame(
                data=UITaskGroupCompletedData(
                    task_id=message.task_id,
                    at=message.at,
                )
            )
        if frame is None:
            return
        await agent.queue_frame(frame)


def ui_agent(*agent_names: str):
    """Class decorator: bridge a root agent's pipeline to its UIAgents.

    Apply to the root agent class that owns the transport and RTVI
    processor. When the agent becomes ready, the decorator wires the
    first-class UI RTVI channels to the agent bus in both directions
    (see the module docstring), routing inbound client events and
    snapshots to each named ``UIAgent``.

    The root agent's pipeline task must be built with
    ``enable_rtvi=True``.

    Example::

        @ui_agent("assistant")
        class AppRoot(BaseAgent):
            ...

        # Multiple UIAgents on the same transport:
        @ui_agent("planner", "researcher")
        class AppRoot(BaseAgent):
            ...

    Args:
        *agent_names: Names of the ``UIAgent`` instances that should
            receive inbound UI events and snapshots. Each inbound
            message is delivered to every listed agent. With no names,
            messages broadcast to every ``UIAgent`` on the bus.

    Returns:
        The decorated class, with ``on_ready`` wrapped to wire the
        bridge after the original hook runs.
    """

    def decorate(cls: type[BaseAgent]) -> type[BaseAgent]:
        original_on_ready = cls.on_ready

        @functools.wraps(original_on_ready)
        async def on_ready(self) -> None:
            await original_on_ready(self)
            _attach_ui_bridge(self, targets=list(agent_names) or [None])

        cls.on_ready = on_ready
        return cls

    return decorate

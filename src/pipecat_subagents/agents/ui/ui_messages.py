#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Subagents-internal bus carriers for the UI Agent Protocol.

These dataclasses are the on-the-bus shape that ``UIAgent`` and the
bridge installed by ``attach_ui_bridge`` exchange. They are NOT the
on-the-wire format the client sees; that lives in
``pipecat.processors.frameworks.rtvi.models`` (since pipecat-ai
1.2.0). The bridge translates between the two.

- ``BusUIEventMessage`` and ``BusUICommandMessage`` carry client
  events and server commands respectively.
- ``BusUITaskGroupStartedMessage``, ``BusUITaskUpdateMessage``,
  ``BusUITaskCompletedMessage``, and ``BusUITaskGroupCompletedMessage``
  carry the four phases of a user-facing task group's lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipecat_subagents.bus.messages import BusDataMessage


@dataclass
class BusUIEventMessage(BusDataMessage):
    """A UI event sent from the client to a server-side agent.

    Emitted by ``attach_ui_bridge`` when the client dispatches an event
    via ``UIAgentClient.sendEvent(name, payload)``. ``UIAgent`` subclasses
    dispatch these to ``@on_ui_event(name)`` handlers.

    Parameters:
        event_name: App-defined event name.
        payload: App-defined payload. Schemaless by design.
    """

    event_name: str = ""
    payload: Any = None


@dataclass
class BusUICommandMessage(BusDataMessage):
    """A UI command sent from a server-side agent to the client.

    Published by ``UIAgent.send_command(name, payload)``. The bridge
    installed by ``attach_ui_bridge`` translates this to an
    ``RTVIServerMessageFrame`` with ``data == {"type": "ui.command",
    "name": command_name, "payload": payload}`` and pushes it through
    the root agent's pipeline.

    Parameters:
        command_name: App-defined command name.
        payload: App-defined payload (already a plain dict by the time
            it lands on the bus).
    """

    command_name: str = ""
    payload: Any = None


# ---------------------------------------------------------------------------
# UI task lifecycle
# ---------------------------------------------------------------------------


@dataclass
class BusUITaskGroupStartedMessage(BusDataMessage):
    """A user-facing task group has been dispatched.

    Published by ``UIAgent.user_task_group(...)`` on entry. The bridge
    forwards it to the client as a ``ui.task`` envelope with
    ``kind = "group_started"``.

    Parameters:
        task_id: Shared task identifier for the group.
        agents: Names of the agents the work was dispatched to.
        label: Optional human-readable label for the group.
        cancellable: Whether the client may request cancellation.
        at: Epoch milliseconds when the group started.
    """

    task_id: str = ""
    agents: list[str] | None = None
    label: str | None = None
    cancellable: bool = True
    at: int = 0


@dataclass
class BusUITaskUpdateMessage(BusDataMessage):
    """Per-task progress for a user-facing task group.

    Forwarded by the ``UIAgent`` whenever a worker emits a
    ``BusTaskUpdateMessage`` whose ``task_id`` matches a registered
    user task group. The bridge forwards to the client as a
    ``ui.task`` envelope with ``kind = "task_update"``.

    Parameters:
        task_id: The shared task identifier.
        agent_name: The worker that produced the update.
        data: The worker's update payload, forwarded verbatim.
        at: Epoch milliseconds when the update was emitted on the bus.
    """

    task_id: str = ""
    agent_name: str = ""
    data: Any = None
    at: int = 0


@dataclass
class BusUITaskCompletedMessage(BusDataMessage):
    """A worker in a user-facing task group has completed.

    Forwarded by the ``UIAgent`` whenever a worker's
    ``BusTaskResponseMessage`` arrives for a registered user task
    group. The bridge forwards to the client as a ``ui.task`` envelope
    with ``kind = "task_completed"``.

    Parameters:
        task_id: The shared task identifier.
        agent_name: The worker that produced the response.
        status: Completion status as a string (``TaskStatus`` value).
        response: The worker's response payload.
        at: Epoch milliseconds when the response was received.
    """

    task_id: str = ""
    agent_name: str = ""
    status: str = ""
    response: Any = None
    at: int = 0


@dataclass
class BusUITaskGroupCompletedMessage(BusDataMessage):
    """A user-facing task group has completed.

    Published when ``UIAgent.user_task_group(...)`` exits, after every
    worker has responded (or the group has been cancelled). The bridge
    forwards to the client as a ``ui.task`` envelope with
    ``kind = "group_completed"``.

    Parameters:
        task_id: The shared task identifier.
        at: Epoch milliseconds when the group completed.
    """

    task_id: str = ""
    at: int = 0


__all__ = [
    "BusUICommandMessage",
    "BusUIEventMessage",
    "BusUITaskCompletedMessage",
    "BusUITaskGroupCompletedMessage",
    "BusUITaskGroupStartedMessage",
    "BusUITaskUpdateMessage",
]

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ClowderAgent: bus-level observability agent.

Subscribes to the agent bus and streams all bus messages to connected
web clients over WebSocket. Tracks agent state (ready, active, parent)
and task state (status, timing) to provide a snapshot on connect.
"""

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserSpeakingFrame,
)
from pydantic import BaseModel
from websockets import ConnectionClosedOK, serve

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.bus.bus import AgentBus
from pipecat_subagents.bus.messages import (
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusAgentReadyMessage,
    BusAgentRegistryMessage,
    BusCancelAgentMessage,
    BusDeactivateAgentMessage,
    BusEndAgentMessage,
    BusFrameMessage,
    BusMessage,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
    BusTaskResponseMessage,
    BusTaskResponseUrgentMessage,
    BusTaskStreamDataMessage,
    BusTaskStreamEndMessage,
    BusTaskStreamStartMessage,
    BusTaskUpdateMessage,
    BusTaskUpdateUrgentMessage,
)


@dataclass
class AgentInfo:
    """Tracked state for a known agent."""

    name: str
    parent: str | None = None
    runner: str | None = None
    active: bool = False
    ready: bool = False
    bridged: bool = False
    started_at: float | None = None


@dataclass
class TaskInfo:
    """Tracked state for a task group."""

    task_id: str
    source: str
    targets: list[str] = field(default_factory=list)
    task_name: str | None = None
    status: str = "running"
    started_at: float = 0.0
    completed_at: float | None = None


_TASK_MESSAGES = (
    BusTaskRequestMessage,
    BusTaskResponseMessage,
    BusTaskResponseUrgentMessage,
    BusTaskUpdateMessage,
    BusTaskUpdateUrgentMessage,
    BusTaskCancelMessage,
    BusTaskStreamStartMessage,
    BusTaskStreamDataMessage,
    BusTaskStreamEndMessage,
)

_LIFECYCLE_MESSAGES = (
    BusActivateAgentMessage,
    BusDeactivateAgentMessage,
    BusEndAgentMessage,
    BusCancelAgentMessage,
    BusAgentReadyMessage,
    BusAgentRegistryMessage,
    BusAddAgentMessage,
)


def _categorize(message: BusMessage) -> str:
    if isinstance(message, BusFrameMessage):
        return "frame"
    if isinstance(message, _TASK_MESSAGES):
        return "task"
    if isinstance(message, _LIFECYCLE_MESSAGES):
        return "lifecycle"
    return "other"


def _serialize_value(obj: Any) -> Any:
    """Recursively convert an object to a JSON-safe value."""
    if obj is None or isinstance(obj, (int, float, bool, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _serialize_value(getattr(obj, f.name))
            for f in obj.__dataclass_fields__.values()
            if getattr(obj, f.name) is not None
        }
    if isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize_value(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, BaseModel):
        return _serialize_value(obj.model_dump())
    return f"<{type(obj).__name__}>"


def _serialize_message(message: BusMessage) -> dict:
    """Convert a bus message to a JSON-safe dict for the web client."""
    msg_type = type(message).__name__
    category = _categorize(message)

    data: dict[str, Any] = {}

    # Extract fields specific to each message type, skipping base fields
    if isinstance(message, BusFrameMessage):
        data["frame_type"] = type(message.frame).__name__
        data["direction"] = message.direction.name.lower()
        if message.bridge:
            data["bridge"] = message.bridge
        frame_data = _serialize_value(message.frame)
        if isinstance(frame_data, dict):
            data["frame"] = frame_data
    elif isinstance(message, BusActivateAgentMessage):
        if message.args:
            data["args"] = _serialize_value(message.args)
    elif isinstance(message, (BusEndAgentMessage, BusCancelAgentMessage)):
        if hasattr(message, "reason") and message.reason:
            data["reason"] = message.reason
    elif isinstance(message, BusAgentReadyMessage):
        data["runner"] = message.runner
        if message.parent:
            data["parent"] = message.parent
        data["active"] = message.active
        data["bridged"] = message.bridged
        if message.started_at:
            data["started_at"] = message.started_at
    elif isinstance(message, BusAgentRegistryMessage):
        data["runner"] = message.runner
        data["agents"] = [_serialize_value(e) for e in message.agents]
    elif isinstance(message, BusAddAgentMessage):
        data["agent_name"] = message.agent.name
    elif isinstance(message, BusTaskRequestMessage):
        data["task_id"] = message.task_id
        if message.task_name:
            data["task_name"] = message.task_name
        if message.payload:
            data["payload"] = _serialize_value(message.payload)
    elif isinstance(message, (BusTaskResponseMessage, BusTaskResponseUrgentMessage)):
        data["task_id"] = message.task_id
        data["status"] = message.status.value
        if message.response:
            data["response"] = _serialize_value(message.response)
    elif isinstance(message, (BusTaskUpdateMessage, BusTaskUpdateUrgentMessage)):
        data["task_id"] = message.task_id
        if message.update:
            data["update"] = _serialize_value(message.update)
    elif isinstance(message, BusTaskCancelMessage):
        data["task_id"] = message.task_id
        if message.reason:
            data["reason"] = message.reason
    elif isinstance(
        message, BusTaskStreamStartMessage | BusTaskStreamDataMessage | BusTaskStreamEndMessage
    ):
        data["task_id"] = message.task_id
        if message.data:
            data["data"] = _serialize_value(message.data)

    return {
        "type": "bus_message",
        "timestamp": time.time(),
        "message_type": msg_type,
        "category": category,
        "source": message.source,
        "target": message.target,
        "data": data,
    }


class ClowderAgent(BaseAgent):
    """Observability agent that streams bus messages to web clients.

    Subscribes to the agent bus and captures every message. Connected
    web clients receive a snapshot of current state on connect, then
    a continuous stream of bus message events.

    Example::

        clowder = ClowderAgent("clowder", bus=bus, port=7070)
        runner.add_agent(clowder)
    """

    # No frame types excluded by default.
    DEFAULT_EXCLUDE_FRAMES: tuple[type[Frame], ...] = (
        InputAudioRawFrame,
        OutputAudioRawFrame,
        UserSpeakingFrame,
        BotSpeakingFrame,
    )

    def __init__(
        self,
        name: str = "clowder",
        *,
        bus: AgentBus,
        host: str = "localhost",
        port: int = 7070,
        exclude_frames: tuple[type[Frame], ...] | None = None,
    ):
        """Initialize the ClowderAgent.

        Args:
            name: Agent name. Defaults to "clowder".
            bus: The `AgentBus` to observe.
            host: WebSocket server bind address. Defaults to "localhost".
            port: WebSocket server port. Defaults to 7070.
            exclude_frames: Frame types to skip. Defaults to
                high-frequency audio frames.
        """
        super().__init__(name, bus=bus)
        self._host = host
        self._port = port
        self._exclude_frames = exclude_frames or self.DEFAULT_EXCLUDE_FRAMES

        # Tracked state
        self._agents: dict[str, AgentInfo] = {}
        self._tasks: dict[str, TaskInfo] = {}
        self._event_history: list[dict] = []

        # WebSocket clients
        self._clients: set = set()
        self._server = None

    async def on_ready(self) -> None:
        """Start the WebSocket server."""
        await super().on_ready()
        self._server = await serve(self._ws_handler, self._host, self._port)
        logger.info(f"ᓚᘏᗢ Clowder: server listening on ws://{self._host}:{self._port}")

    async def cleanup(self) -> None:
        """Shut down the WebSocket server."""
        logger.info("ᓚᘏᗢ Clowder: closing server")
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        await super().cleanup()

    async def on_bus_message(self, message: BusMessage) -> None:
        """Capture every bus message, update state, and broadcast to clients.

        Overrides the base implementation to see ALL messages, including
        those targeted at other agents. High-frequency audio frames are
        skipped to avoid flooding the connection.
        """
        if isinstance(message, BusFrameMessage) and isinstance(message.frame, self._exclude_frames):
            return

        # Update internal state
        self._update_state(message)

        # Serialize, buffer, and broadcast
        event = _serialize_message(message)
        self._event_history.append(event)
        await self._broadcast(event)

        await super().on_bus_message(message)

    def _update_state(self, message: BusMessage) -> None:
        now = time.time()

        # Track agents from ready announcements (root and child)
        if isinstance(message, BusAgentReadyMessage):
            info = self._agents.setdefault(message.source, AgentInfo(name=message.source))
            info.runner = message.runner
            info.parent = message.parent
            info.active = message.active
            info.ready = True
            info.bridged = message.bridged
            info.started_at = message.started_at

        # Track agents from registry announcements (batch snapshot)
        elif isinstance(message, BusAgentRegistryMessage):
            for entry in message.agents:
                info = self._agents.setdefault(entry.name, AgentInfo(name=entry.name))
                info.runner = message.runner
                info.parent = entry.parent
                info.active = entry.active
                info.bridged = entry.bridged
                info.started_at = entry.started_at
                info.ready = True

        # Track agents from add-agent (child agents, local only)
        elif isinstance(message, BusAddAgentMessage):
            child_name = message.agent.name
            info = self._agents.setdefault(child_name, AgentInfo(name=child_name))
            info.parent = message.source

        # Track activation state
        elif isinstance(message, BusActivateAgentMessage) and message.target:
            info = self._agents.setdefault(message.target, AgentInfo(name=message.target))
            info.active = True

        elif isinstance(message, BusDeactivateAgentMessage) and message.target:
            info = self._agents.get(message.target)
            if info:
                info.active = False

        # Track task lifecycle
        elif isinstance(message, BusTaskRequestMessage) and message.target:
            task = self._tasks.get(message.task_id)
            if task:
                task.targets.append(message.target)
            else:
                self._tasks[message.task_id] = TaskInfo(
                    task_id=message.task_id,
                    source=message.source,
                    targets=[message.target],
                    task_name=message.task_name,
                    started_at=now,
                )

        elif isinstance(message, (BusTaskResponseMessage, BusTaskResponseUrgentMessage)):
            task = self._tasks.get(message.task_id)
            if task:
                remaining = set(task.targets) - {message.source}
                # Check if all targets have responded
                responded = set()
                for t in task.targets:
                    if t == message.source or t not in remaining:
                        responded.add(t)
                if responded == set(task.targets):
                    task.status = message.status.value
                    task.completed_at = now

        elif isinstance(message, BusTaskCancelMessage):
            task = self._tasks.get(message.task_id)
            if task:
                task.status = "cancelled"
                task.completed_at = now

    async def _ws_handler(self, websocket) -> None:
        self._clients.add(websocket)
        logger.info(f"ᓚᘏᗢ Clowder: client connected ({len(self._clients)} total)")
        try:
            # Send current state snapshot
            snapshot = self._build_snapshot()
            await websocket.send(json.dumps(snapshot))

            # Keep connection alive (read-only, ignore client messages)
            async for _ in websocket:
                pass
        except ConnectionClosedOK:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info(f"ᓚᘏᗢ Clowder: client disconnected ({len(self._clients)} total)")

    def _build_snapshot(self) -> dict:
        return {
            "type": "snapshot",
            "timestamp": time.time(),
            "agents": {name: asdict(info) for name, info in self._agents.items()},
            "tasks": {tid: asdict(info) for tid, info in self._tasks.items()},
            "events": self._event_history,
        }

    async def _broadcast(self, event: dict) -> None:
        if not self._clients:
            return
        data = json.dumps(event)
        dead = set()
        for client in self._clients:
            try:
                await client.send(data)
            except Exception:
                dead.add(client)
        self._clients -= dead

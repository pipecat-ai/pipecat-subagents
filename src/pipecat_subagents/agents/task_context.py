#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Task group types for structured concurrent task execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pipecat_subagents.agents.base_agent import BaseAgent


class TaskStatus(StrEnum):
    """Status of a completed task.

    Inherits from ``str`` so values compare naturally with plain strings
    and serialize without extra handling.

    Attributes:
        COMPLETED: The task finished successfully.
        CANCELLED: The task was cancelled by the requester.
        FAILED: The task failed due to a logical or business error.
        ERROR: The task encountered an unexpected runtime error.
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ERROR = "error"


class TaskError(Exception):
    """Raised when a task is cancelled due to a worker error or timeout."""

    pass


class TaskGroupError(Exception):
    """Raised when a task group is cancelled due to a worker error or timeout."""

    pass


@dataclass
class TaskGroupResponse:
    """Collected results from a completed task group.

    Parameters:
        task_id: The shared task identifier.
        responses: Collected responses keyed by agent name.
    """

    task_id: str
    responses: dict[str, dict]


@dataclass
class TaskEvent:
    """An event received from a worker during single-agent task execution.

    Parameters:
        type: The event type.
        data: Optional event payload.
    """

    UPDATE: ClassVar[str] = "update"
    STREAM_START: ClassVar[str] = "stream_start"
    STREAM_DATA: ClassVar[str] = "stream_data"
    STREAM_END: ClassVar[str] = "stream_end"

    type: str
    data: dict | None = None


@dataclass
class TaskGroupEvent:
    """An event received from a worker during task group execution.

    Parameters:
        type: The event type.
        agent_name: The name of the agent that sent the event.
        data: Optional event payload.
    """

    UPDATE: ClassVar[str] = "update"
    STREAM_START: ClassVar[str] = "stream_start"
    STREAM_DATA: ClassVar[str] = "stream_data"
    STREAM_END: ClassVar[str] = "stream_end"

    type: str
    agent_name: str
    data: dict | None = None


@dataclass
class TaskGroup:
    """Tracks a group of task agents launched together.

    Parameters:
        task_id: Shared identifier for all agents in this group.
        agent_names: Names of the agents in the group.
        responses: Collected responses keyed by agent name.
        timeout_task: Optional asyncio task that cancels the group on timeout.
        cancel_on_error: Whether to cancel the group if a worker errors.
        event_queue: Optional queue for streaming events to a
            ``TaskGroupContext`` async iterator.
    """

    task_id: str
    agent_names: set[str]
    responses: dict[str, dict] = field(default_factory=dict)
    timeout_task: asyncio.Task | None = None
    cancel_on_error: bool = True
    event_queue: asyncio.Queue | None = field(default=None, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _error: str | None = field(default=None, repr=False)

    @property
    def is_done(self) -> bool:
        """Whether the group has completed or failed."""
        return self._done.is_set()

    async def wait(self) -> None:
        """Wait for all agents in the group to respond.

        Raises:
            TaskGroupError: If the group was cancelled due to error or timeout.
        """
        await self._done.wait()
        if self._error:
            raise TaskGroupError(self._error)

    def complete(self) -> None:
        """Signal that all agents have responded."""
        self._done.set()
        if self.event_queue:
            self.event_queue.put_nowait(None)

    def fail(self, reason: str | None = None) -> None:
        """Signal that the group was cancelled.

        Args:
            reason: Human-readable reason for the failure.
        """
        self._error = reason
        self._done.set()
        if self.event_queue:
            self.event_queue.put_nowait(None)


class TaskGroupContext:
    """Async context manager and iterator for structured task group execution.

    Sends task requests on enter, waits for all responses on exit.
    Supports ``async for`` to receive intermediate events (updates
    and streaming data) from workers while waiting for completion.

    On normal completion, results are available via ``responses``.
    On worker error (with ``cancel_on_error=True``) or timeout, raises
    ``TaskGroupError``. If the ``async with`` block raises, remaining
    tasks are cancelled.

    Example::

        async with self.task_group("w1", "w2", payload=data) as tg:
            async for event in tg:
                print(f"{event.agent_name} [{event.type}]: {event.data}")

        for name, result in tg.responses.items():
            print(name, result)
    """

    def __init__(
        self,
        agent: BaseAgent,
        agent_names: tuple[str, ...],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
    ):
        """Initialize the TaskGroupContext.

        Args:
            agent: The parent `BaseAgent` that owns this task group.
            agent_names: Names of the agents to send the task to.
            name: Optional task name for routing to named handlers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and task execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
        """
        self._agent = agent
        self._agent_names = agent_names
        self._name = name
        self._payload = payload
        self._timeout = timeout
        self._cancel_on_error = cancel_on_error
        self._group: TaskGroup | None = None

    @property
    def task_id(self) -> str:
        """The shared task identifier for this group."""
        if not self._group:
            raise RuntimeError("Task group has not been started")
        return self._group.task_id

    @property
    def responses(self) -> dict[str, dict]:
        """Collected responses keyed by agent name."""
        if not self._group:
            raise RuntimeError("Task group has not been started")
        return self._group.responses

    def __aiter__(self):
        return self

    async def __anext__(self) -> TaskGroupEvent:
        if not self._group or not self._group.event_queue:
            raise StopAsyncIteration
        event = await self._group.event_queue.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def __aenter__(self) -> TaskGroupContext:
        self._group = await self._agent.create_task_group_and_request_task(
            list(self._agent_names),
            name=self._name,
            payload=self._payload,
            timeout=self._timeout,
            cancel_on_error=self._cancel_on_error,
        )
        self._group.event_queue = asyncio.Queue()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            if self._group and self._group.task_id in self._agent.task_groups:
                # Shield the cleanup so it completes even if the
                # surrounding task is being cancelled (e.g. tool
                # interruption).
                await asyncio.shield(
                    self._agent.cancel_task(self._group.task_id, reason="context exited with error")
                )
            return False

        await self._group.wait()
        return False


class TaskContext:
    """Async context manager and iterator for a single-agent task.

    Sends a task request on enter, waits for the response on exit.
    Supports ``async for`` to receive intermediate events (updates
    and streaming data) from the worker while waiting for completion.

    On normal completion, the result is available via ``response``.
    On worker error or timeout, raises ``TaskError``. If the
    ``async with`` block raises, the task is cancelled.

    Example::

        async with self.task("worker", payload=data) as t:
            async for event in t:
                print(f"[{event.type}]: {event.data}")

        print(t.response)
    """

    def __init__(
        self,
        agent: BaseAgent,
        agent_name: str,
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
    ):
        """Initialize the TaskContext.

        Args:
            agent: The parent `BaseAgent` that owns this task.
            agent_name: Name of the agent to send the task to.
            name: Optional task name for routing to a named handler.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and task execution.
        """
        self._agent = agent
        self._agent_name = agent_name
        self._name = name
        self._payload = payload
        self._timeout = timeout
        self._group: TaskGroup | None = None

    @property
    def task_id(self) -> str:
        """The task identifier."""
        if not self._group:
            raise RuntimeError("Task has not been started")
        return self._group.task_id

    @property
    def response(self) -> dict:
        """The worker's response payload."""
        if not self._group:
            raise RuntimeError("Task has not been started")
        return self._group.responses.get(self._agent_name, {})

    def __aiter__(self):
        return self

    async def __anext__(self) -> TaskEvent:
        if not self._group or not self._group.event_queue:
            raise StopAsyncIteration
        event = await self._group.event_queue.get()
        if event is None:
            raise StopAsyncIteration
        return TaskEvent(type=event.type, data=event.data)

    async def __aenter__(self) -> TaskContext:
        self._group = await self._agent.create_task_group_and_request_task(
            [self._agent_name],
            name=self._name,
            payload=self._payload,
            timeout=self._timeout,
            cancel_on_error=True,
        )
        self._group.event_queue = asyncio.Queue()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            if self._group and self._group.task_id in self._agent.task_groups:
                # Shield the cleanup so it completes even if the
                # surrounding task is being cancelled (e.g. tool
                # interruption).
                await asyncio.shield(
                    self._agent.cancel_task(self._group.task_id, reason="context exited with error")
                )
            return False

        try:
            await self._group.wait()
        except TaskGroupError as e:
            raise TaskError(str(e)) from e
        return False

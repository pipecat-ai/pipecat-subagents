#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-facing task group context.

Wraps ``TaskGroupContext`` so the work it dispatches is also surfaced
to the UI client through the UI Agent SDK protocol. Apps reach this
via ``UIAgent.user_task_group(...)`` rather than constructing it
directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pipecat_subagents.agents.task_context import TaskGroupContext
from pipecat_subagents.agents.ui.ui_messages import (
    BusUITaskGroupCompletedMessage,
    BusUITaskGroupStartedMessage,
)

if TYPE_CHECKING:
    from pipecat_subagents.agents.ui.ui_agent import UIAgent


class UserTaskGroupContext(TaskGroupContext):
    """Task group whose lifecycle is forwarded to the UI client.

    Behaves exactly like ``TaskGroupContext`` for the dispatching
    code. Additionally, on enter the context registers the group with
    its parent ``UIAgent`` and publishes a
    ``BusUITaskGroupStartedMessage``. The agent forwards any
    subsequent ``BusTaskUpdateMessage`` / ``BusTaskResponseMessage``
    whose ``task_id`` matches a registered group as
    ``BusUITaskUpdateMessage`` / ``BusUITaskCompletedMessage``. On
    exit the context publishes ``BusUITaskGroupCompletedMessage`` and
    deregisters.

    Workers don't need to know about the UI surface: any
    ``send_task_update`` they emit against the group's ``task_id`` is
    forwarded automatically.

    Example::

        async with self.user_task_group(
            "researcher_a", "researcher_b",
            payload={"query": query},
            label=f"Research: {query}",
            cancellable=True,
        ) as tg:
            async for event in tg:
                ...
            results = tg.responses
    """

    def __init__(
        self,
        agent: UIAgent,
        agent_names: tuple[str, ...],
        *,
        name: str | None = None,
        payload: dict | None = None,
        timeout: float | None = None,
        cancel_on_error: bool = True,
        label: str | None = None,
        cancellable: bool = True,
    ):
        """Initialize the UserTaskGroupContext.

        Args:
            agent: The parent ``UIAgent`` that owns this task group.
            agent_names: Names of the agents to send the task to.
            name: Optional task name for routing to named ``@task``
                handlers on the workers.
            payload: Optional structured data describing the work.
            timeout: Optional timeout in seconds covering both the
                ready-wait and task execution.
            cancel_on_error: Whether to cancel the group if a worker
                errors. Defaults to True.
            label: Optional human-readable label surfaced to the
                client (e.g. ``"Research: Radiohead"``). The client UI
                uses it to title the in-flight task card.
            cancellable: Whether the client may request cancellation
                of this group via the reserved ``__cancel_task`` event.
                Defaults to True.
        """
        super().__init__(
            agent,
            agent_names,
            name=name,
            payload=payload,
            timeout=timeout,
            cancel_on_error=cancel_on_error,
        )
        self._ui_agent = agent
        self._label = label
        self._cancellable = cancellable

    @property
    def label(self) -> str | None:
        """The group's human-readable label, if any."""
        return self._label

    @property
    def cancellable(self) -> bool:
        """Whether the client may request cancellation."""
        return self._cancellable

    async def __aenter__(self) -> UserTaskGroupContext:
        await super().__aenter__()
        task_id = self.task_id
        self._ui_agent._register_user_task_group(
            task_id=task_id,
            agent_names=list(self._agent_names),
            label=self._label,
            cancellable=self._cancellable,
        )
        await self._ui_agent.bus.send(
            BusUITaskGroupStartedMessage(
                source=self._ui_agent.name,
                target=None,
                task_id=task_id,
                agents=list(self._agent_names),
                label=self._label,
                cancellable=self._cancellable,
                at=int(time.time() * 1000),
            )
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        task_id = self._group.task_id if self._group else None
        try:
            return await super().__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if task_id:
                self._ui_agent._unregister_user_task_group(task_id)
                await self._ui_agent.bus.send(
                    BusUITaskGroupCompletedMessage(
                        source=self._ui_agent.name,
                        target=None,
                        task_id=task_id,
                        at=int(time.time() * 1000),
                    )
                )

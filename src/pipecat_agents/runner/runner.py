#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent runner for orchestrating multi-agent lifecycle and pipeline tasks."""

import asyncio
from typing import Optional

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from pipecat.utils.base_object import BaseObject

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import (
    AgentBus,
    BusAddAgentMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusMessage,
    LocalAgentBus,
)


class AgentRunner(BaseObject):
    """Lifecycle orchestrator for multi-agent systems.

    Manages agent lifecycle and coordinates pipeline tasks via
    `PipelineRunner`. On graceful end, sends `BusEndAgentMessage` to all
    agents in parallel — agents that need ordered shutdown (e.g. ending
    sub-agents before themselves) handle it in their own
    ``on_bus_message()``.

    Event handlers:

    on_runner_started(runner)
        Fired after all registered agents have been started.

    Example::

        runner = AgentRunner()

        cred = CREDAgent("cred", bus=runner.bus, ...)
        await runner.add_agent(cred)

        await runner.run()
    """

    def __init__(
        self,
        *,
        bus: Optional[AgentBus] = None,
        handle_sigint: bool = True,
    ):
        """Initialize the AgentRunner.

        Args:
            bus: Optional `AgentBus` instance. Creates a default one if None.
            handle_sigint: Whether `PipelineRunner` handles SIGINT.
                Defaults to True.
        """
        super().__init__()
        self._bus = bus or LocalAgentBus()

        self._running: bool = False
        self._agents: dict[str, BaseAgent] = {}
        self._running_agent_tasks: dict[str, asyncio.Task] = {}
        self._pipecat_runner = PipelineRunner(handle_sigint=handle_sigint)
        self._shutdown_event = asyncio.Event()

        self._register_event_handler("on_runner_started")

        @self._bus.event_handler("on_message")
        async def on_message(bus, message: BusMessage):
            await self._handle_bus_message(message)

    @property
    def bus(self) -> AgentBus:
        """The bus instance for agent communication."""
        return self._bus

    async def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to this runner.

        Can be called before or after run(). When called after run() has
        started, the agent's pipeline task is created and started immediately.

        Args:
            agent: The agent to add.

        Raises:
            ValueError: If an agent with this name already exists.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already exists")
        self._agents[agent.name] = agent
        logger.debug(f"AgentRunner: added agent '{agent.name}'")

        if self._running:
            await self._start_agent_task(agent)

    async def run(self) -> None:
        """Start all agents, block until shutdown.

        Starts all registered agents, fires ``on_runner_started``, then blocks
        until `end()` or `cancel()` is called. New agents can be added
        dynamically via `add_agent()` after ``run()`` has started.
        """
        self._running = True
        self._shutdown_event.clear()

        await self._bus.start()

        for agent in self._agents.values():
            await self._start_agent_task(agent)

        await self._call_event_handler("on_runner_started")

        await self._shutdown_event.wait()

        # Wait for remaining agent tasks to finish cleanup
        remaining = [t for t in self._running_agent_tasks.values() if not t.done()]
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)

        await self._bus.stop()
        self._running = False

    async def end(self, reason: Optional[str] = None) -> None:
        """Gracefully end all agent pipelines and shut down.

        Sends `BusEndAgentMessage` to all agents in parallel, waits for
        their pipelines to finish, then shuts down. Agents that need
        ordered shutdown handle it themselves. Idempotent — subsequent
        calls are ignored.

        Args:
            reason: Optional human-readable reason for ending.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"AgentRunner: ending gracefully (reason={reason})")
        self._shutdown_event.set()
        for name in self._agents:
            await self._bus.send(BusEndAgentMessage(source=self.name, target=name, reason=reason))

    async def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the runner and all agent tasks.

        Sends targeted `BusCancelAgentMessage` to each agent, then cancels
        the `PipelineRunner`. Idempotent — subsequent calls are ignored.

        Args:
            reason: Optional human-readable reason for cancelling.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"AgentRunner: cancelling (reason={reason})")
        self._shutdown_event.set()
        for name in self._agents:
            await self._bus.send(
                BusCancelAgentMessage(source=self.name, target=name, reason=reason)
            )
        await self._pipecat_runner.cancel()

    async def _handle_bus_message(self, message: BusMessage) -> None:
        """Handle bus messages directed at the runner."""
        if message.source == self.name:
            return
        if isinstance(message, BusEndMessage):
            asyncio.create_task(self.end(message.reason))
        elif isinstance(message, BusCancelMessage):
            asyncio.create_task(self.cancel(message.reason))
        elif isinstance(message, BusAddAgentMessage) and message.agent:
            await self.add_agent(message.agent)

    async def _start_agent_task(self, agent: BaseAgent) -> None:
        """Create an agent's pipeline task and start it as a background asyncio task."""
        pipeline_task = await agent.create_pipeline_task()
        asyncio_task = asyncio.create_task(
            self._pipecat_runner.run(pipeline_task),
            name=f"agent_{agent.name}",
        )
        self._running_agent_tasks[agent.name] = asyncio_task
        asyncio_task.add_done_callback(self._on_agent_task_done)

    def _on_agent_task_done(self, task: asyncio.Task) -> None:
        """Remove a completed agent task from the running tasks dict."""
        name = task.get_name().removeprefix("agent_")
        self._running_agent_tasks.pop(name, None)

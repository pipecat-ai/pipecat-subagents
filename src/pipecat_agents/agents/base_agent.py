#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base agent with bus integration and lifecycle management.

Provides the `BaseAgent` class that all agents inherit from, handling bus
subscription, pipeline creation, deferred start, and agent transfer.
"""

from abc import abstractmethod
from typing import Optional

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.utils.base_object import BaseObject

from pipecat_agents.bus import (
    AgentActivationArgs,
    AgentBus,
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusAgentRegisteredMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusFrameMessage,
    BusMessage,
)


class BaseAgent(BaseObject):
    """Abstract base class for agents in the multi-agent framework.

    Each agent owns a pipeline whose processors are defined by subclasses.
    The agent's pipeline is always running once started, regardless of
    whether the agent is active.

    **Active state**: An agent is *active* when it is currently receiving
    ``BusFrameMessage`` frames from the bus (user audio, text, etc.).
    Only active agents have bus frames queued into their pipeline.
    Non-frame bus messages (activation, end, cancel) are always delivered
    regardless of active state.

    Event handlers:

        on_agent_started(agent): Fired once when the agent's pipeline is
            ready. Use for one-time setup.

        on_agent_activated(agent, args): Fired each time the agent is
            activated via `BusActivateAgentMessage` (or created with
            ``active=True``). Receives the optional `AgentActivationArgs`.

        on_agent_deactivated(agent): Fired when `deactivate_agent()` is called
            and the agent is deactivated.

        on_bus_message(agent, message): Fired for non-frame bus messages after
            default lifecycle handling. Override `on_bus_message()` for custom
            dispatch instead of using this event.

    Example::

        agent = MyAgent(name="my_agent", bus=bus)

        @agent.event_handler("on_agent_activated")
        async def on_activated(agent, args: Optional[AgentActivationArgs]):
            logger.info(f"Agent activated with args: {args}")
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        parent: Optional[str] = None,
        active: bool = False,
    ):
        """Initialize the BaseAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            parent: Optional name of the parent agent. When set, ``end()``
                sends a targeted `BusEndAgentMessage` to the parent instead
                of an untargeted `BusEndMessage`.
            active: Whether the agent starts active (receiving bus frames).
                When True, ``on_agent_activated`` fires after the pipeline
                starts. Defaults to False.
        """
        super().__init__(name=name)
        self._bus = bus
        self._parent = parent
        self._active = active
        self._task: Optional[PipelineTask] = None
        self._pipeline_started = False
        self._pending_activation = active
        self._activation_args: Optional[AgentActivationArgs] = None

        self._register_event_handler("on_agent_started", sync=True)
        self._register_event_handler("on_agent_activated", sync=True)
        self._register_event_handler("on_agent_deactivated", sync=True)
        self._register_event_handler("on_bus_message", sync=True)

        @bus.event_handler("on_message")
        async def on_message(bus, message: BusMessage):
            await self._handle_bus_message(message)

    @property
    def bus(self) -> AgentBus:
        """The bus instance for agent communication."""
        return self._bus

    @property
    def active(self) -> bool:
        """Whether this agent is currently receiving bus frames.

        An active agent has ``BusFrameMessage`` frames queued into its
        pipeline. Non-frame bus messages are delivered regardless.
        """
        return self._active

    @property
    def task(self) -> PipelineTask:
        """The PipelineTask for this agent, created by create_pipeline_task()."""
        if not self._task:
            raise RuntimeError(f"Agent '{self}': task not available.")
        return self._task

    @property
    def activation_args(self) -> Optional[AgentActivationArgs]:
        """The most recent activation arguments, if any."""
        return self._activation_args

    async def end(self, *, reason: Optional[str] = None) -> None:
        """Request a graceful end of the session.

        When a parent is set, sends a targeted `BusEndAgentMessage` to
        the parent, letting it orchestrate shutdown of its sub-agents.
        Without a parent, sends an untargeted `BusEndMessage` handled
        by the runner.

        Args:
            reason: Optional human-readable reason for ending (e.g.
                "customer said goodbye").
        """
        if self._parent:
            await self.send_message(
                BusEndAgentMessage(source=self.name, target=self._parent, reason=reason)
            )
        else:
            await self.send_message(BusEndMessage(source=self.name, reason=reason))

    async def cancel(self) -> None:
        """Broadcast a hard cancel to all agents via the bus."""
        await self.send_message(BusCancelMessage(source=self.name))

    async def add_agent(self, agent: "BaseAgent") -> None:
        """Request the local runner to add a new agent.

        Args:
            agent: The `BaseAgent` instance to register with the runner.
        """
        await self.send_message(BusAddAgentMessage(source=self.name, agent=agent))

    async def deactivate_agent(self) -> None:
        """Deactivate this agent so it stops receiving bus frames."""
        logger.debug(f"Agent '{self}': deactivated")
        self._active = False
        await self._call_event_handler("on_agent_deactivated")

    async def activate_agent(
        self,
        agent_name: str,
        *,
        args: Optional[AgentActivationArgs] = None,
    ) -> None:
        """Activate another agent without stopping this one.

        Unlike ``transfer_to()``, this does not deactivate the current agent.
        Use this from a main agent to start sub-agents (e.g. activate a router
        on client connect).

        Args:
            agent_name: The name of the agent to activate.
            args: Optional `AgentActivationArgs` forwarded to the target agent's
                ``on_agent_activated`` handler.
        """
        await self.send_message(
            BusActivateAgentMessage(source=self.name, target=agent_name, args=args)
        )

    async def transfer_to(
        self,
        agent_name: str,
        *,
        args: Optional[AgentActivationArgs] = None,
    ) -> None:
        """Stop this agent and request transfer to the named agent.

        Args:
            agent_name: The name of the agent to transfer to.
            args: Optional `AgentActivationArgs` forwarded to the target agent's
                ``on_agent_activated`` handler.
        """
        await self.deactivate_agent()
        await self.send_message(
            BusActivateAgentMessage(source=self.name, target=agent_name, args=args)
        )

    async def send_message(self, message: BusMessage) -> None:
        """Send a message to the bus.

        Args:
            message: The `BusMessage` to publish on the agent bus.
        """
        await self._bus.send(message)

    @abstractmethod
    async def build_pipeline_task(self) -> PipelineTask:
        """Build and return a `PipelineTask` for this agent.

        Subclasses implement this to create their pipeline and task with
        whatever params they need. Lifecycle registration is handled
        automatically by `create_pipeline_task()`.

        Returns:
            The created `PipelineTask`.
        """
        pass

    async def create_pipeline_task(self) -> PipelineTask:
        """Create the agent's pipeline task and register lifecycle events.

        Calls `build_pipeline_task()` to get the task from the subclass,
        then wires up pipeline start (sends `BusAgentRegisteredMessage`,
        fires ``on_agent_started``, triggers deferred activation) and
        pipeline finish (cancel propagation on `CancelFrame`).

        Returns:
            The registered `PipelineTask`.
        """
        task = await self.build_pipeline_task()
        self._task = task

        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame: StartFrame):
            logger.debug(f"Agent '{self}': pipeline started")
            self._pipeline_started = True
            await self.send_message(
                BusAgentRegisteredMessage(source=self.name, agent_name=self.name)
            )
            await self._call_event_handler("on_agent_started")
            await self._maybe_activate()

        @task.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(task, frame):
            logger.debug(f"Agent '{self}': pipeline finished ({frame})")
            if isinstance(frame, CancelFrame):
                await self.cancel()

        return task

    async def queue_frame(self, frame) -> None:
        """Queue a frame into this agent's pipeline.

        Args:
            frame: The frame to inject into the pipeline task's queue.
        """
        if self._task:
            await self._task.queue_frame(frame)

    async def queue_frames(self, frames) -> None:
        """Queue multiple frames into this agent's pipeline.

        Args:
            frames: The frames to inject into the pipeline task's queue.
        """
        if self._task:
            await self._task.queue_frames(frames)

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle non-frame bus messages.

        Override to handle custom bus messages. Called for any `BusMessage`
        that is not a `BusFrameMessage` (those are queued as pipeline frames).
        The default implementation handles `BusActivateAgentMessage` (deferred
        start), `BusEndAgentMessage` (graceful pipeline end), and
        `BusCancelMessage` (task cancellation).

        Args:
            message: The `BusMessage` to handle.
        """
        if isinstance(message, BusActivateAgentMessage):
            self._activation_args = message.args
            self._pending_activation = True
            await self._maybe_activate()
        elif isinstance(message, BusEndAgentMessage):
            logger.debug(f"Agent '{self}': received end, ending pipeline")
            if self._task:
                await self._task.queue_frame(EndFrame(reason=message.reason))
        elif isinstance(message, BusCancelAgentMessage):
            logger.debug(f"Agent '{self}': received cancel, cancelling task")
            if self._task:
                await self._task.cancel()

    async def _maybe_activate(self) -> None:
        """Activate the agent and fire on_agent_activated.

        Called when the pipeline is ready and a start has been requested.
        Passes any `AgentActivationArgs` from the activation to the event handler.
        """
        if self._pipeline_started and self._pending_activation:
            logger.debug(f"Agent '{self}': activated")
            self._active = True
            self._pending_activation = False
            await self._call_event_handler("on_agent_activated", self._activation_args)

    async def _handle_bus_message(self, message: BusMessage) -> None:
        """Handle a raw bus message: filter, queue frames, dispatch others."""
        # Ignore targeted messages for other agents
        if message.target and message.target != self.name:
            return

        if isinstance(message, BusFrameMessage):
            # Ignore own frames to prevent infinite loops
            if not self._active or message.source == self.name:
                return
            await self.queue_frame(message.frame)
        else:
            await self.on_bus_message(message)

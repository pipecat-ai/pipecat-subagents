#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base agent for the multi-agent framework.

Provides the `BaseAgent` class that all agents inherit from, handling
agent lifecycle, activation, transfer, and parent-child relationships.
"""

import asyncio
from abc import abstractmethod
from typing import Optional, Tuple, Type

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StopFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
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
    BusMessage,
)
from pipecat_agents.bus.messages import BusFrameMessage
from pipecat_agents.bus.subscriber import BusSubscriber

_LIFECYCLE_FRAMES = (StartFrame, EndFrame, CancelFrame, StopFrame)


class _BusEdgeProcessor(FrameProcessor, BusSubscriber):
    """Pipeline edge that bridges pipeline frames and bus messages.

    Captures pipeline frames traveling in ``direction`` and sends them to the
    bus.  Receives bus frames traveling in the *opposite* direction and pushes
    them into the pipeline.

    Place at pipeline start with ``direction=UPSTREAM`` or at pipeline end with
    ``direction=DOWNSTREAM``.
    """

    def __init__(self, *, bus, agent, direction, exclude_frames=None, **kwargs):
        """Initialize the edge processor.

        Args:
            bus: The ``AgentBus`` used for sending and receiving messages.
            agent: The ``BaseAgent`` that owns this edge processor.
            direction: The ``FrameDirection`` this edge captures. Frames
                traveling in this direction are forwarded to the bus;
                bus frames traveling in the opposite direction are pushed
                into the pipeline.
            exclude_frames: Tuple of frame types to exclude from bus
                forwarding. Defaults to an empty tuple.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._bus = bus
        self._agent = agent
        self._direction = direction
        self._exclude_frames = exclude_frames or ()

    async def setup(self, setup: FrameProcessorSetup):
        """Subscribe to the bus when the processor is set up.

        Args:
            setup: The ``FrameProcessorSetup`` configuration.
        """
        await super().setup(setup)
        await self._bus.subscribe(self)

    async def cleanup(self):
        """Unsubscribe from the bus when the processor is cleaned up."""
        await super().cleanup()
        await self._bus.unsubscribe(self)

    async def process_frame(self, frame, direction):
        """Forward pipeline frames to the bus.

        Passes the frame through the pipeline unchanged. If the frame is
        traveling in the configured direction and is not a lifecycle or
        excluded frame, it is also sent to the bus as a
        ``BusFrameMessage``.

        Args:
            frame: The pipeline frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if direction != self._direction:
            return
        if isinstance(frame, _LIFECYCLE_FRAMES):
            return
        if isinstance(frame, self._exclude_frames):
            return
        await self._bus.send(
            BusFrameMessage(source=self._agent.name, frame=frame, direction=direction)
        )

    async def on_bus_message(self, message):
        """Receive bus frame messages and push them into the pipeline.

        Only processes ``BusFrameMessage`` instances traveling in the
        opposite direction from this edge. Ignores messages from the
        owning agent, messages when the agent is inactive, and messages
        targeted at other agents.

        Args:
            message: The incoming ``BusMessage``.
        """
        if not isinstance(message, BusFrameMessage):
            return
        if message.source == self._agent.name:
            return
        if message.direction == self._direction:
            return
        if not self._agent.active:
            return
        if message.target and message.target != self._agent.name:
            return
        await self.push_frame(message.frame, message.direction)


class BaseAgent(BaseObject, BusSubscriber):
    """Abstract base class for agents in the multi-agent framework.

    Each agent owns a Pipecat pipeline defined by subclasses via
    ``build_pipeline()``. The pipeline runs continuously once started,
    regardless of whether the agent is active.

    An agent is *active* when it is receiving frames from other agents.
    Lifecycle messages (activation, end, cancel) are always delivered
    regardless of active state.

    Overridable lifecycle methods (always call ``super()``):

    - ``on_agent_started()``: Called once when the agent's pipeline is ready.
    - ``on_agent_activated(args)``: Called each time the agent is activated.
    - ``on_agent_deactivated()``: Called when the agent is deactivated.
    - ``on_bus_message(message)``: Called for bus messages after default
      lifecycle handling.

    Event handlers:

    - on_agent_started(agent)
    - on_agent_activated(agent, args)
    - on_agent_deactivated(agent)
    - on_bus_message(agent, message)

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
        active: bool = False,
        enable_bus_sinks: bool = False,
        exclude_frames: Optional[Tuple[Type[Frame], ...]] = None,
    ):
        """Initialize the BaseAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active (receiving frames
                from other agents). When True, ``on_agent_activated``
                fires after the pipeline starts. Defaults to False.
            enable_bus_sinks: Whether to forward pipeline frames to the
                bus and receive frames from the bus. Defaults to False.
            exclude_frames: Frame types to exclude from bus forwarding.
                Lifecycle frames are always excluded. Only used when
                ``enable_bus_sinks`` is True.
        """
        super().__init__(name=name)
        self._bus = bus
        self._parent: Optional[str] = None
        self._active = active
        self._enable_bus_sinks = enable_bus_sinks
        self._task: Optional[PipelineTask] = None
        self._children: list["BaseAgent"] = []
        self._finished: asyncio.Event = asyncio.Event()
        self._pipeline_started = False
        self._pending_activation = active
        self._activation_args: Optional[AgentActivationArgs] = None
        self._exclude_frames = exclude_frames

        self._register_event_handler("on_agent_started")
        self._register_event_handler("on_agent_activated")
        self._register_event_handler("on_agent_deactivated")
        self._register_event_handler("on_bus_message")

    @property
    def bus(self) -> AgentBus:
        """The bus instance for agent communication."""
        return self._bus

    @property
    def parent(self) -> Optional[str]:
        """The name of the parent agent, or None if this is a root agent."""
        return self._parent

    @property
    def children(self) -> list["BaseAgent"]:
        """The list of child agents added via ``add_agent()``."""
        return self._children

    @property
    def active(self) -> bool:
        """Whether this agent is currently receiving frames from other agents."""
        return self._active

    @property
    def task(self) -> PipelineTask:
        """The `PipelineTask` for this agent.

        Raises:
            RuntimeError: If the pipeline task has not been created yet.
        """
        if not self._task:
            raise RuntimeError(f"Agent '{self}': task not available.")
        return self._task

    @property
    def activation_args(self) -> Optional[AgentActivationArgs]:
        """The most recent activation arguments, if any."""
        return self._activation_args

    async def on_agent_started(self) -> None:
        """Called once when the agent's pipeline is ready."""
        pass

    async def on_agent_activated(self, args: Optional[AgentActivationArgs]) -> None:
        """Called each time the agent is activated.

        Override in subclasses to react to activation (e.g. set tools,
        append messages). Always call ``super().on_agent_activated(args)``.

        Args:
            args: Optional activation arguments.
        """
        pass

    async def on_agent_deactivated(self) -> None:
        """Called when the agent is deactivated.

        Override in subclasses for cleanup on deactivation. Always call
        ``super().on_agent_deactivated()``.
        """
        pass

    async def on_bus_message(self, message: BusMessage) -> None:
        """Process an incoming bus message.

        Handles frame delivery, activation, end, and cancel messages.
        Messages targeted at other agents are ignored.

        Args:
            message: The `BusMessage` to process.
        """
        # Frame messages are handled by edge processors
        if isinstance(message, BusFrameMessage):
            return

        # Ignore targeted messages for other agents
        if message.target and message.target != self.name:
            return

        if isinstance(message, BusActivateAgentMessage):
            await self._activate(message)
        elif isinstance(message, BusEndAgentMessage):
            await self._end(message)
        elif isinstance(message, BusCancelAgentMessage):
            await self._cancel(message)

        await self._call_event_handler("on_bus_message", message)

    @abstractmethod
    async def build_pipeline(self) -> Pipeline:
        """Return this agent's pipeline.

        Subclasses implement this to define their pipeline.  When
        ``enable_bus_sinks`` is True, edge-to-bus sink processors are added around
        the returned pipeline.

        Returns:
            A ``Pipeline`` object.

        """
        pass

    def build_pipeline_task(self, pipeline: Pipeline) -> PipelineTask:
        """Create the ``PipelineTask`` for this agent's pipeline.

        Override to customize task parameters (e.g. enable interruptions).
        The default creates a task with default ``PipelineParams``.

        Args:
            pipeline: The fully assembled pipeline (including any
                edge-to-bus wiring).

        Returns:
            A ``PipelineTask``.
        """
        return PipelineTask(
            pipeline,
            enable_rtvi=False,
            idle_timeout_secs=None,
        )

    async def create_pipeline_task(self) -> PipelineTask:
        """Create and configure the agent's pipeline task.

        Calls ``build_pipeline()``, and ``build_pipeline_task()``, then wires
        lifecycle event handling.

        Returns:
            The configured `PipelineTask`.

        """
        await self._bus.subscribe(self)

        user_pipeline = await self.build_pipeline()

        # Wrap with edge-to-bus processors when opted in
        if self._enable_bus_sinks:
            edge_source = _BusEdgeProcessor(
                bus=self._bus,
                agent=self,
                direction=FrameDirection.UPSTREAM,
                exclude_frames=self._exclude_frames,
                name=f"{self.name}::EdgeSource",
            )
            edge_sink = _BusEdgeProcessor(
                bus=self._bus,
                agent=self,
                direction=FrameDirection.DOWNSTREAM,
                exclude_frames=self._exclude_frames,
                name=f"{self.name}::EdgeSink",
            )
            pipeline = Pipeline([edge_source, user_pipeline, edge_sink])
        else:
            pipeline = Pipeline([user_pipeline])

        task = self.build_pipeline_task(pipeline)
        self._task = task

        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame: StartFrame):
            logger.debug(f"Agent '{self}': pipeline started")
            self._pipeline_started = True
            await self.send_message(
                BusAgentRegisteredMessage(source=self.name, agent_name=self.name)
            )
            await self.on_agent_started()
            await self._call_event_handler("on_agent_started")
            await self._maybe_activate()

        @task.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(task, frame):
            logger.debug(f"Agent '{self}': pipeline finished ({frame})")
            if isinstance(frame, CancelFrame):
                await self.cancel()

        return task

    async def end(self, *, reason: Optional[str] = None) -> None:
        """Request a graceful end of the session.

        Args:
            reason: Optional human-readable reason for ending.
        """
        await self.send_message(BusEndMessage(source=self.name, reason=reason))

    async def cancel(self) -> None:
        """Request an immediate cancellation of all agents."""
        await self.send_message(BusCancelMessage(source=self.name))

    async def add_agent(self, agent: "BaseAgent") -> None:
        """Register a child agent under this parent.

        The child's lifecycle (end, cancel) is automatically managed
        by this parent agent.

        Args:
            agent: The child `BaseAgent` instance to add.

        Raises:
            ValueError: If the agent already has a parent.
        """
        if agent._parent is not None:
            raise ValueError(f"Agent '{agent.name}' already has parent '{agent._parent}'")
        agent._parent = self.name
        self._children.append(agent)
        await self.send_message(BusAddAgentMessage(source=self.name, agent=agent))

    async def wait(self) -> None:
        """Wait for this agent's pipeline to finish."""
        await self._finished.wait()

    def notify_finished(self) -> None:
        """Signal that this agent's pipeline has finished."""
        self._finished.set()

    async def activate_agent(
        self,
        agent_name: str,
        *,
        args: Optional[AgentActivationArgs] = None,
    ) -> None:
        """Activate another agent without stopping this one.

        Args:
            agent_name: The name of the agent to activate.
            args: Optional `AgentActivationArgs` forwarded to the target agent's
                ``on_agent_activated`` handler.
        """
        await self.send_message(
            BusActivateAgentMessage(source=self.name, target=agent_name, args=args)
        )

    async def deactivate_agent(self) -> None:
        """Deactivate this agent so it stops receiving frames from other agents."""
        logger.debug(f"Agent '{self}': deactivated")
        self._active = False
        await self.on_agent_deactivated()
        await self._call_event_handler("on_agent_deactivated")

    async def send_message(self, message: BusMessage) -> None:
        """Send a message on the bus.

        Args:
            message: The `BusMessage` to send.
        """
        await self._bus.send(message)

    async def queue_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Queue a frame into this agent's pipeline task.

        Args:
            frame: The frame to queue into the pipeline task.
            direction: Direction the frame should travel. Defaults to
                ``FrameDirection.DOWNSTREAM``.
        """
        if self._task:
            await self._task.queue_frame(frame, direction)

    async def queue_frames(
        self, frames, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Queue multiple frames into this agent's pipeline task.

        Args:
            frames: The frames to queue into the pipeline task.
            direction: Direction the frames should travel. Defaults to
                ``FrameDirection.DOWNSTREAM``.
        """
        if self._task:
            await self._task.queue_frames(frames, direction)

    async def _activate(self, message: BusActivateAgentMessage) -> None:
        """Handle an activation message.

        Stores the activation arguments and marks the agent as pending
        activation, then delegates to ``_maybe_activate()``.

        Args:
            message: The ``BusActivateAgentMessage`` requesting activation.
        """
        self._activation_args = message.args
        self._pending_activation = True
        await self._maybe_activate()

    async def _end(self, message: BusEndAgentMessage) -> None:
        """Propagate end to children, wait for them, then end own pipeline.

        Args:
            message: The ``BusEndAgentMessage`` requesting a graceful end.
        """
        logger.debug(f"Agent '{self}': received end, ending pipeline")
        for child in self._children:
            await self.send_message(
                BusEndAgentMessage(source=self.name, target=child.name, reason=message.reason)
            )
        await asyncio.gather(*(child.wait() for child in self._children))
        if self._task:
            await self._task.queue_frame(EndFrame(reason=message.reason))

    async def _cancel(self, message: BusCancelAgentMessage) -> None:
        """Propagate cancel to children, then cancel own pipeline.

        Args:
            message: The ``BusCancelAgentMessage`` requesting cancellation.
        """
        logger.debug(f"Agent '{self}': received cancel, cancelling task")
        for child in self._children:
            await self.send_message(
                BusCancelAgentMessage(source=self.name, target=child.name, reason=message.reason)
            )
        if self._task:
            await self._task.cancel()

    async def _maybe_activate(self) -> None:
        """Activate the agent, call on_agent_activated, and fire event handlers."""
        if self._pipeline_started and self._pending_activation:
            logger.debug(f"Agent '{self}': activated")
            self._active = True
            self._pending_activation = False
            await self.on_agent_activated(self._activation_args)
            await self._call_event_handler("on_agent_activated", self._activation_args)

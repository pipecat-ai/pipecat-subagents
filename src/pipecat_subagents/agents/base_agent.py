#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base agent for the multi-agent framework.

Provides the `BaseAgent` class that all agents inherit from, handling
agent lifecycle, activation, transfer, parent-child relationships, and
long-running task coordination.
"""

import asyncio
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Type, Union

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
from pydantic import BaseModel

from pipecat_subagents.bus import (
    AgentBus,
    BusActivateAgentMessage,
    BusAddAgentMessage,
    BusAgentRegisteredMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusMessage,
    BusTaskCancelMessage,
    BusTaskRequestMessage,
    BusTaskResponseMessage,
    BusTaskStreamDataMessage,
    BusTaskStreamEndMessage,
    BusTaskStreamStartMessage,
    BusTaskUpdateMessage,
    BusTaskUpdateRequestMessage,
)
from pipecat_subagents.bus.messages import BusFrameMessage
from pipecat_subagents.bus.subscriber import BusSubscriber
from pipecat_subagents.types import TaskStatus

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


@dataclass
class TaskGroup:
    """Tracks a group of task agents launched together.

    Parameters:
        task_id: Shared identifier for all agents in this group.
        agent_names: Names of the agents in the group.
        responses: Collected responses keyed by agent name.
    """

    task_id: str
    agent_names: set[str]
    responses: dict[str, dict] = field(default_factory=dict)
    timeout_task: Optional[asyncio.Task] = None


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
    - ``on_agent_registered(agent_name)``: Called when a child agent's
      pipeline has started and is ready to receive messages.
    - ``on_agent_activated(args)``: Called each time the agent is activated.
    - ``on_agent_deactivated()``: Called when the agent is deactivated.
    - ``on_task_request(task_id, requester, payload)``: Called when a task
      request is received.
    - ``on_task_response(task_id, agent_name, response, status)``: Called
      when a task agent sends a response.
    - ``on_task_update(task_id, agent_name, update)``: Called when a task
      agent sends a progress update.
    - ``on_task_update_requested(task_id)``: Called when the requester asks
      for a progress update.
    - ``on_task_completed(task_id, responses)``: Called when all agents in a
      task group have responded.
    - ``on_task_stream_start(task_id, agent_name, data)``: Called when a task
      agent begins streaming.
    - ``on_task_stream_data(task_id, agent_name, data)``: Called for each
      streaming chunk from a task agent.
    - ``on_task_stream_end(task_id, agent_name, data)``: Called when a task
      agent finishes streaming.
    - ``on_task_cancelled(task_id, reason)``: Called when this agent's task
      is cancelled by the requester.
    - ``on_bus_message(message)``: Called for bus messages after default
      lifecycle handling.

    Event handlers:

    - on_agent_started(agent)
    - on_agent_registered(agent, agent_name)
    - on_agent_activated(agent, args)
    - on_agent_deactivated(agent)
    - on_task_request(agent, task_id, requester, payload)
    - on_task_response(agent, task_id, agent_name, response, status)
    - on_task_update(agent, task_id, agent_name, update)
    - on_task_update_requested(agent, task_id)
    - on_task_completed(agent, task_id, responses)
    - on_task_stream_start(agent, task_id, agent_name, data)
    - on_task_stream_data(agent, task_id, agent_name, data)
    - on_task_stream_end(agent, task_id, agent_name, data)
    - on_task_cancelled(agent, task_id, reason)
    - on_bus_message(agent, message)

    Example::

        agent = MyAgent(name="my_agent", bus=bus)

        @agent.event_handler("on_agent_activated")
        async def on_activated(agent, args: Optional[dict]):
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
        self._activation_args: Optional[dict] = None
        self._exclude_frames = exclude_frames

        # Task state (as worker)
        self._task_id: Optional[str] = None
        self._task_requester: Optional[str] = None
        # Task state (as requester)
        self._task_groups: dict[str, TaskGroup] = {}

        self._register_event_handler("on_agent_started")
        self._register_event_handler("on_agent_registered")
        self._register_event_handler("on_agent_activated")
        self._register_event_handler("on_agent_deactivated")
        self._register_event_handler("on_bus_message")
        self._register_event_handler("on_task_request")
        self._register_event_handler("on_task_response")
        self._register_event_handler("on_task_update")
        self._register_event_handler("on_task_update_requested")
        self._register_event_handler("on_task_completed")
        self._register_event_handler("on_task_stream_start")
        self._register_event_handler("on_task_stream_data")
        self._register_event_handler("on_task_stream_end")
        self._register_event_handler("on_task_cancelled")

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
    def activation_args(self) -> Optional[dict]:
        """The most recent activation arguments, if any."""
        return self._activation_args

    @property
    def task_id(self) -> Optional[str]:
        """The ID of the task this agent is currently working on."""
        return self._task_id

    async def on_agent_started(self) -> None:
        """Called once when the agent's pipeline is ready."""
        pass

    async def on_agent_registered(self, agent_name: str) -> None:
        """Called when a child agent has registered on the bus.

        The child is ready to receive messages.

        Args:
            agent_name: The name of the child agent that registered.
        """
        pass

    async def on_agent_activated(self, args: Optional[dict]) -> None:
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

    async def on_task_request(self, task_id: str, requester: str, payload: Optional[dict]) -> None:
        """Called when this agent receives a task request.

        Override to perform work. Use ``send_task_update()`` to report
        progress and ``send_task_response()`` to return results.

        Args:
            task_id: The unique identifier for this task.
            requester: The name of the agent that launched this task.
            payload: Optional structured data describing the work.
        """
        pass

    async def on_task_response(
        self, task_id: str, agent_name: str, response: Optional[dict], status: TaskStatus
    ) -> None:
        """Called when a task agent sends a response.

        Override to process individual results as they arrive.

        Args:
            task_id: The task identifier.
            agent_name: The name of the agent that responded.
            response: Optional result data from the agent.
            status: Completion status.
        """
        pass

    async def on_task_update(self, task_id: str, agent_name: str, update: Optional[dict]) -> None:
        """Called when a task agent sends a progress update.

        Override to handle intermediate progress reports.

        Args:
            task_id: The task identifier.
            agent_name: The name of the agent that sent the update.
            update: Optional progress data.
        """
        pass

    async def on_task_update_requested(self, task_id: str) -> None:
        """Called when the requester asks for a progress update.

        Override to send back a progress update via ``send_task_update()``.

        Args:
            task_id: The task identifier.
        """
        pass

    async def on_task_completed(self, task_id: str, responses: dict) -> None:
        """Called when all agents in a task group have responded.

        Override to process the collected results from all agents.

        Args:
            task_id: The task identifier.
            responses: Collected responses keyed by agent name.
        """
        pass

    async def on_task_stream_start(
        self, task_id: str, agent_name: str, data: Optional[dict]
    ) -> None:
        """Called when a task agent begins streaming.

        Args:
            task_id: The task identifier.
            agent_name: The name of the streaming agent.
            data: Optional metadata about the stream.
        """
        pass

    async def on_task_stream_data(
        self, task_id: str, agent_name: str, data: Optional[dict]
    ) -> None:
        """Called for each streaming chunk from a task agent.

        Args:
            task_id: The task identifier.
            agent_name: The name of the streaming agent.
            data: The chunk payload.
        """
        pass

    async def on_task_stream_end(self, task_id: str, agent_name: str, data: Optional[dict]) -> None:
        """Called when a task agent finishes streaming.

        Args:
            task_id: The task identifier.
            agent_name: The name of the streaming agent.
            data: Optional final metadata.
        """
        pass

    async def on_task_cancelled(self, task_id: str, reason: Optional[str]) -> None:
        """Called when this agent's task is cancelled by the requester.

        Override to clean up resources or stop in-progress work.

        Args:
            task_id: The task identifier.
            reason: Optional human-readable reason for cancellation.
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

        if isinstance(message, BusAgentRegisteredMessage):
            await self._handle_agent_registered(message)
        elif isinstance(message, BusActivateAgentMessage):
            await self._handle_agent_activate(message)
        elif isinstance(message, BusEndAgentMessage):
            await self._handle_agent_end(message)
        elif isinstance(message, BusCancelAgentMessage):
            await self._handle_agent_cancel(message)
        elif isinstance(message, BusTaskRequestMessage):
            await self._handle_task_request(message)
        elif isinstance(message, BusTaskResponseMessage):
            await self._handle_task_response(message)
        elif isinstance(message, BusTaskUpdateMessage):
            await self._handle_task_update(message)
        elif isinstance(message, BusTaskUpdateRequestMessage):
            await self._handle_task_update_request(message)
        elif isinstance(message, BusTaskCancelMessage):
            await self._handle_task_cancel(message)
        elif isinstance(message, BusTaskStreamStartMessage):
            await self._handle_task_stream_start(message)
        elif isinstance(message, BusTaskStreamDataMessage):
            await self._handle_task_stream_data(message)
        elif isinstance(message, BusTaskStreamEndMessage):
            await self._handle_task_stream_end(message)

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
        args: Union[BaseModel, dict, None] = None,
    ) -> None:
        """Activate another agent without stopping this one.

        Args:
            agent_name: The name of the agent to activate.
            args: Optional activation arguments forwarded to the target
                agent's ``on_agent_activated`` handler. Accepts a
                ``BaseModel`` (e.g. ``LLMActivationArgs``), a plain
                dict, or None.
        """
        if isinstance(args, BaseModel):
            args = args.model_dump(exclude_none=True)
        await self.send_message(
            BusActivateAgentMessage(source=self.name, target=agent_name, args=args)
        )

    async def deactivate_agent(self) -> None:
        """Deactivate this agent so it stops receiving frames from other agents."""
        logger.debug(f"Agent '{self}': deactivated")
        self._active = False
        await self.on_agent_deactivated()
        await self._call_event_handler("on_agent_deactivated")

    async def start_task(
        self,
        *agents: "BaseAgent",
        args: Optional[dict] = None,
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Launch one or more task agents with a shared task_id.

        Adds agents (via ``add_agent`` + ``BusAddAgentMessage``), activates
        them, and sends ``BusTaskRequestMessage`` to each. Returns the
        task_id.

        Args:
            *agents: One or more agent instances to launch as task workers.
            args: Optional activation arguments forwarded to each agent's
                ``on_agent_activated``.
            payload: Optional structured data describing the work, forwarded
                via ``BusTaskRequestMessage``.
            timeout: Optional timeout in seconds. If set, the task is
                automatically cancelled after this duration.

        Returns:
            The generated task_id shared by all agents in the group.
        """
        task_id = str(uuid.uuid4())
        group = TaskGroup(task_id=task_id, agent_names={a.name for a in agents})
        self._task_groups[task_id] = group

        for agent in agents:
            await self.add_agent(agent)
            await self.activate_agent(agent.name, args=args)
            await self.send_message(
                BusTaskRequestMessage(
                    source=self.name, target=agent.name, task_id=task_id, payload=payload
                )
            )

        if timeout is not None:
            group.timeout_task = asyncio.create_task(self._task_timeout(task_id, timeout))

        return task_id

    async def _task_timeout(self, task_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        await self.cancel_task(task_id, reason="timeout")

    async def cancel_task(self, task_id: str, *, reason: Optional[str] = None) -> None:
        """Cancel a running task group.

        Args:
            task_id: The task identifier to cancel.
            reason: Optional human-readable reason for cancellation.
        """
        group = self._task_groups.pop(task_id, None)
        if group:
            if group.timeout_task:
                group.timeout_task.cancel()
            for agent_name in group.agent_names:
                await self.send_message(
                    BusTaskCancelMessage(
                        source=self.name, target=agent_name, task_id=task_id, reason=reason
                    )
                )

    async def request_task_update(self, task_id: str, agent_name: str) -> None:
        """Request a progress update from a task agent.

        Args:
            task_id: The task identifier.
            agent_name: The name of the agent to request an update from.
        """
        await self.send_message(
            BusTaskUpdateRequestMessage(source=self.name, target=agent_name, task_id=task_id)
        )

    async def send_task_response(
        self, response: Optional[dict] = None, *, status: TaskStatus = TaskStatus.COMPLETED
    ) -> None:
        """Send a task response back to the requester (called by the task agent).

        Clears the agent's task state (``_task_id`` / ``_task_requester``)
        after sending so that the agent is ready for a new task.

        Args:
            response: Optional result data.
            status: Completion status. Defaults to ``TaskStatus.COMPLETED``.

        Raises:
            RuntimeError: If this agent has no active task.
        """
        if not self._task_id or not self._task_requester:
            raise RuntimeError(f"Agent '{self}': no active task to respond to")
        await self.send_message(
            BusTaskResponseMessage(
                source=self.name,
                target=self._task_requester,
                task_id=self._task_id,
                response=response,
                status=status,
            )
        )
        self._task_id = None
        self._task_requester = None

    async def send_task_update(self, update: Optional[dict] = None) -> None:
        """Send a progress update to the requester (called by the task agent).

        Args:
            update: Optional progress data.

        Raises:
            RuntimeError: If this agent has no active task.
        """
        if not self._task_id or not self._task_requester:
            raise RuntimeError(f"Agent '{self}': no active task to update")
        await self.send_message(
            BusTaskUpdateMessage(
                source=self.name,
                target=self._task_requester,
                task_id=self._task_id,
                update=update,
            )
        )

    async def send_task_stream_start(self, data: Optional[dict] = None) -> None:
        """Begin streaming task results back to the requester.

        Args:
            data: Optional metadata about the stream.

        Raises:
            RuntimeError: If this agent has no active task.
        """
        if not self._task_id or not self._task_requester:
            raise RuntimeError(f"Agent '{self}': no active task to stream")
        await self.send_message(
            BusTaskStreamStartMessage(
                source=self.name,
                target=self._task_requester,
                task_id=self._task_id,
                data=data,
            )
        )

    async def send_task_stream_data(self, data: Optional[dict] = None) -> None:
        """Send a streaming chunk to the requester.

        Args:
            data: The chunk payload.

        Raises:
            RuntimeError: If this agent has no active task.
        """
        if not self._task_id or not self._task_requester:
            raise RuntimeError(f"Agent '{self}': no active task to stream")
        await self.send_message(
            BusTaskStreamDataMessage(
                source=self.name,
                target=self._task_requester,
                task_id=self._task_id,
                data=data,
            )
        )

    async def send_task_stream_end(self, data: Optional[dict] = None) -> None:
        """End the stream. Triggers group completion on the requester.

        Args:
            data: Optional final metadata.

        Raises:
            RuntimeError: If this agent has no active task.
        """
        if not self._task_id or not self._task_requester:
            raise RuntimeError(f"Agent '{self}': no active task to stream")
        await self.send_message(
            BusTaskStreamEndMessage(
                source=self.name,
                target=self._task_requester,
                task_id=self._task_id,
                data=data,
            )
        )

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

    async def _handle_agent_registered(self, message: BusAgentRegisteredMessage) -> None:
        """Notify this agent that a child has started.

        Only fires for agents that are direct children of this agent.
        """
        child_names = {c.name for c in self._children}
        if message.agent_name in child_names:
            await self.on_agent_registered(message.agent_name)
            await self._call_event_handler("on_agent_registered", message.agent_name)

    async def _handle_agent_activate(self, message: BusActivateAgentMessage) -> None:
        """Handle an activation message.

        Stores the activation arguments and marks the agent as pending
        activation, then delegates to ``_maybe_activate()``.

        Args:
            message: The ``BusActivateAgentMessage`` requesting activation.
        """
        self._activation_args = message.args
        self._pending_activation = True
        await self._maybe_activate()

    async def _handle_agent_end(self, message: BusEndAgentMessage) -> None:
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

    async def _handle_agent_cancel(self, message: BusCancelAgentMessage) -> None:
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

    async def _handle_task_request(self, message: BusTaskRequestMessage) -> None:
        """Handle an incoming task request."""
        self._task_id = message.task_id
        self._task_requester = message.source
        await self.on_task_request(message.task_id, message.source, message.payload)
        await self._call_event_handler(
            "on_task_request", message.task_id, message.source, message.payload
        )

    async def _handle_task_response(self, message: BusTaskResponseMessage) -> None:
        """Handle a task response and track group completion."""
        await self.on_task_response(
            message.task_id, message.source, message.response, message.status
        )
        await self._call_event_handler(
            "on_task_response",
            message.task_id,
            message.source,
            message.response,
            message.status,
        )
        await self._track_task_group_response(message.task_id, message.source, message.response)

    async def _handle_task_update(self, message: BusTaskUpdateMessage) -> None:
        """Handle a task progress update."""
        await self.on_task_update(message.task_id, message.source, message.update)
        await self._call_event_handler(
            "on_task_update", message.task_id, message.source, message.update
        )

    async def _handle_task_update_request(self, message: BusTaskUpdateRequestMessage) -> None:
        """Handle a task update request from the requester."""
        if self._task_id == message.task_id:
            await self.on_task_update_requested(message.task_id)
            await self._call_event_handler("on_task_update_requested", message.task_id)

    async def _handle_task_cancel(self, message: BusTaskCancelMessage) -> None:
        """Handle a task cancellation.

        Calls the ``on_task_cancelled`` hook for cleanup, then
        automatically sends a cancelled response back to the requester.
        The requester receives ``on_task_response`` with
        ``status="cancelled"`` — same path as completed or failed tasks.
        """
        if self._task_id == message.task_id:
            await self.on_task_cancelled(message.task_id, message.reason)
            await self._call_event_handler("on_task_cancelled", message.task_id, message.reason)
            await self.send_task_response(status=TaskStatus.CANCELLED)

    async def _handle_task_stream_start(self, message: BusTaskStreamStartMessage) -> None:
        """Handle the start of a streaming task response."""
        await self.on_task_stream_start(message.task_id, message.source, message.data)
        await self._call_event_handler(
            "on_task_stream_start", message.task_id, message.source, message.data
        )

    async def _handle_task_stream_data(self, message: BusTaskStreamDataMessage) -> None:
        """Handle a streaming task data chunk."""
        await self.on_task_stream_data(message.task_id, message.source, message.data)
        await self._call_event_handler(
            "on_task_stream_data", message.task_id, message.source, message.data
        )

    async def _handle_task_stream_end(self, message: BusTaskStreamEndMessage) -> None:
        """Handle the end of a streaming task response."""
        await self.on_task_stream_end(message.task_id, message.source, message.data)
        await self._call_event_handler(
            "on_task_stream_end", message.task_id, message.source, message.data
        )
        await self._track_task_group_response(message.task_id, message.source, message.data)

    async def _track_task_group_response(
        self, task_id: str, source: str, response: Optional[dict]
    ) -> None:
        """Record a task agent's response and fire completion when all have responded."""
        group = self._task_groups.get(task_id)
        if group:
            group.responses[source] = response or {}
            if group.responses.keys() >= group.agent_names:
                if group.timeout_task:
                    group.timeout_task.cancel()
                del self._task_groups[task_id]
                await self.on_task_completed(task_id, group.responses)
                await self._call_event_handler("on_task_completed", task_id, group.responses)

    async def _maybe_activate(self) -> None:
        """Activate the agent, call on_agent_activated, and fire event handlers."""
        if self._pipeline_started and self._pending_activation:
            logger.debug(f"Agent '{self}': activated")
            self._active = True
            self._pending_activation = False
            await self.on_agent_activated(self._activation_args)
            await self._call_event_handler("on_agent_activated", self._activation_args)

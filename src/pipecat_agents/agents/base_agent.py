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
from typing import List, Optional

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import CANCEL_TIMEOUT_SECS, PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserverParams, RTVIProcessor
from pipecat.utils.base_object import BaseObject

from pipecat_agents.bus import (
    AgentActivatedArgs,
    AgentBus,
    BusAddAgentMessage,
    BusAgentRegisteredMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusFrameMessage,
    BusMessage,
    BusOutputProcessor,
    BusStartAgentMessage,
)


class BaseAgent(BaseObject):
    """Abstract base class for agents in the multi-agent framework.

    Each agent owns a pipeline whose processors are defined by subclasses.
    Bus messages are received by subscribing to the bus `on_message` event;
    `BusFrameMessage` frames are queued directly to the agent's pipeline task,
    while other messages are dispatched to `on_bus_message()`.

    Event handlers:

        on_agent_started(agent): Fired once when the agent's pipeline is
            ready. Use for one-time setup.

        on_agent_activated(agent, args): Fired each time the agent is
            activated via `BusStartAgentMessage` (or created with
            ``active=True``). Receives the optional `AgentActivatedArgs`.

        on_agent_deactivated(agent): Fired when `stop_agent()` is called
            and the agent is deactivated.

        on_bus_message(agent, message): Fired for non-frame bus messages after
            default lifecycle handling. Override `on_bus_message()` for custom
            dispatch instead of using this event.

    Example::

        agent = MyAgent(name="my_agent", bus=bus)

        @agent.event_handler("on_agent_activated")
        async def on_activated(agent, args: Optional[AgentActivatedArgs]):
            logger.info(f"Agent activated with args: {args}")
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        active: bool = False,
        enable_bus_output: bool = True,
        enable_rtvi: bool = False,
        rtvi_processor: Optional[RTVIProcessor] = None,
        rtvi_observer_params: Optional[RTVIObserverParams] = None,
        pipeline_params: Optional[PipelineParams] = None,
        observers: Optional[List[BaseObserver]] = None,
        cancel_on_idle_timeout: bool = False,
        cancel_timeout_secs: float = CANCEL_TIMEOUT_SECS,
    ):
        """Initialize the BaseAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            active: Whether the agent starts active. Defaults to False.
            enable_bus_output: Whether to append a `BusOutputProcessor` to the
                pipeline. Defaults to True.
            enable_rtvi: Whether to enable RTVI on the pipeline task.
                Defaults to False.
            rtvi_processor: Optional `RTVIProcessor` for the pipeline task.
            rtvi_observer_params: Optional `RTVIObserverParams` for the
                pipeline task.
            pipeline_params: Optional `PipelineParams` for this agent's task.
            observers: Optional list of `BaseObserver` instances for monitoring.
            cancel_on_idle_timeout: Whether to cancel the pipeline task when
                idle timeout is reached. Defaults to False.
            cancel_timeout_secs: Seconds to wait for graceful cancellation
                before forcing. Defaults to `CANCEL_TIMEOUT_SECS`.
        """
        super().__init__(name=name)
        self._bus = bus
        self._active = active
        self._enable_bus_output = enable_bus_output
        self._enable_rtvi = enable_rtvi
        self._rtvi_processor = rtvi_processor
        self._rtvi_observer_params = rtvi_observer_params
        self._pipeline_params = pipeline_params or PipelineParams()
        self._observers = observers
        self._cancel_on_idle_timeout = cancel_on_idle_timeout
        self._cancel_timeout_secs = cancel_timeout_secs
        self._task: Optional[PipelineTask] = None
        self._pipeline_started = False
        self._pending_activation = False
        self._pending_activation_args: Optional[AgentActivatedArgs] = None

        self._register_event_handler("on_agent_started", sync=True)
        self._register_event_handler("on_agent_activated", sync=True)
        self._register_event_handler("on_agent_deactivated", sync=True)
        self._register_event_handler("on_bus_message", sync=True)

        @bus.event_handler("on_message")
        async def on_message(bus, message: BusMessage):
            await self._handle_bus_message(message)

    @property
    def active(self) -> bool:
        """Whether this agent is active and processing frames."""
        return self._active

    @property
    def task(self) -> PipelineTask:
        """The PipelineTask for this agent, created by create_pipeline_task()."""
        if not self._task:
            raise RuntimeError(f"Agent '{self}': task not available.")
        return self._task

    async def send_message(self, message: BusMessage) -> None:
        """Send a message to the bus.

        Args:
            message: The `BusMessage` to publish on the agent bus.
        """
        await self._bus.send(message)

    async def add_agent(self, agent: "BaseAgent") -> None:
        """Request the local runner to add a new agent.

        Args:
            agent: The `BaseAgent` instance to register with the runner.
        """
        await self.send_message(BusAddAgentMessage(source=self.name, agent=agent))

    async def stop_agent(self) -> None:
        """Deactivate this agent."""
        logger.debug(f"Agent '{self}': deactivated")
        self._active = False
        await self._call_event_handler("on_agent_deactivated")

    async def cancel(self) -> None:
        """Broadcast a hard cancel to all agents via the bus."""
        await self.send_message(BusCancelMessage(source=self.name))

    async def end(self, *, reason: Optional[str] = None) -> None:
        """Request a graceful end of the entire session.

        Sends a `BusEndMessage` to the bus. The runner handles the
        shutdown sequence.

        Args:
            reason: Optional human-readable reason for ending (e.g.
                "customer said goodbye").
        """
        await self.send_message(BusEndMessage(source=self.name, reason=reason))

    async def transfer_to(
        self,
        agent_name: str,
        *,
        args: Optional[AgentActivatedArgs] = None,
    ) -> None:
        """Stop this agent and request transfer to the named agent.

        Args:
            agent_name: The name of the agent to transfer to.
            args: Optional `AgentActivatedArgs` forwarded to the target agent's
                ``on_agent_activated`` handler.
        """
        await self.stop_agent()
        await self.send_message(
            BusStartAgentMessage(source=self.name, target=agent_name, args=args)
        )

    @abstractmethod
    def build_pipeline_processors(self) -> List[FrameProcessor]:
        """Return the list of FrameProcessors for this agent's pipeline.

        Do NOT include the `BusOutputProcessor` — it is appended
        automatically when `enable_bus_output` is True.

        Returns:
            Ordered list of processors for the pipeline.
        """
        pass

    async def create_pipeline_task(self) -> PipelineTask:
        """Build the agent's pipeline and create a `PipelineTask`.

        Appends a `BusOutputProcessor` when `enable_bus_output` is True.
        When the pipeline starts, sends a `BusAgentRegisteredMessage` to
        announce this agent is available on the bus.

        Returns:
            The created `PipelineTask`.
        """
        bus_output = BusOutputProcessor(
            bus=self._bus,
            agent_name=self.name,
            name=f"{self.name}::BusOutput",
        )

        processors = self.build_pipeline_processors()
        if self._enable_bus_output:
            processors.append(bus_output)
        pipeline = Pipeline(processors)

        self._task = PipelineTask(
            pipeline,
            params=self._pipeline_params,
            enable_rtvi=self._enable_rtvi,
            rtvi_processor=self._rtvi_processor,
            rtvi_observer_params=self._rtvi_observer_params,
            observers=self._observers,
            cancel_on_idle_timeout=self._cancel_on_idle_timeout,
            cancel_timeout_secs=self._cancel_timeout_secs,
        )

        @self._task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame: StartFrame):
            self._pipeline_started = True
            await self.send_message(
                BusAgentRegisteredMessage(source=self.name, agent_name=self.name)
            )
            await self._call_event_handler("on_agent_started")
            await self._maybe_activate()

        @self._task.event_handler("on_pipeline_finished")
        async def on_pipeline_canceled(task, frame):
            if isinstance(frame, CancelFrame):
                await self.cancel()

        return self._task

    async def queue_frame(self, frame) -> None:
        """Queue a frame into this agent's pipeline.

        Args:
            frame: The frame to inject into the pipeline task's queue.
        """
        if self._task:
            await self._task.queue_frame(frame)

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle non-frame bus messages.

        Override to handle custom bus messages. Called for any `BusMessage`
        that is not a `BusFrameMessage` (those are queued as pipeline frames).
        The default implementation handles `BusStartAgentMessage` (deferred
        start), `BusEndAgentMessage` (graceful pipeline end), and
        `BusCancelMessage` (task cancellation).

        Args:
            message: The `BusMessage` to handle.
        """
        if isinstance(message, BusStartAgentMessage):
            self._pending_activation_args = message.args
            self._pending_activation = True
            await self._maybe_activate()
        elif isinstance(message, BusEndAgentMessage):
            logger.debug(f"Agent '{self}': received end, ending pipeline")
            if self._task:
                await self._task.queue_frame(EndFrame(reason=message.reason))
        elif isinstance(message, BusCancelMessage):
            logger.debug(f"Agent '{self}': received cancel, cancelling task")
            if self._task:
                await self._task.cancel()

    async def _maybe_activate(self) -> None:
        """Activate the agent and fire on_agent_activated.

        Called when the pipeline is ready and a start has been requested.
        Passes any `AgentActivatedArgs` from the activation to the event handler.
        """
        if self._pipeline_started and self._pending_activation:
            logger.debug(f"Agent '{self}': activated")
            self._active = True
            self._pending_activation = False
            args = self._pending_activation_args
            self._pending_activation_args = None
            await self._call_event_handler("on_agent_activated", args)

    async def _handle_bus_message(self, message: BusMessage) -> None:
        """Handle a raw bus message: filter, queue frames, dispatch others."""
        # Ignore own messages
        if message.source == self.name:
            return
        # Ignore targeted messages for other agents
        if message.target and message.target != self.name:
            return

        if isinstance(message, BusFrameMessage):
            if self._active:
                await self.queue_frame(message.frame)
        else:
            await self.on_bus_message(message)

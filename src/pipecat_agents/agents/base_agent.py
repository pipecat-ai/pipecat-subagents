#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base agent with bus integration and lifecycle management.

Provides the `BaseAgent` class that all agents inherit from, handling bus
subscription, pipeline creation, deferred start, and agent transfer.
"""

import asyncio
from abc import abstractmethod
from typing import Any, Callable, List, Optional

from loguru import logger
from pipecat.frames.frames import CancelFrame, FunctionCallResultProperties, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import CANCEL_TIMEOUT_SECS, PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserverParams, RTVIProcessor
from pipecat.utils.base_object import BaseObject

FunctionCallResultCallback = Callable[..., Any]

from pipecat_agents.bus import (
    AgentBus,
    BusAddAgentMessage,
    BusAgentRegisteredMessage,
    BusCancelMessage,
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
    """

    def __init__(
        self,
        name: str,
        *,
        bus: AgentBus,
        enabled: bool = False,
        enable_bus_output: bool = True,
        enable_rtvi: bool = False,
        rtvi_processor: Optional[RTVIProcessor] = None,
        rtvi_observer_params: Optional[RTVIObserverParams] = None,
        pipeline_params: Optional[PipelineParams] = None,
        cancel_on_idle_timeout: bool = False,
        cancel_timeout_secs: float = CANCEL_TIMEOUT_SECS,
    ):
        """Initialize the BaseAgent.

        Args:
            name: Unique name for this agent.
            bus: The `AgentBus` for inter-agent communication.
            enabled: Whether the agent starts enabled. Defaults to False.
            enable_bus_output: Whether to append a `BusOutputProcessor` to the
                pipeline. Defaults to True.
            enable_rtvi: Whether to enable RTVI on the pipeline task.
                Defaults to False.
            rtvi_processor: Optional `RTVIProcessor` for the pipeline task.
            rtvi_observer_params: Optional `RTVIObserverParams` for the
                pipeline task.
            pipeline_params: Optional `PipelineParams` for this agent's task.
        """
        super().__init__(name=name)
        self._bus = bus
        self._enabled = enabled
        self._enable_bus_output = enable_bus_output
        self._enable_rtvi = enable_rtvi
        self._rtvi_processor = rtvi_processor
        self._rtvi_observer_params = rtvi_observer_params
        self._pipeline_params = pipeline_params or PipelineParams()
        self._cancel_on_idle_timeout = cancel_on_idle_timeout
        self._cancel_timeout_secs = cancel_timeout_secs
        self._task: Optional[PipelineTask] = None
        self._pipeline_started = False
        self._pending_start = False

        self._register_event_handler("on_agent_started", sync=True)
        self._register_event_handler("on_agent_stopped", sync=True)
        self._register_event_handler("on_bus_message", sync=True)

        @bus.event_handler("on_message")
        async def on_message(bus, message: BusMessage):
            await self._handle_bus_message(message)

    @property
    def enabled(self) -> bool:
        """Whether this agent is enabled and processing frames."""
        return self._enabled

    @property
    def task(self) -> PipelineTask:
        """The PipelineTask for this agent, created by create_pipeline_task()."""
        if not self._task:
            raise RuntimeError(f"Agent '{self}': task not available.")
        return self._task

    async def send_message(self, message: BusMessage) -> None:
        """Send a message to the bus."""
        await self._bus.send(message)

    async def add_agent(self, agent: "BaseAgent") -> None:
        """Request the local runner to add a new agent."""
        await self.send_message(BusAddAgentMessage(source=self.name, agent=agent))

    async def stop_agent(self) -> None:
        """Disable this agent."""
        logger.debug(f"Agent '{self}': stopped")
        self._enabled = False
        await self._call_event_handler("on_agent_stopped")

    async def cancel(self) -> None:
        """Broadcast a hard cancel to all agents via the bus."""
        await self.send_message(BusCancelMessage(source=self.name))

    async def end(self) -> None:
        """Request a graceful end of the entire session."""
        await self.send_message(BusEndMessage(source=self.name))

    async def transfer_to(
        self,
        agent_name: str,
        *,
        result_callback: Optional[FunctionCallResultCallback] = None,
    ) -> None:
        """Stop this agent and request transfer to the named agent.

        When called from a function handler, pass ``params.result_callback``
        so the transfer waits for the function-call result to be committed
        to the context before activating the target agent.

        Args:
            agent_name: The name of the agent to transfer to.
            result_callback: The ``result_callback`` from FunctionCallParams.
                When provided, the transfer is sent after the function result
                is added to the context (via ``on_context_updated``).
        """
        if result_callback:
            context_updated = asyncio.Event()

            async def _on_context_updated():
                context_updated.set()

            await result_callback(
                "Help the user with their latest request.",
                properties=FunctionCallResultProperties(
                    run_llm=False,
                    on_context_updated=_on_context_updated,
                ),
            )
            await context_updated.wait()

        await self.stop_agent()
        await self.send_message(BusStartAgentMessage(source=self.name, target=agent_name))

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
            cancel_on_idle_timeout=self._cancel_on_idle_timeout,
            cancel_timeout_secs=self._cancel_timeout_secs,
        )

        @self._task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame: StartFrame):
            self._pipeline_started = True
            await self.send_message(
                BusAgentRegisteredMessage(source=self.name, agent_name=self.name)
            )
            await self._maybe_start_agent()

        @self._task.event_handler("on_pipeline_finished")
        async def on_pipeline_canceled(task, frame):
            if isinstance(frame, CancelFrame):
                await self.cancel()

        return self._task

    async def queue_frame(self, frame) -> None:
        """Queue a frame into this agent's pipeline."""
        if self._task:
            await self._task.queue_frame(frame)

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle non-frame bus messages.

        Override to handle custom bus messages. Called for any BusMessage
        that is not a BusFrameMessage (those are queued as pipeline frames).

        Default implementation handles lifecycle messages.
        """
        if isinstance(message, BusStartAgentMessage):
            self._pending_start = True
            await self._maybe_start_agent()
        elif isinstance(message, BusCancelMessage):
            logger.debug(f"Agent '{self}': received cancel, cancelling task")
            if self._task:
                await self._task.cancel()

    async def _maybe_start_agent(self) -> None:
        """Enable the agent and fire on_agent_started.

        Called when the pipeline is ready and a start has been requested.
        """
        if self._pipeline_started and self._pending_start:
            logger.debug(f"Agent '{self}': started")
            self._enabled = True
            self._pending_start = False
            await self._call_event_handler("on_agent_started")

    async def _handle_bus_message(self, message: BusMessage) -> None:
        """Handle a raw bus message: filter, queue frames, dispatch others."""
        # Ignore own messages
        if message.source == self.name:
            return
        # Ignore targeted messages for other agents
        if message.target and message.target != self.name:
            return

        if isinstance(message, BusFrameMessage):
            if self._enabled:
                await self.queue_frame(message.frame)
        else:
            await self.on_bus_message(message)

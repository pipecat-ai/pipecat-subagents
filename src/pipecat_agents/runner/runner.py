#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent runner for orchestrating multi-agent lifecycle and pipeline tasks."""

import asyncio
from typing import Optional

from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import BaseTransport
from pipecat.utils.base_object import BaseObject
from pydantic import BaseModel, ConfigDict

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import (
    AgentBus,
    BusAddAgentMessage,
    BusAssistantTurnStartedMessage,
    BusAssistantTurnStoppedMessage,
    BusCancelMessage,
    BusClientConnectedMessage,
    BusClientDisconnectedMessage,
    BusEndMessage,
    BusMessage,
    BusStartAgentMessage,
    BusUserTurnStartedMessage,
    BusUserTurnStoppedMessage,
)
from pipecat_agents.runner.user_agent import UserAgent


class UserAgentParams(BaseModel):
    """Configuration for the user agent (transport bridge).

    Args:
        transport: The transport (WebRTC, WebSocket, etc.) to bridge.
        context: Optional shared LLMContext for turn detection.
        vad_analyzer: Optional VAD analyzer for speech detection.
        stt: Optional STT service.
        tts: Optional TTS service.
        pipeline_params: Optional PipelineParams for the pipeline task.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: BaseTransport
    context: Optional[LLMContext] = None
    vad_analyzer: Optional[VADAnalyzer] = None
    stt: Optional[FrameProcessor] = None
    tts: Optional[FrameProcessor] = None
    pipeline_params: Optional[PipelineParams] = None


class AgentRunner(BaseObject):
    """Lifecycle orchestrator for multi-agent systems.

    Manages agent lifecycle and coordinates pipeline tasks via
    `PipelineRunner`. The user agent (transport bridge) is created
    internally from `UserAgentParams`; other agents are added via
    `add_agent()`.
    """

    def __init__(
        self,
        *,
        user_agent_params: Optional[UserAgentParams] = None,
        bus: Optional[AgentBus] = None,
        handle_sigint: bool = True,
    ):
        """Initialize the AgentRunner.

        Args:
            user_agent_params: Optional user agent configuration. When
                provided, creates a user agent internally to bridge the
                transport to the bus.
            bus: Optional `AgentBus` instance. Creates a default one if None.
            handle_sigint: Whether `PipelineRunner` handles SIGINT.
                Defaults to True.
        """
        super().__init__()
        self._bus = bus or AgentBus()

        self._running: bool = False
        self._agents: dict[str, BaseAgent] = {}
        self._running_agent_tasks: dict[str, asyncio.Task] = {}
        self._pipecat_runner = PipelineRunner(handle_sigint=handle_sigint)
        self._shutdown_event = asyncio.Event()

        self._user_agent: Optional[UserAgent] = None
        if user_agent_params:
            self._user_agent = UserAgent(
                bus=self._bus,
                transport=user_agent_params.transport,
                context=user_agent_params.context,
                vad_analyzer=user_agent_params.vad_analyzer,
                stt=user_agent_params.stt,
                tts=user_agent_params.tts,
                pipeline_params=user_agent_params.pipeline_params,
            )
            self._agents[self._user_agent.name] = self._user_agent

        self._register_event_handler("on_runner_started")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_user_turn_started")
        self._register_event_handler("on_user_turn_stopped")
        self._register_event_handler("on_assistant_turn_started")
        self._register_event_handler("on_assistant_turn_stopped")

        @self._bus.event_handler("on_message")
        async def on_message(bus, message: BusMessage):
            await self._handle_bus_message(message)

    @property
    def bus(self) -> AgentBus:
        """The bus instance for agent communication."""
        return self._bus

    @property
    def context_aggregator(self) -> LLMContextAggregatorPair | None:
        """The user agent's context aggregator pair, or None if no context."""
        if self._user_agent:
            return self._user_agent.context_aggregator
        return None

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

    async def activate_agent(self, name: str) -> None:
        """Send a BusStartAgentMessage to the named agent."""
        await self._bus.send(BusStartAgentMessage(source="", target=name))

    async def run(self) -> None:
        """Start all agents, block until cancelled.

        Starts all registered agents, fires on_runner_started, then blocks
        until cancel() is called. New agents can be added dynamically via
        add_agent() after run() has started.
        """
        self._running = True
        self._shutdown_event.clear()

        for agent in self._agents.values():
            await self._start_agent_task(agent)

        await self._call_event_handler("on_runner_started")

        await self._shutdown_event.wait()

        # Wait for remaining agent tasks to finish cleanup
        remaining = [t for t in self._running_agent_tasks.values() if not t.done()]
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)
        self._running = False

    async def end(self, reason: Optional[str] = None) -> None:
        """Gracefully end all agent pipelines and shut down.

        Queues an `EndFrame` to every running agent task so each pipeline
        flushes in-flight work (TTS, audio buffers) before stopping.

        Args:
            reason: Optional human-readable reason for ending.
        """
        logger.info(f"AgentRunner: ending gracefully (reason={reason})")
        for agent in self._agents.values():
            if agent.task:
                await agent.task.queue_frame(EndFrame())
        self._shutdown_event.set()

    async def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the runner and all agent tasks."""
        await self._bus.send(BusCancelMessage(source="", reason=reason))
        await self._pipecat_runner.cancel()
        self._shutdown_event.set()

    async def _handle_bus_message(self, message: BusMessage) -> None:
        """Handle bus messages directed at the runner."""
        if isinstance(message, BusEndMessage):
            await self.end(message.reason)
        elif isinstance(message, BusAddAgentMessage) and message.agent:
            await self.add_agent(message.agent)
        elif isinstance(message, BusClientConnectedMessage):
            await self._call_event_handler("on_client_connected", message.client)
        elif isinstance(message, BusClientDisconnectedMessage):
            await self._call_event_handler("on_client_disconnected", message.client)
        elif isinstance(message, BusUserTurnStartedMessage):
            await self._call_event_handler("on_user_turn_started")
        elif isinstance(message, BusUserTurnStoppedMessage):
            await self._call_event_handler("on_user_turn_stopped", message.message)
        elif isinstance(message, BusAssistantTurnStartedMessage):
            await self._call_event_handler("on_assistant_turn_started")
        elif isinstance(message, BusAssistantTurnStoppedMessage):
            await self._call_event_handler("on_assistant_turn_stopped", message.message)

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

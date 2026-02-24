#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User agent that bridges a transport to the agent bus.

Provides the `UserAgent` class that owns the transport pipeline (input,
VAD, STT, TTS, output) and connects it to the bus via `BusBridgeProcessor`.
Optionally creates shared context aggregators for turn detection.
"""

from typing import List, Optional

from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.task import CANCEL_TIMEOUT_SECS, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import BaseTransport
from pydantic import BaseModel, ConfigDict

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import (
    AgentBus,
    BusAssistantTurnStartedMessage,
    BusAssistantTurnStoppedMessage,
    BusBridgeProcessor,
    BusClientConnectedMessage,
    BusClientDisconnectedMessage,
    BusFrameMessage,
    BusMessage,
    BusUserTurnStartedMessage,
    BusUserTurnStoppedMessage,
)

USER_AGENT_NAME = "user"


class UserAgentParams(BaseModel):
    """Configuration for the user agent (transport bridge).

    Parameters:
        transport: The transport (WebRTC, WebSocket, etc.) to bridge.
        context: Optional shared `LLMContext` for turn detection.
        vad_analyzer: Optional VAD analyzer for speech detection.
        stt: Optional STT service.
        tts: Optional TTS service.
        pipeline_params: Optional `PipelineParams` for the pipeline task.
        observers: Optional list of `BaseObserver` instances for monitoring.
        user_aggregator_params: Optional `LLMUserAggregatorParams` for the
            user aggregator.
        assistant_aggregator_params: Optional `LLMAssistantAggregatorParams`
            for the assistant aggregator.
        audio_buffer: Optional audio buffer processor for recording.
        cancel_on_idle_timeout: Whether to cancel the pipeline on idle
            timeout. Defaults to True.
        cancel_timeout_secs: Timeout in seconds for pipeline cancellation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: BaseTransport
    context: Optional[LLMContext] = None
    vad_analyzer: Optional[VADAnalyzer] = None
    stt: Optional[FrameProcessor] = None
    tts: Optional[FrameProcessor] = None
    pipeline_params: Optional[PipelineParams] = None
    observers: Optional[List[BaseObserver]] = None
    user_aggregator_params: Optional[LLMUserAggregatorParams] = None
    assistant_aggregator_params: Optional[LLMAssistantAggregatorParams] = None
    audio_buffer: Optional[FrameProcessor] = None
    cancel_on_idle_timeout: bool = True
    cancel_timeout_secs: float = CANCEL_TIMEOUT_SECS


class UserAgent(BaseAgent):
    """The user agent — owns the transport pipeline and bridges it to the bus.

    Bridges the transport (WebRTC, WebSocket, etc.) to the bus, with shared
    STT/TTS. When a shared `LLMContext` is provided, creates context
    aggregators for turn detection and context tracking:

        ``input → [VAD] → [STT] → user_agg → BusBridge → [TTS] → output → assistant_agg``

    Without context, the pipeline is a simple transport bridge:

        ``input → [VAD] → [STT] → BusBridge → [TTS] → output``
    """

    def __init__(self, *, bus: AgentBus, params: UserAgentParams):
        """Initialize the UserAgent.

        Args:
            bus: The `AgentBus` for inter-agent communication.
            params: Configuration for the transport, context, and services.
        """
        super().__init__(
            USER_AGENT_NAME,
            bus=bus,
            active=True,
            enable_bus_output=False,
            enable_rtvi=True,
            pipeline_params=params.pipeline_params,
            observers=params.observers,
            cancel_on_idle_timeout=params.cancel_on_idle_timeout,
            cancel_timeout_secs=params.cancel_timeout_secs,
        )
        self._params = params

        self._context_aggregator_pair = self._build_aggregators() if params.context else None

        transport = params.transport

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await self.send_message(BusClientConnectedMessage(source=self.name, client=client))

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await self.send_message(BusClientDisconnectedMessage(source=self.name, client=client))

    def build_pipeline_processors(self) -> List[FrameProcessor]:
        """Build the transport pipeline processors.

        Assembles the pipeline chain and wires aggregator events if a
        shared context was provided.

        Returns:
            Ordered list of frame processors for the pipeline.
        """
        user_agg, assistant_agg = None, None
        if self._context_aggregator_pair:
            user_agg, assistant_agg = self._context_aggregator_pair
            self._wire_aggregator_events(user_agg, assistant_agg)

        bus_bridge = BusBridgeProcessor(
            bus=self._bus,
            agent_name=USER_AGENT_NAME,
            name="UserBusBridge",
        )

        processors: list[FrameProcessor] = []

        # Input transport
        processors.append(self._params.transport.input())

        # VAD Analyzer
        if self._params.vad_analyzer:
            processors.append(VADProcessor(vad_analyzer=self._params.vad_analyzer))

        # STT
        if self._params.stt:
            processors.append(self._params.stt)

        # User aggregator
        if user_agg:
            processors.append(user_agg)

        # Pipeline <-> Bus bridge
        processors.append(bus_bridge)

        # TTS
        if self._params.tts:
            processors.append(self._params.tts)

        # Output transport
        processors.append(self._params.transport.output())

        # Audio buffer (recording)
        if self._params.audio_buffer:
            processors.append(self._params.audio_buffer)

        # Assistant aggregator
        if assistant_agg:
            processors.append(assistant_agg)

        return processors

    @property
    def context_aggregator(self) -> LLMContextAggregatorPair | None:
        """The context aggregator pair, or None if no context was provided."""
        return self._context_aggregator_pair

    def _build_aggregators(self) -> LLMContextAggregatorPair:
        """Create the context aggregator pair.

        Returns:
            The `LLMContextAggregatorPair` with the configured params.
        """
        kwargs = {}
        if self._params.user_aggregator_params:
            kwargs["user_params"] = self._params.user_aggregator_params
        if self._params.assistant_aggregator_params:
            kwargs["assistant_params"] = self._params.assistant_aggregator_params
        return LLMContextAggregatorPair(self._params.context, **kwargs)

    def _wire_aggregator_events(self, user_agg, assistant_agg):
        @user_agg.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            await self.send_message(BusUserTurnStartedMessage(source=self.name))

        @user_agg.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
            await self.send_message(BusUserTurnStoppedMessage(source=self.name, message=message))

        @assistant_agg.event_handler("on_assistant_turn_started")
        async def on_assistant_turn_started(aggregator):
            await self.send_message(BusAssistantTurnStartedMessage(source=self.name))

        @assistant_agg.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            await self.send_message(
                BusAssistantTurnStoppedMessage(source=self.name, message=message)
            )

    async def _handle_bus_message(self, message: BusMessage) -> None:
        """Handle non-frame bus messages. Frame routing is handled by BusBridge."""
        # Ignore own messages
        if message.source == self.name:
            return
        # BusBridgeProcessor handles BusFrameMessage directly
        if isinstance(message, BusFrameMessage):
            return
        # Ignore targeted messages for other agents
        if message.target and message.target != self.name:
            return

        await self.on_bus_message(message)

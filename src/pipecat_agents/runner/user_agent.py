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
from pipecat.pipeline.task import PipelineParams
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
from pipecat.utils.context.llm_context_summarization import LLMContextSummarizationConfig

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


class UserAgent(BaseAgent):
    """The user agent — owns the transport pipeline and bridges it to the bus.

    Bridges the transport (WebRTC, WebSocket, etc.) to the bus, with shared
    STT/TTS. When a shared `LLMContext` is provided, creates context
    aggregators for turn detection and context tracking:

        ``input → [VAD] → [STT] → user_agg → BusBridge → [TTS] → output → assistant_agg``

    Without context, the pipeline is a simple transport bridge:

        ``input → [VAD] → [STT] → BusBridge → [TTS] → output``
    """

    def __init__(
        self,
        *,
        bus: AgentBus,
        transport: BaseTransport,
        context: Optional[LLMContext] = None,
        vad_analyzer: Optional[VADAnalyzer] = None,
        stt: Optional[FrameProcessor] = None,
        tts: Optional[FrameProcessor] = None,
        pipeline_params: Optional[PipelineParams] = None,
    ):
        """Initialize the UserAgent.

        Args:
            bus: The `AgentBus` for inter-agent communication.
            transport: The transport (WebRTC, WebSocket, etc.) to bridge.
            context: Optional shared `LLMContext`. When provided, creates
                aggregators for turn detection and context tracking.
            vad_analyzer: Optional VAD analyzer for speech detection.
            stt: Optional STT service shared across all agents.
            tts: Optional TTS service shared across all agents.
            pipeline_params: Optional `PipelineParams` for the pipeline task.
        """
        super().__init__(
            USER_AGENT_NAME,
            bus=bus,
            enabled=True,
            enable_bus_output=False,
            enable_rtvi=True,
            context=context,
            pipeline_params=pipeline_params,
        )
        self._transport = transport
        self._context = context
        self._vad_analyzer = vad_analyzer
        self._stt = stt
        self._tts = tts

        self._context_aggregator_pair = LLMContextAggregatorPair(context) if context else None

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await self.send_message(BusClientConnectedMessage(source=self.name, client=client))

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await self.send_message(BusClientDisconnectedMessage(source=self.name, client=client))

    def build_pipeline_processors(self) -> List[FrameProcessor]:
        user_agg, assistant_agg = self._build_aggregators()

        bus_bridge = BusBridgeProcessor(
            bus=self._bus,
            agent_name=USER_AGENT_NAME,
            name="UserBusBridge",
        )

        processors: list[FrameProcessor] = []

        # Input transport
        processors.append(self._transport.input())

        # VAD Analyzer
        if self._vad_analyzer:
            processors.append(VADProcessor(vad_analyzer=self._vad_analyzer))

        # STT
        if self._stt:
            processors.append(self._stt)

        # User aggregator
        if user_agg:
            processors.append(user_agg)

        # Pipeline <-> Bus bridge
        processors.append(bus_bridge)

        # TTS
        if self._tts:
            processors.append(self._tts)

        # Output transport
        processors.append(self._transport.output())

        # Assistant aggregator
        if assistant_agg:
            processors.append(assistant_agg)

        return processors

    @property
    def context_aggregator(self) -> LLMContextAggregatorPair | None:
        """The context aggregator pair, or None if no context was provided."""
        return self._context_aggregator_pair

    def _build_aggregators(self):
        """Wire context aggregators if a shared context is provided.

        Returns (user_agg, assistant_agg) or (None, None).
        """
        if not self._context_aggregator_pair:
            return None, None

        user_agg, assistant_agg = LLMContextAggregatorPair(
            self._context,
            user_params=LLMUserAggregatorParams(
                # This uses the LLM to allow the user more time to response,
                # based on the context of the conversation.
                filter_incomplete_user_turns=True,
            ),
            assistant_params=LLMAssistantAggregatorParams(
                enable_context_summarization=True,
                context_summarization_config=LLMContextSummarizationConfig(
                    max_context_tokens=8000,  # Trigger at 8000 tokens
                    target_context_tokens=6000,  # Target summary size
                    max_unsummarized_messages=20,  # Or trigger after 20 new messages
                    min_messages_after_summary=4,  # Keep last 4 messages uncompressed
                ),
            ),
        )

        self._wire_aggregator_events(user_agg, assistant_agg)

        return user_agg, assistant_agg

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

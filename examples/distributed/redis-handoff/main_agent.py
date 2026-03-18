#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Main transport agent — run on Machine A.

Handles audio I/O (STT, TTS) and bridges frames to the bus. The LLM agents
run as separate processes (possibly on separate machines) connected via Redis.

Usage:
    python main_agent.py --redis-url redis://localhost:6379

Requirements:
    DEEPGRAM_API_KEY, CARTESIA_API_KEY, DAILY_API_KEY
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from redis.asyncio import Redis

from pipecat_subagents.agents import BaseAgent, LLMActivationArgs
from pipecat_subagents.bus import AgentBus, BusAgentRegisteredMessage, BusBridgeProcessor, BusMessage
from pipecat_subagents.bus.network.redis import RedisBus
from pipecat_subagents.bus.serializers import JSONMessageSerializer
from pipecat_subagents.runner import AgentRunner

load_dotenv(override=True)


class AcmeAgent(BaseAgent):
    """Owns the transport pipeline and bridges frames to/from the bus.

    Has no LLM — the BusBridge sends user speech to whichever LLM agent is
    active, and receives responses back through TTS to the user.
    """

    def __init__(self, name: str, *, bus: AgentBus, transport: DailyTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    async def on_bus_message(self, message: BusMessage) -> None:
        await super().on_bus_message(message)
        # In the distributed case, LLM agents aren't children — listen for
        # their registration directly via bus messages.
        if isinstance(message, BusAgentRegisteredMessage) and message.agent_name == "greeter":
            await self.activate_agent(
                "greeter",
                args=LLMActivationArgs(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Welcome the user to Acme Corp, mention the available "
                                "products and ask how you can help."
                            ),
                        },
                    ],
                ),
            )

    def build_pipeline_task(self, pipeline: Pipeline) -> PipelineTask:
        return PipelineTask(pipeline, enable_rtvi=True)

    async def build_pipeline(self) -> Pipeline:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaTTSSettings(
                voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
            ),
        )

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        bridge = BusBridgeProcessor(
            bus=self.bus,
            agent_name=self.name,
            name=f"{self.name}::BusBridge",
        )

        @self._transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await self.activate_agent(self.name)

        @self._transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await self.cancel()

        return Pipeline(
            [
                self._transport.input(),
                stt,
                context_aggregator.user(),
                bridge,
                tts,
                self._transport.output(),
                context_aggregator.assistant(),
            ]
        )


async def main():
    parser = argparse.ArgumentParser(description="Main transport agent")
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis URL"
    )
    parser.add_argument(
        "--channel", default="pipecat:acme", help="Redis pub/sub channel"
    )
    args = parser.parse_args()

    redis = Redis.from_url(args.redis_url)
    serializer = JSONMessageSerializer()
    bus = RedisBus(redis=redis, serializer=serializer, channel=args.channel)

    transport = DailyTransport(
        os.getenv("DAILY_ROOM_URL", ""),
        os.getenv("DAILY_ROOM_TOKEN", ""),
        "Acme Bot",
        DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    runner = AgentRunner(bus=bus, handle_sigint=True)
    main_agent = AcmeAgent("acme", bus=bus, transport=transport)
    await runner.add_agent(main_agent)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())

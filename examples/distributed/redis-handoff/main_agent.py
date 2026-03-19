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
    DEEPGRAM_API_KEY, CARTESIA_API_KEY
"""

import argparse
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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from redis.asyncio import Redis

from pipecat_subagents.agents import BaseAgent, LLMActivationArgs
from pipecat_subagents.bus import AgentBus, BusBridgeProcessor
from pipecat_subagents.bus.network.redis import RedisBus
from pipecat_subagents.runner import AgentRunner
from pipecat_subagents.types import RegisteredAgentData

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


class AcmeAgent(BaseAgent):
    """Owns the transport pipeline and bridges frames to/from the bus.

    Has no LLM — the BusBridge sends user speech to whichever LLM agent is
    active, and receives responses back through TTS to the user.
    """

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport
        self._client_connected = False
        self._greeter_registered = False

    async def on_started(self) -> None:
        await super().on_started()
        await self.watch_agent("greeter")

    async def on_agent_ready(self, agent_info: RegisteredAgentData) -> None:
        self._greeter_registered = True
        await self._maybe_activate_greeter()

    async def _maybe_activate_greeter(self) -> None:
        if not self._client_connected or not self._greeter_registered:
            return
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
            self._client_connected = True
            await self._maybe_activate_greeter()

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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    redis = Redis.from_url(runner_args.cli_args.redis_url)
    bus = RedisBus(redis=redis, channel=runner_args.cli_args.channel)

    runner = AgentRunner(bus=bus, handle_sigint=True)
    main_agent = AcmeAgent("acme", bus=bus, transport=transport)
    await runner.add_agent(main_agent)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    parser = argparse.ArgumentParser(description="Main transport agent")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--channel", default="pipecat:acme", help="Redis pub/sub channel")

    main(parser)

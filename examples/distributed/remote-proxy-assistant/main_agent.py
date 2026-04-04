#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Main transport agent with a WebSocket proxy to a remote LLM server.

Handles audio I/O (STT, TTS) and bridges frames to the bus. A proxy
agent forwards bus messages to a remote LLM server over WebSocket.

Usage:
    python main_agent.py --remote-agent-url ws://localhost:8765/ws

Requirements:
    DEEPGRAM_API_KEY, CARTESIA_API_KEY
"""

import argparse
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
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

from pipecat_subagents.agents import BaseAgent, LLMAgentActivationArgs, agent_ready
from pipecat_subagents.agents.proxy import WebSocketProxyClientAgent
from pipecat_subagents.bus import AgentBus, BusBridgeProcessor, BusFrameMessage
from pipecat_subagents.runner import AgentRunner
from pipecat_subagents.types import AgentReadyData

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

    A WebSocket proxy forwards bus messages to the remote LLM server.
    """

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    @agent_ready("assistant")
    async def on_assistant_ready(self, data: AgentReadyData) -> None:
        logger.info("Remote assistant agent is ready, activating")
        await self.activate_agent(
            "assistant",
            args=LLMAgentActivationArgs(
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
        return PipelineTask(
            pipeline,
            enable_rtvi=True,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

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
            await self.activate_agent("proxy")

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
    runner = AgentRunner(handle_sigint=runner_args.handle_sigint)

    main_agent = AcmeAgent("acme", bus=runner.bus, transport=transport)
    await runner.add_agent(main_agent)

    # Create a proxy to the remote LLM server. Will connect when activated.
    proxy = WebSocketProxyClientAgent(
        "proxy",
        bus=runner.bus,
        url=runner_args.cli_args.remote_agent_url,
        local_agent_name="acme",
        remote_agent_name="assistant",
        forward_messages=(BusFrameMessage,),
    )
    await runner.add_agent(proxy)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    parser = argparse.ArgumentParser(description="Main transport agent with proxy")
    parser.add_argument(
        "--remote-agent-url",
        default="ws://localhost:8765/ws",
        help="WebSocket URL of the remote LLM server",
    )

    main(parser)

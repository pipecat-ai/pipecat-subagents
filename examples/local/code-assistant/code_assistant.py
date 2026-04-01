#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice code assistant powered by Claude Agent SDK.

Talk to your codebase hands-free. Ask questions like "what does the
auth middleware do?" or "find all TODO comments" and get spoken answers
based on actual file contents. The Claude Agent SDK worker navigates
the filesystem using Read, Bash, Glob, and Grep tools.

Architecture:

    CodeAssistant (transport + BusBridge)
      └── Voice Agent (LLM, bridged)
            └── @tool ask_code(question)
                  └── Code Worker (Claude Agent SDK)

Requirements:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

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
from voice_agent import VoiceAgent

from pipecat_subagents.agents import BaseAgent, LLMAgentActivationArgs
from pipecat_subagents.bus import BusBridgeProcessor
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


class CodeAssistant(BaseAgent):
    """Owns the transport and bridges frames to the voice agent via the bus."""

    def __init__(self, name: str, *, bus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    async def on_agent_ready(self, data: AgentReadyData):
        await super().on_agent_ready(data)
        await self.activate_agent(
            "voice",
            args=LLMAgentActivationArgs(
                messages=[
                    {
                        "role": "user",
                        "content": "Greet the user and tell them you're a code assistant.",
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
            voice = VoiceAgent("voice", bus=self.bus)
            await self.add_agent(voice)
            await self.watch_agent("voice")

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
    assistant = CodeAssistant("main", bus=runner.bus, transport=transport)
    await runner.add_agent(assistant)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Single agent example — the simplest use of the agents framework.

A single BaseAgent that runs a complete voice pipeline (transport, STT, LLM,
TTS) through the AgentRunner. No bus bridge, no multi-agent coordination —
just one agent doing everything.

Requirements:
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
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
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat_subagents.agents import BaseAgent
from pipecat_subagents.bus import AgentBus
from pipecat_subagents.runner import AgentRunner

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


class SimpleAgent(BaseAgent):
    """A single agent with a complete voice pipeline."""

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

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
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a helpful assistant in a voice conversation. Your responses "
                    "will be spoken aloud, so avoid emojis, bullet points, or other formatting "
                    "that can't be spoken. Respond to what the user said in a creative, helpful, "
                    "and brief way."
                ),
            ),
        )

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        @self._transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await self.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {"role": "user", "content": "Greet the user and ask how you can help."}
                    ],
                    run_llm=True,
                )
            )

        @self._transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await self.end()

        return Pipeline(
            [
                self._transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                self._transport.output(),
                context_aggregator.assistant(),
            ]
        )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    runner = AgentRunner(handle_sigint=runner_args.handle_sigint)
    agent = SimpleAgent("assistant", bus=runner.bus, transport=transport)
    await runner.add_agent(agent)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

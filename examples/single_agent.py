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
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat_agents.agents import BaseAgent
from pipecat_agents.bus import AgentBus
from pipecat_agents.runner import AgentRunner

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

    async def build_pipeline_task(self) -> PipelineTask:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            system_instruction="You are a friendly voice assistant. Keep responses brief and conversational.",
        )

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())
                    ]
                ),
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        pipeline = Pipeline(
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

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @self._transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await self.activate_agent(self.name)

        @self._transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.cancel()

        return task


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    runner = AgentRunner(handle_sigint=runner_args.handle_sigint)
    agent = SimpleAgent("assistant", bus=runner.bus, transport=transport)

    @agent.event_handler("on_agent_activated")
    async def on_agent_activated(agent, args):
        await agent.queue_frame(
            LLMMessagesAppendFrame(
                messages=[
                    {"role": "system", "content": "Greet the user and ask how you can help."}
                ],
                run_llm=True,
            )
        )

    await runner.add_agent(agent)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

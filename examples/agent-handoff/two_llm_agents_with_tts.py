#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Two LLM agents with per-agent TTS voices.

Similar to two_llm_agents.py, but each LLM agent has its own TTS with a
distinct voice. The main agent has no TTS — audio comes from the LLM agents
through the bus. Agents announce transfers ("let me connect you with...")
before handing off.

Architecture:

    Main agent (no TTS):
      transport.in → STT → context_agg.user → BusBridge → transport.out
        → context_agg.assistant

    LLM agent (with TTS):
      BusInput → LLM → TTS → BusOutput

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
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat_subagents.agents import BaseAgent, LLMActivationArgs, LLMAgent, tool
from pipecat_subagents.bus import AgentBus, BusBridgeProcessor
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


class AcmeTTSAgent(LLMAgent):
    """Base agent for Acme Corp with per-agent TTS voice."""

    def __init__(self, name: str, *, bus: AgentBus, voice_id: str):
        super().__init__(name, bus=bus)
        self._voice_id = voice_id

    async def build_pipeline(self) -> Pipeline:
        pipeline = await super().build_pipeline()
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaTTSSettings(voice=self._voice_id),
        )
        return Pipeline([pipeline, tts])

    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(self, params: FunctionCallParams, agent: str, reason: str):
        """Transfer the user to another agent.

        Args:
            agent (str): The agent to transfer to (e.g. 'greeter', 'support').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(
                messages=[
                    {"role": "user", "content": f"Tell the user about the transfer ({reason})."}
                ],
                run_llm=True,
            )
        )
        await self.deactivate_agent(result_callback=params.result_callback)
        await self.activate_agent(
            agent,
            args=LLMActivationArgs(
                messages=[{"role": "user", "content": reason}],
            ),
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"Agent '{self.name}': ending conversation ({reason})")
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": reason}], run_llm=True)
        )
        await self.end(reason=reason, result_callback=params.result_callback)


class GreeterAgent(AcmeTTSAgent):
    """Greets the user and routes to support when needed."""

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(
            name,
            bus=bus,
            voice_id="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        )

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a friendly greeter for Acme Corp. The available products "
                    "are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme "
                    "Tornado Kit. Ask which one they'd like to learn more about. "
                    "When the user picks a product or asks a question about one, "
                    "call the transfer_to_agent tool with agent 'support'. "
                    "Do not answer product questions yourself. If the user says goodbye, "
                    "call the end_conversation tool. "
                    "Keep responses brief — this is a voice conversation."
                )
            ),
        )


class SupportAgent(AcmeTTSAgent):
    """Handles support questions and can transfer back to the greeter."""

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(
            name,
            bus=bus,
            voice_id="a167e0f3-df7e-4d52-a9c3-f949145efdab",  # Blake
        )

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a support agent for Acme Corp. You know about three "
                    "products: Acme Rocket Boots (jet-powered boots, $299, run up "
                    "to 60 mph), Acme Invisible Paint (makes anything invisible for "
                    "24 hours, $49 per can), and Acme Tornado Kit (portable tornado "
                    "generator, $199, batteries included). Answer the user's questions "
                    "about these products. If the user wants to browse other products "
                    "or start over, call the transfer_to_agent tool with agent "
                    "'greeter'. "
                    "If the user says goodbye, call the end_conversation tool. "
                    "Keep responses brief — this is a voice conversation."
                ),
            ),
        )


class AcmeAgent(BaseAgent):
    """Owns the transport pipeline and bridges frames to/from the bus.

    Has no LLM or TTS — audio and text come from LLM agents through the bus.
    The context_aggregator.assistant() captures text frames for shared context.
    """

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    async def on_agent_started(self) -> None:
        await super().on_agent_started()

        greeter = GreeterAgent("greeter", bus=self.bus)
        support = SupportAgent("support", bus=self.bus)
        for agent in [greeter, support]:
            await self.add_agent(agent)

    async def on_agent_registered(self, agent_name: str) -> None:
        if agent_name == "greeter":
            await self.activate_agent(
                "greeter",
                args=LLMActivationArgs(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Welcome the user to Acme Corp, mention the available products "
                                "and ask how you can help."
                            ),
                        },
                    ],
                ),
            )

    def build_pipeline_task(self, pipeline: Pipeline) -> PipelineTask:
        return PipelineTask(pipeline, enable_rtvi=True)

    async def build_pipeline(self) -> Pipeline:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

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
                self._transport.output(),
                context_aggregator.assistant(),
            ]
        )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    runner = AgentRunner(handle_sigint=runner_args.handle_sigint)
    main = AcmeAgent("acme", bus=runner.bus, transport=transport)
    await runner.add_agent(main)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Parallel debate using task groups.

A voice agent receives a topic from the user and spawns three worker
agents in parallel using the request_task_group() context manager. Each worker
runs its own LLM pipeline with a context, so it remembers previous
topics. The voice agent collects all three perspectives and synthesizes
a balanced response.

Architecture:

    Debate Agent (transport + BusBridge)
      └── Moderator Agent (LLM, bridged)
            └── @tool debate(topic)
                  └── request_task_group(advocate, critic, analyst)
                         └── Debate Worker (LLM + context aggregators)

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
    AssistantTurnStoppedMessage,
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

ROLE_PROMPTS = {
    "advocate": (
        "You argue IN FAVOR of the topic. Present the strongest case for why "
        "this is a good idea, with concrete benefits. Be persuasive but honest."
        "Be concise, just 2-3 sentences."
    ),
    "critic": (
        "You argue AGAINST the topic. Present the strongest concerns, risks, "
        "and downsides. Be critical but fair. Be concise, just 2-3 sentences."
    ),
    "analyst": (
        "You provide a BALANCED, NEUTRAL analysis. Weigh both sides objectively "
        "and highlight the key trade-offs. Be concise, just 2-3 sentences."
    ),
}


class DebateWorker(LLMAgent):
    """Worker that generates a perspective using its own LLM pipeline.

    Each worker keeps its own LLMContext, so it remembers previous topics
    across multiple debate rounds.
    """

    def __init__(self, name: str, *, bus: AgentBus, role: str):
        super().__init__(name, bus=bus)
        self._role = role

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(system_instruction=ROLE_PROMPTS[self._role]),
        )

    async def build_pipeline(self) -> Pipeline:
        llm = self.build_llm()

        context = LLMContext()
        user_agg, assistant_agg = LLMContextAggregatorPair(context)

        @assistant_agg.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            text = message.content
            logger.info(f"Worker '{self.name}': completed ({len(text)} chars)")
            await self.send_task_response({"role": self._role, "text": text})

        return Pipeline([user_agg, llm, assistant_agg])

    async def on_task_request(self, task_id: str, requester: str, payload):
        await super().on_task_request(task_id, requester, payload)
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": f"Topic: {payload['topic']}"}],
                run_llm=True,
            )
        )


class ModeratorAgent(LLMAgent):
    """Debate moderator that spawns parallel workers via request_task_group()."""

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(name, bus=bus, bridged=True)
        self._workers = []

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a debate moderator in a voice conversation. When the user "
                    "gives you a topic, call the debate tool to gather perspectives from "
                    "three viewpoints (advocate, critic, analyst). Then synthesize the "
                    "results into a clear, balanced summary for the user. Keep your "
                    "responses concise and natural for speaking."
                ),
            ),
        )

    async def on_ready(self) -> None:
        await super().on_ready()
        self._workers = [
            DebateWorker(role, bus=self.bus, role=role)
            for role in ("advocate", "critic", "analyst")
        ]

    @tool(cancel_on_interruption=False)
    async def debate(self, params: FunctionCallParams, topic: str):
        """Analyze a topic from multiple perspectives (advocate, critic, analyst).

        Args:
            topic (str): The topic or question to debate.
        """
        logger.info(f"Agent '{self.name}': starting debate on '{topic}'")

        async with self.request_task_group(
            *self._workers, payload={"topic": topic}, timeout=30
        ) as tg:
            pass

        result = "\n\n".join(f"{r['role'].upper()}: {r['text']}" for r in tg.responses.values())

        logger.info(f"Agent '{self.name}': debate complete, synthesizing")

        await params.result_callback(result)


class DebateAgent(BaseAgent):
    """Owns the transport and bridges frames to the voice agent via the bus."""

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    async def on_agent_ready(self, data: AgentReadyData):
        await super().on_agent_ready(data)
        if data.agent_name == "moderator":
            await self.activate_agent(
                "moderator",
                args=LLMActivationArgs(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Greet the user and tell them you can moderate a debate "
                                "on any topic. Ask what they'd like to explore."
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
            moderator = ModeratorAgent("moderator", bus=self.bus)
            await self.add_agent(moderator)

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
    debate = DebateAgent("main", bus=runner.bus, transport=transport)
    await runner.add_agent(debate)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

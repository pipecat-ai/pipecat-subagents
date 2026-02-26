#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Two LLM agents with a main agent bridging transport to the bus.

Demonstrates multi-agent coordination: a main agent handles transport I/O
(STT, TTS) and bridges frames to the bus. Two LLM agents — a greeter and a
support agent — each run their own pipeline and transfer control between
each other.

The user talks to one agent at a time. Transfers are seamless — the LLM
decides when to hand off based on its tools.

Requirements:
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
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
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat_agents.agents import BaseAgent, LLMAgent
from pipecat_agents.bus import (
    AgentActivationArgs,
    AgentBus,
    BusBridgeProcessor,
    BusEndAgentMessage,
    BusEndMessage,
    BusMessage,
)
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


# ── LLM Agents ──────────────────────────────────────────────────


class GreeterAgent(LLMAgent):
    """Greets the user and routes to support when needed."""

    def build_llm(self) -> LLMService:
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            system_instruction=(
                "You are a friendly greeter for Acme Corp. Welcome the user, ask how "
                "you can help, and transfer them to support when they have a question "
                "or issue. Keep responses brief — this is a voice conversation."
            ),
        )
        llm.register_function(
            "transfer_to_support", self._handle_transfer_to_support, cancel_on_interruption=False
        )
        llm.register_function("end_conversation", self._handle_end)
        return llm

    def build_tools(self):
        return [
            FunctionSchema(
                name="transfer_to_support",
                description="Transfer to the support agent when the user needs help.",
                properties={
                    "reason": {
                        "type": "string",
                        "description": "Brief summary of what the user needs help with.",
                    },
                },
                required=["reason"],
            ),
            FunctionSchema(
                name="end_conversation",
                description="End the conversation when the user says goodbye.",
                properties={},
                required=[],
            ),
        ]

    async def _handle_transfer_to_support(self, params):
        reason = params.arguments["reason"]
        logger.info(f"Greeter: transferring to support ({reason})")
        await self.transfer_to(
            "support",
            args=AgentActivationArgs(
                messages=[{"role": "user", "content": f"Help the user with: {reason}"}],
            ),
            result_callback=params.result_callback,
        )

    async def _handle_end(self, params):
        logger.info("Greeter: ending conversation")
        await self.end(
            reason="User said goodbye",
            result="Say goodbye briefly.",
            result_callback=params.result_callback,
        )


class SupportAgent(LLMAgent):
    """Handles support questions and can transfer back to the greeter."""

    def build_llm(self) -> LLMService:
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            system_instruction=(
                "You are a helpful support agent for Acme Corp. Answer the user's "
                "questions about products and services. If the user wants to do "
                "something else (like start over or say hi again), transfer them "
                "back to the greeter. Keep responses brief — this is a voice conversation."
            ),
        )
        llm.register_function(
            "transfer_to_greeter", self._handle_transfer_to_greeter, cancel_on_interruption=False
        )
        llm.register_function("end_conversation", self._handle_end)
        return llm

    def build_tools(self):
        return [
            FunctionSchema(
                name="transfer_to_greeter",
                description="Transfer back to the greeter when the user wants to start over.",
                properties={
                    "reason": {
                        "type": "string",
                        "description": "Why the user is being transferred back.",
                    },
                },
                required=["reason"],
            ),
            FunctionSchema(
                name="end_conversation",
                description="End the conversation when the user says goodbye.",
                properties={},
                required=[],
            ),
        ]

    async def _handle_transfer_to_greeter(self, params):
        reason = params.arguments["reason"]
        logger.info(f"Support: transferring back to greeter ({reason})")
        await self.transfer_to(
            "greeter",
            args=AgentActivationArgs(
                messages=[{"role": "user", "content": f"The user is back: {reason}"}],
            ),
            result_callback=params.result_callback,
        )

    async def _handle_end(self, params):
        logger.info("Support: ending conversation")
        await self.end(
            reason="User said goodbye",
            result="Say goodbye briefly.",
            result_callback=params.result_callback,
        )


# ── Main Agent ───────────────────────────────────────────────────


class MainAgent(BaseAgent):
    """Owns the transport pipeline and bridges frames to/from the bus.

    Has no LLM — the BusBridge forwards user speech to whichever LLM agent
    is active, and routes its responses back through TTS to the user.
    """

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus, active=True)
        self._transport = transport
        self._sub_agents: list[str] = []

    async def build_pipeline_task(self) -> PipelineTask:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
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

        bus_bridge = BusBridgeProcessor(
            bus=self.bus,
            agent_name=self.name,
            name=f"{self.name}::BusBridge",
        )

        # Create sub-agents
        greeter = GreeterAgent("greeter", bus=self.bus, parent=self.name)
        support = SupportAgent("support", bus=self.bus, parent=self.name)
        for agent in [greeter, support]:
            self._sub_agents.append(agent.name)
            await self.add_agent(agent)

        # Wire transport events
        @self._transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await self.activate_agent(
                "greeter",
                args=AgentActivationArgs(
                    messages=[
                        {
                            "role": "user",
                            "content": "Greet the user and ask how you can help.",
                        }
                    ],
                ),
            )

        @self._transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await self.cancel()

        pipeline = Pipeline(
            [
                self._transport.input(),
                stt,
                context_aggregator.user(),
                bus_bridge,
                tts,
                self._transport.output(),
                context_aggregator.assistant(),
            ]
        )

        return PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    async def on_bus_message(self, message: BusMessage) -> None:
        """End sub-agents before ending self."""
        if isinstance(message, BusEndAgentMessage):
            for sub_name in self._sub_agents:
                await self.send_message(
                    BusEndAgentMessage(source=self.name, target=sub_name, reason=message.reason)
                )
            if self._task:
                await self._task.queue_frame(EndFrame(reason=message.reason))
            await self.send_message(BusEndMessage(source=self.name, reason=message.reason))
        else:
            await super().on_bus_message(message)


# ── Entry point ──────────────────────────────────────────────────


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    runner = AgentRunner(handle_sigint=runner_args.handle_sigint)
    main = MainAgent("main", bus=runner.bus, transport=transport)
    await runner.add_agent(main)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

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
from pipecat.frames.frames import EndFrame, LLMContextFrame
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

from pipecat_agents.agents import BaseAgent, LLMContextAgent
from pipecat_agents.bus import (
    AgentActivationArgs,
    AgentBus,
    BusEndAgentMessage,
    BusEndMessage,
    BusInputProcessor,
    BusMessage,
    BusOutputProcessor,
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


class AcmeLLMAgent(LLMContextAgent):
    """Base agent for Acme Corp with transfer and end tools."""

    def build_llm(self) -> LLMService:
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        llm.register_function(
            "transfer_to_agent", self._handle_transfer, cancel_on_interruption=False
        )
        llm.register_function("end_conversation", self._handle_end)
        return llm

    def build_tools(self):
        return [
            FunctionSchema(
                name="transfer_to_agent",
                description="Transfer the user to another agent.",
                properties={
                    "agent": {
                        "type": "string",
                        "description": "The agent to transfer to (e.g. 'greeter', 'support').",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the user is being transferred.",
                    },
                },
                required=["agent", "reason"],
            ),
            FunctionSchema(
                name="end_conversation",
                description="End the conversation when the user says goodbye.",
                properties={},
                required=[],
            ),
        ]

    async def _handle_transfer(self, params):
        agent = params.arguments["agent"]
        reason = params.arguments["reason"]
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await self.transfer_to(
            agent,
            args=AgentActivationArgs(
                messages=[{"role": "user", "content": reason}],
            ),
            result_callback=params.result_callback,
        )

    async def _handle_end(self, params):
        logger.info(f"Agent '{self.name}': ending conversation")
        await self.end(
            reason="User said goodbye",
            result="Say goodbye briefly.",
            result_callback=params.result_callback,
        )


class GreeterAgent(AcmeLLMAgent):
    """Greets the user and routes to support when needed."""

    def __init__(self, name: str, **kwargs):
        super().__init__(
            name,
            system_messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly greeter for Acme Corp. The available products "
                        "are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme "
                        "Tornado Kit. Ask which one they'd like to learn more about. "
                        "When the user picks a product or asks a question about one, "
                        "immediately call the transfer_to_agent tool with agent 'support'. "
                        "Do not answer product questions yourself. If the user says goodbye, "
                        "call the end_conversation tool. Do not mention transferring — just do it "
                        "seamlessly. Keep responses brief — this is a voice conversation."
                    ),
                }
            ],
            **kwargs,
        )


class SupportAgent(AcmeLLMAgent):
    """Handles support questions and can transfer back to the greeter."""

    def __init__(self, name: str, **kwargs):
        super().__init__(
            name,
            system_messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a support agent for Acme Corp. You know about three "
                        "products: Acme Rocket Boots (jet-powered boots, $299, run up "
                        "to 60 mph), Acme Invisible Paint (makes anything invisible for "
                        "24 hours, $49 per can), and Acme Tornado Kit (portable tornado "
                        "generator, $199, batteries included). Answer the user's questions "
                        "about these products. If the user wants to browse other products "
                        "or start over, call the transfer_to_agent tool with agent "
                        "'greeter'. If the user says goodbye, call the end_conversation "
                        "tool. Do not mention transferring — just do it seamlessly. "
                        "Keep responses brief — this is a voice conversation."
                    ),
                }
            ],
            **kwargs,
        )


class AcmeAgent(BaseAgent):
    """Owns the transport pipeline and bridges frames to/from the bus.

    Has no LLM — the BusOutput sends user speech to whichever LLM agent is
    active, and BusInput receives responses back through TTS to the user.
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

        bus_output = BusOutputProcessor(
            bus=self.bus,
            agent_name=self.name,
            output_frames=(LLMContextFrame,),
            pass_through=True,
            name=f"{self.name}::BusOutput",
        )
        bus_input = BusInputProcessor(
            bus=self.bus,
            agent_name=self.name,
            name=f"{self.name}::BusInput",
        )

        pipeline = Pipeline(
            [
                self._transport.input(),
                stt,
                context_aggregator.user(),
                bus_output,
                bus_input,
                tts,
                self._transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

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
                            "content": (
                                "Welcome the user to Acme Corp, mention the available products "
                                "and ask how you can help."
                            ),
                        },
                    ],
                ),
            )

        @self._transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await self.cancel()

        return task

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

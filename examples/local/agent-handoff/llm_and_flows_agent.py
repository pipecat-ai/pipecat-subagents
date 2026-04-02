#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent + Flows agent with a main agent bridging transport.

Demonstrates mixing agent types: a main agent handles transport I/O, an LLM
agent acts as a router, and a Flows agent handles a structured restaurant
reservation flow. The router transfers to the reservation agent when the user
wants to book a table, and the reservation agent can transfer back to the
router when done.

Requirements:
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
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
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat_flows import FlowManager, FlowResult, NodeConfig

from pipecat_subagents.agents import (
    BaseAgent,
    LLMAgent,
    LLMAgentActivationArgs,
    tool,
)
from pipecat_subagents.agents.flows_agent import FlowsAgent
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


class MockReservationSystem:
    """Simulates a restaurant reservation API."""

    booked_times = {"7:00 PM", "8:00 PM"}

    async def check_availability(self, party_size: int, time: str) -> tuple[bool, list[str]]:
        await asyncio.sleep(0.5)
        is_available = time not in self.booked_times
        alternatives = []
        if not is_available:
            all_times = ["5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM"]
            alternatives = [t for t in all_times if t not in self.booked_times]
        return is_available, alternatives


class ReservationAgent(FlowsAgent):
    """Structured reservation flow using Pipecat Flows."""

    def __init__(self, name: str, *, bus: AgentBus, context_aggregator, reservation_system):
        super().__init__(name, bus=bus, context_aggregator=context_aggregator)
        self._reservation_system = reservation_system

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a reservation assistant for La Maison, an upscale French "
                    "restaurant. Be casual and friendly. This is a voice conversation."
                ),
            ),
        )

    def build_initial_node(self) -> NodeConfig:
        return {
            "name": "initial",
            "task_messages": [
                {
                    "role": "user",
                    "content": "Ask how many people are in their party.",
                }
            ],
            "functions": [self.collect_party_size],
        }

    def _create_time_selection_node(self) -> NodeConfig:
        return {
            "name": "get_time",
            "task_messages": [
                {
                    "role": "user",
                    "content": "Ask what time they'd like to dine. The restaurant is open 5 PM to 10 PM.",
                }
            ],
            "functions": [self.check_availability],
        }

    def _create_confirmation_node(self) -> NodeConfig:
        return {
            "name": "confirm",
            "task_messages": [
                {
                    "role": "user",
                    "content": "Confirm the reservation details and ask if there's anything else.",
                }
            ],
            "functions": [self.end_reservation],
        }

    async def collect_party_size(
        self, flow_manager: FlowManager, size: int
    ) -> tuple[FlowResult, NodeConfig]:
        """Record the number of people in the party.

        Args:
            size (int): Number of people in the party. Must be between 1 and 12.
        """
        return {"size": size, "status": "success"}, self._create_time_selection_node()

    async def check_availability(
        self, flow_manager: FlowManager, time: str, party_size: int
    ) -> tuple[FlowResult, NodeConfig]:
        """Check availability for the requested time.

        Args:
            time (str): Reservation time (e.g., '6:00 PM').
            party_size (int): Number of people in the party.
        """
        is_available, alternatives = await self._reservation_system.check_availability(
            party_size, time
        )

        if is_available:
            return {"time": time, "available": True}, self._create_confirmation_node()
        else:
            times_list = ", ".join(alternatives)
            return {"time": time, "available": False, "alternatives": alternatives}, {
                "name": "no_availability",
                "task_messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Apologize that {time} is not available. "
                            f"Suggest these alternative times: {times_list}."
                        ),
                    }
                ],
                "functions": [self.check_availability, self.end_reservation],
            }

    async def end_reservation(self, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Confirm and end the reservation."""
        return None, {
            "name": "end",
            "task_messages": [
                {
                    "role": "user",
                    "content": "Thank them for their reservation and say goodbye.",
                }
            ],
            "post_actions": [{"type": "end_conversation"}],
        }

    @tool
    async def transfer_to_agent(
        self, flow_manager: FlowManager, agent: str, reason: str
    ) -> tuple[FlowResult, NodeConfig]:
        """Transfer the user to another agent.

        Args:
            agent (str): The agent to transfer to (e.g. 'router').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await self.handoff_to(
            agent,
            args=LLMAgentActivationArgs(
                messages=[{"role": "user", "content": reason}],
            ),
        )
        return {"status": "success"}, self.build_initial_node()


class RouterAgent(LLMAgent):
    """Routes the user to the reservation agent or answers general questions."""

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a friendly assistant for La Maison restaurant. You can help "
                    "with general questions about the restaurant. When the user wants to "
                    "make a reservation, call the transfer_to_agent tool with agent "
                    "'reservation'. If the user says goodbye, call the end_conversation "
                    "tool. Do not mention transferring — just do it seamlessly. "
                    "Keep responses brief — this is a voice conversation."
                ),
            ),
        )

    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(self, params: FunctionCallParams, agent: str, reason: str):
        """Transfer the user to another agent.

        Args:
            agent (str): The agent to transfer to (e.g. 'reservation').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await self.handoff_to(
            agent,
            args=LLMAgentActivationArgs(
                messages=[{"role": "user", "content": reason}],
            ),
            result_callback=params.result_callback,
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


class RestaurantAgent(BaseAgent):
    """Owns the transport pipeline and routes frames to/from the bus."""

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    async def on_agent_ready(self, data: AgentReadyData) -> None:
        await super().on_agent_ready(data)

        if data.agent_name != "router":
            return

        await self.activate_agent(
            "router",
            args=LLMAgentActivationArgs(
                messages=[
                    {
                        "role": "user",
                        "content": "Greet the user and ask how you can help.",
                    }
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
        self._context_aggregator = LLMContextAggregatorPair(
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
            router = RouterAgent("router", bus=self.bus, bridged=())
            reservation = ReservationAgent(
                "reservation",
                bus=self.bus,
                context_aggregator=self._context_aggregator,
                reservation_system=MockReservationSystem(),
            )
            for agent in [router, reservation]:
                await self.add_agent(agent)

        @self._transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await self.cancel()

        return Pipeline(
            [
                self._transport.input(),
                stt,
                self._context_aggregator.user(),
                bridge,
                tts,
                self._transport.output(),
                self._context_aggregator.assistant(),
            ]
        )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    runner = AgentRunner(handle_sigint=runner_args.handle_sigint)

    restaurant = RestaurantAgent("restaurant", bus=runner.bus, transport=transport)
    await runner.add_agent(restaurant)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

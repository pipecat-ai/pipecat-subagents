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
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame
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
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowResult,
    FlowsFunctionSchema,
    NodeConfig,
)

from pipecat_agents.agents import BaseAgent, FlowsContextAgent, LLMContextAgent
from pipecat_agents.bus import (
    AgentActivationArgs,
    AgentBus,
    BusInputProcessor,
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


class ReservationAgent(FlowsContextAgent):
    """Structured reservation flow using Pipecat Flows."""

    def __init__(self, name: str, *, bus: AgentBus, reservation_system, **kwargs):
        self._reservation_system = reservation_system
        self._party_size_schema = FlowsFunctionSchema(
            name="collect_party_size",
            description="Record the number of people in the party.",
            properties={"size": {"type": "integer", "minimum": 1, "maximum": 12}},
            required=["size"],
            handler=self._handle_collect_party_size,
        )
        self._availability_schema = FlowsFunctionSchema(
            name="check_availability",
            description="Check availability for the requested time.",
            properties={
                "time": {
                    "type": "string",
                    "description": "Reservation time (e.g., '6:00 PM')",
                },
                "party_size": {"type": "integer"},
            },
            required=["time", "party_size"],
            handler=self._handle_check_availability,
        )
        self._end_reservation_schema = FlowsFunctionSchema(
            name="end_reservation",
            description="Confirm and end the reservation.",
            properties={},
            required=[],
            handler=self._handle_end_reservation,
        )
        self._transfer_to_agent_schema = FlowsFunctionSchema(
            name="transfer_to_agent",
            description="Transfer the user to another agent.",
            properties={
                "agent": {
                    "type": "string",
                    "description": "The agent to transfer to (e.g. 'router').",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the user is being transferred.",
                },
            },
            required=["agent", "reason"],
            handler=self._handle_transfer_to_agent,
        )

        super().__init__(
            name,
            bus=bus,
            global_functions=[self._transfer_to_agent_schema],
            **kwargs,
        )

    def build_llm(self) -> LLMService:
        return OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    def build_initial_node(self) -> NodeConfig:
        return {
            "name": "initial",
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a reservation assistant for La Maison, an upscale French "
                        "restaurant. Be casual and friendly. This is a voice conversation."
                    ),
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Ask how many people are in their party.",
                }
            ],
            "functions": [self._party_size_schema],
        }

    def _create_time_selection_node(self) -> NodeConfig:
        return {
            "name": "get_time",
            "task_messages": [
                {
                    "role": "system",
                    "content": "Ask what time they'd like to dine. The restaurant is open 5 PM to 10 PM.",
                }
            ],
            "functions": [self._availability_schema],
        }

    def _create_confirmation_node(self) -> NodeConfig:
        return {
            "name": "confirm",
            "task_messages": [
                {
                    "role": "system",
                    "content": "Confirm the reservation details and ask if there's anything else.",
                }
            ],
            "functions": [self._end_reservation_schema],
        }

    async def _handle_collect_party_size(self, args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
        size = args["size"]
        return {"size": size, "status": "success"}, self._create_time_selection_node()

    async def _handle_check_availability(self, args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
        time = args["time"]
        party_size = args["party_size"]
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
                        "role": "system",
                        "content": (
                            f"Apologize that {time} is not available. "
                            f"Suggest these alternative times: {times_list}."
                        ),
                    }
                ],
                "functions": [self._availability_schema, self._end_reservation_schema],
            }

    async def _handle_end_reservation(self, args: FlowArgs) -> tuple[None, NodeConfig]:
        return None, {
            "name": "end",
            "task_messages": [
                {
                    "role": "system",
                    "content": "Thank them for their reservation and say goodbye.",
                }
            ],
            "post_actions": [{"type": "end_conversation"}],
        }

    async def _handle_transfer_to_agent(
        self, args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[FlowResult, NodeConfig]:
        agent = args["agent"]
        reason = args.get("reason", "")
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await self.transfer_to(
            agent,
            args=AgentActivationArgs(
                messages=[{"role": "system", "content": reason}],
            ),
        )
        return {"status": "success"}, self.build_initial_node()


class RouterAgent(LLMContextAgent):
    """Routes the user to the reservation agent or answers general questions."""

    def __init__(self, name: str, **kwargs):
        super().__init__(
            name,
            system_messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly assistant for La Maison restaurant. You can help "
                        "with general questions about the restaurant. When the user wants to "
                        "make a reservation, call the transfer_to_agent tool with agent "
                        "'reservation'. If the user says goodbye, call the end_conversation "
                        "tool. Do not mention transferring — just do it seamlessly. "
                        "Keep responses brief — this is a voice conversation."
                    ),
                }
            ],
            **kwargs,
        )

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
                        "description": "The agent to transfer to (e.g. 'reservation').",
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
                properties={
                    "reason": {
                        "type": "string",
                        "description": "Why the conversation is ending.",
                    },
                },
                required=["reason"],
            ),
        ]

    async def _handle_transfer(self, params):
        agent = params.arguments["agent"]
        reason = params.arguments["reason"]
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await self.transfer_to(
            agent,
            args=AgentActivationArgs(
                messages=[{"role": "system", "content": reason}],
            ),
            result_callback=params.result_callback,
        )

    async def _handle_end(self, params):
        reason = params.arguments["reason"]
        logger.info(f"Agent '{self.name}': ending conversation ({reason})")
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": reason}], run_llm=True)
        )
        await self.end(reason=reason, result_callback=params.result_callback)


class RestaurantAgent(BaseAgent):
    """Owns the transport pipeline and routes frames to/from the bus."""

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus, active=True)
        self._transport = transport

    async def build_pipeline_task(self) -> PipelineTask:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
        )

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        bus_output = BusOutputProcessor(
            bus=self.bus,
            agent_name=self.name,
            name=f"{self.name}::BusOutput",
            output_frames=(LLMContextFrame,),
            pass_through=True,
        )
        bus_input = BusInputProcessor(
            bus=self.bus,
            agent_name=self.name,
            name=f"{self.name}::BusInput",
        )

        # Create sub-agents
        router = RouterAgent("router", bus=self.bus, parent=self.name)
        reservation = ReservationAgent(
            "reservation",
            bus=self.bus,
            parent=self.name,
            reservation_system=MockReservationSystem(),
        )
        for agent in [router, reservation]:
            await self.add_agent(agent)

        # Wire transport events
        @self._transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await self.activate_agent(
                "router",
                args=AgentActivationArgs(
                    messages=[
                        {
                            "role": "system",
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
                bus_output,
                bus_input,
                tts,
                self._transport.output(),
                context_aggregator.assistant(),
            ]
        )

        return PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    runner = AgentRunner(handle_sigint=runner_args.handle_sigint)

    main = RestaurantAgent("main", bus=runner.bus, transport=transport)
    await runner.add_agent(main)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

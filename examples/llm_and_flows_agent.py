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
from pipecat_agents.agents import BaseAgent, FlowsAgent, LLMAgent
from pipecat_agents.bus import (
    AgentActivationArgs,
    AgentBus,
    BusBridgeProcessor,
    BusEndAgentMessage,
    BusEndMessage,
    BusMessage,
)
from pipecat_agents.runner import AgentRunner
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowResult,
    FlowsFunctionSchema,
    NodeConfig,
)

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


# ── Mock reservation system ─────────────────────────────────────


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


reservation_system = MockReservationSystem()


# ── Flows agent: Reservation ────────────────────────────────────


# Flow function handlers
async def collect_party_size(args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
    size = args["size"]
    return {"size": size, "status": "ok"}, create_time_selection_node()


async def check_availability(args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
    time = args["time"]
    party_size = args["party_size"]
    is_available, alternatives = await reservation_system.check_availability(party_size, time)

    if is_available:
        return {"time": time, "available": True}, create_confirmation_node()
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
            "functions": [availability_schema, end_reservation_schema],
        }


async def end_reservation(args: FlowArgs) -> tuple[None, NodeConfig]:
    return None, {
        "name": "end",
        "task_messages": [
            {"role": "system", "content": "Thank them for their reservation and say goodbye."}
        ],
        "post_actions": [{"type": "end_conversation"}],
    }


# Flow function schemas
party_size_schema = FlowsFunctionSchema(
    name="collect_party_size",
    description="Record the number of people in the party.",
    properties={"size": {"type": "integer", "minimum": 1, "maximum": 12}},
    required=["size"],
    handler=collect_party_size,
)

availability_schema = FlowsFunctionSchema(
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
    handler=check_availability,
)

end_reservation_schema = FlowsFunctionSchema(
    name="end_reservation",
    description="Confirm and end the reservation.",
    properties={},
    required=[],
    handler=end_reservation,
)


# Flow node creators
def create_initial_node() -> NodeConfig:
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
        "functions": [party_size_schema],
    }


def create_time_selection_node() -> NodeConfig:
    return {
        "name": "get_time",
        "task_messages": [
            {
                "role": "system",
                "content": "Ask what time they'd like to dine. The restaurant is open 5 PM to 10 PM.",
            }
        ],
        "functions": [availability_schema],
    }


def create_confirmation_node() -> NodeConfig:
    return {
        "name": "confirm",
        "task_messages": [
            {
                "role": "system",
                "content": "Confirm the reservation details and ask if there's anything else.",
            }
        ],
        "functions": [end_reservation_schema],
    }


# Transfer function (global, available at every node)
transfer_to_router_schema = FlowsFunctionSchema(
    name="transfer_to_router",
    description="Transfer back to the main assistant if the user wants something else.",
    properties={
        "reason": {
            "type": "string",
            "description": "Why the user is being transferred.",
        },
    },
    required=["reason"],
    handler=None,  # Set in ReservationAgent.__init__
)


class ReservationAgent(FlowsAgent):
    """Structured reservation flow using Pipecat Flows."""

    def __init__(self, name: str, *, bus: AgentBus, context_aggregator, **kwargs):
        # Set the transfer handler before super().__init__ registers it
        transfer_to_router_schema.handler = self._handle_transfer_to_router

        super().__init__(
            name,
            bus=bus,
            context_aggregator=context_aggregator,
            global_functions=[transfer_to_router_schema],
            **kwargs,
        )

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        )

    def build_initial_node(self) -> NodeConfig:
        return create_initial_node()

    async def _handle_transfer_to_router(
        self, args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[FlowResult, NodeConfig]:
        reason = args.get("reason", "")
        logger.info(f"Reservation: transferring to router ({reason})")
        await self.transfer_to(
            "router",
            args=AgentActivationArgs(
                messages=[{"role": "user", "content": f"The user is back: {reason}"}],
            ),
        )
        # Return current node (won't be used since we're transferring)
        return {"status": "transferring"}, create_initial_node()


# ── LLM Agent: Router ───────────────────────────────────────────


class RouterAgent(LLMAgent):
    """Routes the user to the reservation agent or answers general questions."""

    def build_llm(self) -> LLMService:
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            system_instruction=(
                "You are a friendly assistant for La Maison restaurant. You can help "
                "with general questions about the restaurant. When the user wants to "
                "make a reservation, transfer them to the reservation system. "
                "Keep responses brief — this is a voice conversation."
            ),
        )
        llm.register_function(
            "transfer_to_reservation",
            self._handle_transfer_to_reservation,
            cancel_on_interruption=False,
        )
        llm.register_function("end_conversation", self._handle_end)
        return llm

    def build_tools(self):
        return [
            FunctionSchema(
                name="transfer_to_reservation",
                description="Transfer to the reservation system when the user wants to book a table.",
                properties={},
                required=[],
            ),
            FunctionSchema(
                name="end_conversation",
                description="End the conversation when the user says goodbye.",
                properties={},
                required=[],
            ),
        ]

    async def _handle_transfer_to_reservation(self, params):
        logger.info("Router: transferring to reservation")
        await self.transfer_to(
            "reservation",
            args=AgentActivationArgs(
                messages=[
                    {
                        "role": "user",
                        "content": "I'd like to make a reservation.",
                    }
                ],
            ),
            result_callback=params.result_callback,
        )

    async def _handle_end(self, params):
        logger.info("Router: ending conversation")
        await self.end(
            reason="User said goodbye",
            result="Say goodbye briefly.",
            result_callback=params.result_callback,
        )


# ── Main Agent ───────────────────────────────────────────────────


class MainAgent(BaseAgent):
    """Owns the transport pipeline and bridges frames to/from the bus."""

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
        router = RouterAgent("router", bus=self.bus, parent=self.name)
        reservation = ReservationAgent(
            "reservation",
            bus=self.bus,
            parent=self.name,
            context_aggregator=context_aggregator,
        )
        for agent in [router, reservation]:
            self._sub_agents.append(agent.name)
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

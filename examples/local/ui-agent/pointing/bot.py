#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pointing — the agent acts on the page to direct user attention.

Composes ``ReplyToolMixin`` from the SDK, which exposes one bundled
LLM tool: ``reply(answer, scroll_to=None, highlight=None)``. One tool
call per turn; the required ``answer`` argument is enforced by the
API schema so the model cannot forget the spoken reply.

When the user asks "where's the iPhone 17?", the LLM finds the
matching ref in the snapshot and emits one ``reply`` call with
``answer="Here's the iPhone 17."`` plus ``scroll_to`` and
``highlight`` set to that ref. The mixin dispatches the UI commands
and completes the in-flight task.

Architecture::

    PointingRoot (BaseAgent, root)              -- transport + UI bridge
      ├── VoiceAgent (LLMAgent, bridged)        -- conversational layer
      │     └── @tool answer_about_screen(query)
      │           └── self.task("ui", payload={"query": query})
      └── PointingAgent (ReplyToolMixin + UIAgent)
            └── inherited: reply(answer, scroll_to=None, highlight=None)

Run::

    uv run python bot.py

Then open the client at ``http://localhost:5173``.

Requirements:
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame
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

from pipecat_subagents.agents import (
    UI_STATE_PROMPT_GUIDE,
    BaseAgent,
    LLMAgent,
    LLMAgentActivationArgs,
    ReplyToolMixin,
    TaskError,
    UIAgent,
    agent_ready,
    attach_ui_bridge,
    tool,
)
from pipecat_subagents.bus import AgentBus, BusBridgeProcessor
from pipecat_subagents.bus.messages import BusTaskRequestMessage
from pipecat_subagents.runner import AgentRunner
from pipecat_subagents.types import AgentReadyData

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


VOICE_PROMPT = """\
You are the voice layer of a screen-aware assistant. A separate UI \
layer sees the page and writes the spoken reply.

For every user utterance that could involve the page, call \
``answer_about_screen`` with the user's request verbatim. The tool's \
response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


UI_PROMPT = (
    """\
You help the user find and look at items on a long page of phone \
listings. The current ``<ui_state>`` block is in your context.

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn, no \
chaining.

``reply(answer, scroll_to=None, highlight=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, one \
short sentence. No markdown, no symbols, no specs read aloud.
- ``scroll_to`` (OPTIONAL): a snapshot ref like ``"e5"``. Set this \
when the item the user wants is tagged ``[offscreen]`` in \
``<ui_state>``.
- ``highlight`` (OPTIONAL): a snapshot ref like ``"e5"``. Set this \
when pointing at a specific item the user named.

## Decision rules

- Pointing at a **visible** item → set ``highlight=ref``; leave \
``scroll_to`` unset.
- Pointing at an **offscreen** item → set both ``scroll_to=ref`` and \
``highlight=ref`` (same ref).
- **Descriptive / conversational** question → leave both unset.

## Examples

- "Where's the iPhone 17?" (offscreen) → \
``reply(answer="Here's the iPhone 17.", scroll_to="e5", highlight="e5")``
- "Where's the iPhone 17 Pro?" (offscreen) → \
``reply(answer="Here's the iPhone 17 Pro.", scroll_to="e8", highlight="e8")``
- "Which one is the Nothing phone?" (visible) → \
``reply(answer="This one, the Nothing Phone 3.", highlight="e29")``
- "Which phones are from Google?" → \
``reply(answer="The Pixel 9, Pixel 9 Pro, and Pixel 9a are from Google.")``
- "Tell me about the iPhone 17 Pro" → \
``reply(answer="It's Apple's 2025 flagship with a 120Hz ProMotion display and periscope zoom.")``"""
    + UI_STATE_PROMPT_GUIDE
)


class VoiceAgent(LLMAgent):
    """Conversational layer. Delegates page-relevant utterances to the UI agent."""

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(name, bus=bus, bridged=())

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                model=os.getenv("OPENAI_MODEL"),
                system_instruction=VOICE_PROMPT,
            ),
        )

    @tool(cancel_on_interruption=False)
    async def answer_about_screen(self, params: FunctionCallParams, query: str):
        """Forward the user's request to the UI layer.

        Args:
            query: The user's request, passed verbatim.
        """
        logger.info(f"{self}: answer_about_screen('{query}')")
        try:
            async with self.task("ui", payload={"query": query}, timeout=10) as t:
                pass
        except TaskError as e:
            logger.warning(f"{self}: ui task failed: {e}")
            await params.result_callback("Something went wrong on my side.")
            return

        speak = (t.response or {}).get("speak")
        if not speak:
            await params.result_callback(None)
            return

        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "assistant", "content": speak}],
                run_llm=False,
            )
        )
        await self.queue_frame(TTSSpeakFrame(text=speak))
        await params.result_callback(None)


class PointingAgent(ReplyToolMixin, UIAgent):
    """UIAgent that points at items using the canonical ``reply`` tool.

    Composes the SDK's ``ReplyToolMixin``, which exposes a single
    ``reply(answer, scroll_to=None, highlight=None)`` LLM tool. One
    tool call per turn; the required ``answer`` argument is enforced
    by the API schema so the model cannot forget the spoken reply
    (the failure mode chainable tools have with smaller models).
    """

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                model=os.getenv("OPENAI_MODEL"),
                system_instruction=UI_PROMPT,
            ),
        )

    async def on_task_request(self, message: BusTaskRequestMessage) -> None:
        # super() records the in-flight task and auto-injects
        # <ui_state>; we feed the user's query in afterward.
        await super().on_task_request(message)
        query = (message.payload or {}).get("query", "")
        logger.info(f"{self}: task query '{query}'")
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": query}],
                run_llm=True,
            )
        )


class PointingRoot(BaseAgent):
    """Root agent. Owns the transport and bridges to the voice layer."""

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    async def on_ready(self) -> None:
        await super().on_ready()
        attach_ui_bridge(self, target="ui")

    @agent_ready(name="voice")
    async def on_voice_ready(self, data: AgentReadyData) -> None:
        await self.activate_agent(
            "voice",
            args=LLMAgentActivationArgs(
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "Greet the user briefly. Tell them they can ask "
                            "to find or scroll to any phone on the list. One "
                            "short sentence."
                        ),
                    }
                ],
            ),
        )

    def build_pipeline_task(self, pipeline: Pipeline) -> PipelineTask:
        task = PipelineTask(pipeline, enable_rtvi=True)

        # Register on_client_ready BEFORE the pipeline starts; the
        # event fires when the client's handshake message arrives,
        # which can land before on_ready runs. See attach_ui_bridge
        # docstring.
        @task.rtvi.event_handler("on_client_ready")
        async def _on_client_ready(_rtvi):
            logger.info("Client ready")
            await self.add_agent(VoiceAgent("voice", bus=self.bus))
            await self.add_agent(PointingAgent("ui", bus=self.bus))

        return task

    async def build_pipeline(self) -> Pipeline:
        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
        tts = CartesiaTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            settings=CartesiaTTSSettings(
                voice=os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
            ),
        )

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        bridge = BusBridgeProcessor(
            bus=self.bus, agent_name=self.name, name=f"{self.name}::BusBridge"
        )

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
    root = PointingRoot("main", bus=runner.bus, transport=transport)
    await runner.add_agent(root)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

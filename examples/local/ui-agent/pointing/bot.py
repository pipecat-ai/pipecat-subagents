#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pointing — the agent acts on the page to direct user attention.

Same canonical UIAgent setup as ``hello-snapshot``, but the UI agent
composes three SDK mixins: ``ScrollToToolMixin``,
``HighlightToolMixin``, and ``AnswerToolMixin``. The LLM gets three
tools: ``scroll_to(ref)`` and ``highlight(ref)`` (pure side effects)
plus ``answer(text)`` (the terminator that completes the task and
hands the spoken reply to TTS). When the user asks "where's the
iPhone 17?", the LLM finds the matching ref in the snapshot, scrolls
it into view if it's tagged ``[offscreen]``, flashes it, and speaks a
brief confirmation, all in one turn via tool chaining.

Architecture::

    PointingRoot (BaseAgent, root)              -- transport + UI bridge
      ├── VoiceAgent (LLMAgent, bridged)        -- conversational layer
      │     └── @tool answer_about_screen(query)
      │           └── self.task("ui", payload={"query": query})
      └── PointingAgent
            (ScrollToToolMixin + HighlightToolMixin + AnswerToolMixin + UIAgent)
            └── inherited: scroll_to(ref), highlight(ref), answer(text)

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
    AnswerToolMixin,
    BaseAgent,
    HighlightToolMixin,
    LLMAgent,
    LLMAgentActivationArgs,
    ScrollToToolMixin,
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

## Tools

- ``scroll_to(ref)``: scrolls the named item into view. Side effect \
only; doesn't speak.
- ``highlight(ref)``: pulses the named item visually. Side effect \
only; doesn't speak.
- ``answer(text)``: speaks ``text`` to the user and ends the turn. \
Always called exactly once per turn, last.

You can chain multiple tools in one turn. The pattern: do whatever \
visual action the request needs, then ``answer`` with a short \
spoken confirmation.

## When pointing at a specific item

Decide based on the snapshot's ``[offscreen]`` state:

- Item is **visible** (no ``[offscreen]`` tag) → ``highlight(ref)`` \
+ ``answer("text")``. Two tool calls.
- Item is **offscreen** (``[offscreen]`` tag) → ``scroll_to(ref)`` \
+ ``highlight(ref)`` + ``answer("text")``. Three tool calls.

Examples:

- "Where's the iPhone 17?" → check snapshot for the iPhone 17's \
ref and offscreen state, then call the right combination ending in \
``answer("Here's the iPhone 17.")``.
- "Which one is the Nothing phone?" → almost always already \
visible (the user is asking about something they can see), so \
``highlight`` + ``answer("This one — the Nothing Phone 3.")``.

## When not pointing at anything

For descriptive questions ("which phones are from Google?", "what's \
the cheapest one?"), just call ``answer(text)`` with the spoken \
reply. No scroll, no highlight.

## Voice rules

Plain language. One short spoken sentence. No markdown, no lists, \
no symbols. Don't read out specs.

"""
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
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
            async with self.task("ui", payload={"query": query}, timeout=30) as t:
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


class PointingAgent(
    ScrollToToolMixin,
    HighlightToolMixin,
    AnswerToolMixin,
    UIAgent,
):
    """UIAgent that points at items by chaining SDK action mixins.

    Composes the three shipped mixins as-is. The action mixins
    (``ScrollToToolMixin``, ``HighlightToolMixin``) are pure side
    effects: each dispatches one UI command and returns without
    completing the in-flight task. ``AnswerToolMixin`` provides the
    terminator: ``answer(text)`` calls ``respond_to_task(speak=text)``
    which closes the task and hands the spoken reply to TTS.

    With this composition the LLM can chain ``scroll_to`` (if
    offscreen) → ``highlight`` → ``answer(text)`` in a single turn,
    or skip straight to ``highlight + answer`` when the target is
    already visible. The prompt below teaches the model which combo
    to pick based on the snapshot's ``[offscreen]`` state tag.
    """

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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

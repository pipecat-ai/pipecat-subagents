#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hello UIAgent — the smallest possible accessibility-snapshot demo.

Three agents wired in the canonical UIAgent pattern. The user speaks;
the conversational layer delegates every utterance to a UIAgent that
sees the page and writes the spoken answer.

Architecture::

    HelloRoot (BaseAgent, root)         -- transport + UI bridge
      ├── VoiceAgent (LLMAgent, bridged) -- conversational layer
      │     └── @tool answer_about_screen(query)
      │           └── self.task("hello", payload={query})
      └── HelloAgent (UIAgent, not bridged) -- snapshot-aware layer
            └── @tool answer(text)

The voice agent has one tool. It forwards every user utterance to the
UI agent and speaks the result verbatim. The UI agent's
``on_task_request`` fires, which auto-injects the latest ``<ui_state>``
block into its LLM context. The UI agent's LLM picks the ``answer``
tool with a spoken reply grounded in what's on screen.

Why two LLMs for "hello world": this is the pattern UIAgent's
auto-inject is built for. A single bridged UIAgent works for delegated
tasks but doesn't auto-inject for a direct voice loop. Setting up
both layers up front means later examples (pointing, form-fill,
deixis, async-tasks) compose new tools onto the same skeleton without
restructuring.

Run::

    uv run python bot.py

Then open the client at ``http://localhost:5173`` (see
``client/README.md``).

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
layer sees the page the user is looking at and writes the spoken \
reply for any question that could plausibly involve the page.

## Routing rule
For every user utterance that could involve the page in any way — \
"what's on screen", "what does this say", "is X on the page", \
factual questions, navigational questions, anything where the page \
content might matter — call ``answer_about_screen`` with the user's \
request verbatim. The tool's response is the spoken reply, already \
TTS-ready; pass it through without paraphrasing.

If the request has nothing to do with the page, still call the \
tool — the UI layer falls back to general knowledge.

## When to answer directly
Only respond directly for pure pleasantries that don't need any \
content awareness:

- Greetings ("hi", "hello").
- Acknowledgements ("thanks", "got it").
- Goodbyes ("bye", "see you").

Keep direct replies to one short spoken sentence. No markdown, no \
lists, no symbols."""


HELLO_PROMPT = (
    """\
You answer the user's question grounded in the page they're looking \
at. The current ``<ui_state>`` block is in your context — use it for \
anything the user could be asking about on screen.

Always call exactly one tool: ``answer(text)``. Put the spoken reply \
in ``text``. Plain language, one or two short sentences, no markdown \
or symbols.

When the question is about something on the page, ground claims in \
the ``<ui_state>`` content. When it's general knowledge with no \
on-page referent (history, geography, definitions), answer from your \
own knowledge. Don't tell the user what you can't see — just answer \
or admit you don't know."""
    + UI_STATE_PROMPT_GUIDE
)


class VoiceAgent(LLMAgent):
    """Conversational layer. Delegates every utterance to ``HelloAgent``."""

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
            async with self.task("hello", payload={"query": query}, timeout=30) as t:
                pass
        except TaskError as e:
            logger.warning(f"{self}: hello task failed: {e}")
            await params.result_callback("Something went wrong on my side.")
            return

        speak = (t.response or {}).get("speak")
        if not speak:
            await params.result_callback(None)
            return

        # Feed the verbatim spoken reply through TTS without re-running
        # the voice LLM. Append it to context so subsequent turns stay
        # coherent.
        await self.queue_frame(TTSSpeakFrame(text=speak, append_to_context=True))
        await params.result_callback(None)


class HelloAgent(UIAgent):
    """Snapshot-aware layer. Answers grounded in ``<ui_state>``.

    ``UIAgent`` defaults to ``active=True`` (unlike ``LLMAgent``)
    because the canonical UIAgent role is an always-on delegate. No
    explicit ``__init__`` override needed.
    """

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                model=os.getenv("OPENAI_MODEL"),
                system_instruction=HELLO_PROMPT,
            ),
        )

    async def on_task_request(self, message: BusTaskRequestMessage) -> None:
        # super() records the in-flight task on ``current_task`` and
        # auto-injects ``<ui_state>``. After that, append the user's
        # query as a user message and trigger the LLM. Without
        # this step the snapshot lands in context but the question
        # never does — the LLM has nothing to answer.
        await super().on_task_request(message)
        query = (message.payload or {}).get("query", "")
        logger.info(f"{self}: task query '{query}'")
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": query}],
                run_llm=True,
            )
        )

    @tool
    async def answer(self, params: FunctionCallParams, text: str):
        """Speak ``text`` back to the user.

        Args:
            text: The spoken reply in plain language. One or two short
                sentences. No markdown, no symbols, no lists.
        """
        logger.info(f"{self}: answer('{text[:80]}…')")
        await self.respond_to_task(speak=text)
        await params.result_callback(None)


class HelloRoot(BaseAgent):
    """Root agent. Owns the transport and bridges to the voice layer."""

    def __init__(self, name: str, *, bus: AgentBus, transport: BaseTransport):
        super().__init__(name, bus=bus)
        self._transport = transport

    async def on_ready(self) -> None:
        await super().on_ready()
        # Route inbound UI events (incl. the reserved snapshot event)
        # at HelloAgent — the snapshot is what HelloAgent reasons over.
        attach_ui_bridge(self, target="hello")

    @agent_ready(name="voice")
    async def on_voice_ready(self, data: AgentReadyData) -> None:
        # First turn is a pure pleasantry, so the voice agent answers
        # directly per its prompt — no delegation needed.
        await self.activate_agent(
            "voice",
            args=LLMAgentActivationArgs(
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "Greet the user briefly. Tell them they can ask "
                            "about anything on this page. One short sentence."
                        ),
                    }
                ],
            ),
        )

    def build_pipeline_task(self, pipeline: Pipeline) -> PipelineTask:
        task = PipelineTask(pipeline, enable_rtvi=True)

        # Register the client-ready handler here, BEFORE the pipeline
        # starts running, so we don't miss the event in the race
        # between WebRTC connect and our agent's on_ready hook.
        # Client-ready is emitted by the RTVI processor as soon as the
        # client's handshake message arrives — that can land before
        # on_ready fires.
        @task.rtvi.event_handler("on_client_ready")
        async def _on_client_ready(_rtvi):
            logger.info("Client ready")
            await self.add_agent(VoiceAgent("voice", bus=self.bus))
            await self.add_agent(HelloAgent("hello", bus=self.bus))

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
    root = HelloRoot("main", bus=runner.bus, transport=transport)
    await runner.add_agent(root)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

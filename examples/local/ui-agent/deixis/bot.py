#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deixis — the agent grounds in what the user just selected.

The page renders an article. The user selects a paragraph (or any
span of text) and asks "explain this", "rephrase that", "where does
it talk about RNA editing?", and so on. The walker captures
``window.getSelection()`` and emits a ``<selection ref="...">selected
text</selection>`` block in the snapshot. The UI agent reads it as a
deictic reference: "this paragraph" resolves to the selected element.

Two directions:

- **Read**: user selects text → ``<selection>`` block in
  ``<ui_state>`` → agent grounds its answer in the selected content.
- **Write**: agent says "this paragraph" → ``select_text=ref`` puts
  the OS-level selection on that element → user sees what the agent
  is referring to.

Same canonical setup as pointing. The interesting bit is that
``DeixisAgent`` does NOT compose ``ReplyToolMixin``: that mixin's
``reply(answer, scroll_to, highlight)`` doesn't have a ``select_text``
field, and we want the LLM to be able to point at content via
selection. So we hand-roll a custom ``@tool reply`` with the extra
field. This is the canonical extension story for the SDK.

Architecture::

    DeixisRoot (BaseAgent, root)              -- transport + UI bridge
      ├── VoiceAgent (LLMAgent, bridged)      -- conversational layer
      │     └── @tool answer_about_screen(query)
      │           └── self.task("ui", payload={"query": query})
      └── DeixisAgent (UIAgent)
            └── @tool reply(answer, scroll_to, highlight, select_text)

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
You are the voice layer of a screen-aware reading assistant. A \
separate UI layer sees the page (and the user's selection) and \
writes the spoken reply.

For every user utterance about the article, call \
``answer_about_screen`` with the user's request verbatim. The tool's \
response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


UI_PROMPT = (
    """\
You help the user read and understand an article. The current \
``<ui_state>`` block is in your context, and may contain a \
``<selection>`` block when the user has highlighted text.

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn, no \
chaining.

``reply(answer, scroll_to=None, highlight=None, select_text=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, two short \
sentences max. No markdown, no symbols, no quoting long passages.
- ``scroll_to`` (OPTIONAL): a snapshot ref. Set when the paragraph \
you want to point at is tagged ``[offscreen]``.
- ``highlight`` (OPTIONAL): a list of snapshot refs to flash briefly. \
Use for short emphasis: "look at this fact". Don't use it for a \
whole paragraph; ``select_text`` is better for that.
- ``select_text`` (OPTIONAL): a single snapshot ref. Sets the page's \
text selection to that element. Use this when you say "this \
paragraph" or "the section that talks about X" so the user sees \
exactly what you're referring to.

## Reading the user's selection

If ``<ui_state>`` contains a ``<selection ref="...">selected \
text</selection>`` block, the user has highlighted something. Treat \
that selection as the deictic referent for words like "this", \
"that", "this paragraph", "what I selected". Ground your answer in \
the selected content, not the article as a whole.

When answering about the user's selection, do NOT also call \
``select_text`` — they already selected it; pointing back at the \
same span is redundant.

## Decision rules

- User has a selection AND asks something deictic ("explain this", \
"rephrase that", "what does this mean") → ground in the selection. \
Just ``answer``; no visual fields.
- User asks "where does it say X?" or "show me the part about X" → \
find the matching paragraph, ``answer`` briefly, set \
``select_text=ref`` to point at it, and ``scroll_to=ref`` if it's \
``[offscreen]``.
- User asks a content question without selection → ``answer`` with \
the relevant fact. Optionally set ``select_text=ref`` if the \
answer is sourced from one specific paragraph.

## Examples

(refs are illustrative; use the actual refs from the current \
``<ui_state>``)

- User selects the third paragraph, asks "explain this" → \
``reply(answer="The skin acts as its own light sensor. Even though \
octopuses are colorblind, their skin can detect light directly, \
which is how they match colors so accurately.")``
- "Where does it talk about RNA editing?" (paragraph e15, offscreen) \
→ ``reply(answer="Here, in the paragraph about RNA editing.", \
scroll_to="e15", select_text="e15")``
- "How many neurons does an octopus have?" (no selection) → \
``reply(answer="About five hundred million, with two thirds of \
them in the arms.", select_text="e7")``
- "Hi, what's this article about?" (no selection) → \
``reply(answer="It's a short essay on octopus cognition. Select any \
paragraph and I'll explain it.")``

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

        await self.queue_frame(TTSSpeakFrame(text=speak, append_to_context=True))
        await params.result_callback(None)


class DeixisAgent(UIAgent):
    """UIAgent with a custom ``reply`` tool that exposes ``select_text``.

    The SDK ships ``ReplyToolMixin`` for the canonical
    ``reply(answer, scroll_to, highlight)`` shape. Deixis needs a
    fourth field — ``select_text`` — so the agent can put the page's
    text selection on a paragraph it's referring to. Rather than
    forcing the SDK mixin to grow, we hand-roll a ``@tool reply``
    with the field we need. This is the SDK's canonical extension
    story: when the bundled mixin doesn't fit, write your own using
    the ``UIAgent`` helper methods (``scroll_to``, ``highlight``)
    plus ``send_command`` for new commands.
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
        # super() resets context (keep_history=False), then auto-injects
        # <ui_state>; we feed the user's query in afterward.
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
    async def reply(
        self,
        params: FunctionCallParams,
        answer: str,
        scroll_to: str | None = None,
        highlight: list[str] | None = None,
        select_text: str | None = None,
    ):
        """Reply to the user. Optionally point at content visually.

        Always called exactly once per turn. ``answer`` is required;
        the visual fields are optional and may be combined.

        Args:
            answer: The spoken reply in plain language. One or two
                short sentences. No markdown, no symbols.
            scroll_to: Optional snapshot ref. When set, scrolls the
                element into view before speaking.
            highlight: Optional list of snapshot refs to flash
                briefly. Best for short emphasis, not whole
                paragraphs.
            select_text: Optional snapshot ref. Sets the page's text
                selection to that element. Best for "this paragraph"
                / "the section that talks about X" so the user sees
                exactly what was meant.
        """
        preview = (answer or "").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        logger.info(
            f"{self}: reply(answer={preview!r}, scroll_to={scroll_to!r}, "
            f"highlight={highlight!r}, select_text={select_text!r})"
        )
        if scroll_to:
            await self.scroll_to(scroll_to)
        if highlight:
            for ref in highlight:
                await self.highlight(ref)
        if select_text:
            await self.select_text(select_text)
        await self.respond_to_task(speak=answer)
        await params.result_callback(None)


class DeixisRoot(BaseAgent):
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
                            "Greet the user briefly. Tell them they can "
                            "select a paragraph and ask you to explain or "
                            "rephrase it. One short sentence."
                        ),
                    }
                ],
            ),
        )

    def build_pipeline_task(self, pipeline: Pipeline) -> PipelineTask:
        task = PipelineTask(pipeline, enable_rtvi=True)

        @task.rtvi.event_handler("on_client_ready")
        async def _on_client_ready(_rtvi):
            logger.info("Client ready")
            await self.add_agent(VoiceAgent("voice", bus=self.bus))
            await self.add_agent(DeixisAgent("ui", bus=self.bus, keep_history=False))

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
    root = DeixisRoot("main", bus=runner.bus, transport=transport)
    await runner.add_agent(root)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async tasks — the agent fans out long-running work and streams progress.

The user asks the assistant to research a topic. The UI agent
dispatches three worker agents (Wikipedia, news, scholarly papers)
in parallel via ``user_task_group``. Each worker emits progress
updates while it works. The SDK forwards every lifecycle event to
the client as ``ui.task`` envelopes (``group_started``,
``task_update``, ``task_completed``, ``group_completed``) which the
client renders as in-flight cards with per-worker status. The user
can cancel any group mid-flight via ``ui.cancelTask(task_id)``,
which sends a reserved ``__cancel_task`` event that the SDK turns
into a ``cancel_task`` call on the registered group.

Architecture::

    AsyncTasksRoot (BaseAgent, root)         -- transport + UI bridge
      ├── VoiceAgent (LLMAgent, bridged)     -- conversational layer
      │     └── @tool answer_about_screen(query)
      │           └── self.task("ui", payload={"query": query})
      ├── ResearchAgent (UIAgent)
      │     └── @tool reply(answer, research_query=None)
      │           └── (if research_query) creates a background task
      │               that runs user_task_group(...)
      └── three workers (BaseAgent each)
            ├── WikipediaResearcher
            ├── NewsResearcher
            └── ScholarResearcher

Workers are deliberately simulated with ``asyncio.sleep`` and
canned summaries so the demo focuses on the protocol, not the AI.
A real app would wire each worker to its own data source.

The reply tool dispatches the task group on a background asyncio
task and returns immediately (with the spoken "researching X"
acknowledgement). This frees the voice agent to receive new user
turns while the workers continue. If the user changes their mind
mid-flight, they cancel via the UI button (or the agent could
emit a cancel command from a follow-up turn — left as an
exercise).

Run::

    uv run python bot.py

Then open the client at ``http://localhost:5173``.

Requirements:
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import asyncio
import os
import random

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
You are the voice layer of a research assistant. A separate UI \
layer sees the page and dispatches research tasks.

For every user utterance involving research (asking about a topic, \
launching a search, asking for follow-ups), call \
``answer_about_screen`` with the user's request verbatim. The \
tool's response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


UI_PROMPT = (
    """\
You help the user research topics. When the user names something \
to look up, kick off a parallel research task across three worker \
sources (Wikipedia, news, scholarly papers).

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn.

``reply(answer, research_query=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, one \
short sentence. No markdown, no symbols.
- ``research_query`` (OPTIONAL): the topic to research. When set, \
the server fans out three worker agents in parallel and streams \
their progress to an in-flight panel on the page. The workers run \
in the background; you do NOT wait for results. Just speak a brief \
acknowledgement.

## Decision rules

- **User asks to research / look up / find out about something** → \
set ``research_query`` to the topic and answer with a brief \
acknowledgement ("Researching the Mariana Trench now"). The server \
handles the rest; results stream onto the page.
- **User asks a quick question you can answer immediately** → just \
``answer``. Don't kick off a research task for trivia or for \
questions about the in-flight tasks themselves.
- **User asks about ongoing research** → just ``answer`` (the \
results panel on screen shows progress; you can describe what's \
visible if needed).

## Examples

- "Research the Mariana Trench." → \
``reply(answer="Researching the Mariana Trench now.", research_query="Mariana Trench")``
- "Look up octopus cognition." → \
``reply(answer="Looking that up.", research_query="octopus cognition")``
- "How many neurons does an octopus have?" (quick question, no \
research needed) → \
``reply(answer="About five hundred million.")``
- "Hi." → ``reply(answer="Hi! What would you like to research?")``

"""
    + UI_STATE_PROMPT_GUIDE
)


class _SimulatedResearcher(BaseAgent):
    """BaseAgent worker that fakes a research task with progress updates.

    Receives a ``payload={"query": ...}``. Emits two or three
    ``send_task_update`` messages with progress text, then a final
    ``send_task_response`` carrying a canned summary. ``asyncio.sleep``
    randomization makes the workers feel like they're running at
    different paces, which helps demonstrate the streaming UI.

    Subclasses set ``source_name`` (used in updates) and provide
    ``summarize(query)`` returning the canned summary text.
    """

    source_name: str = "researcher"

    def summarize(self, query: str) -> str:
        return f"Generic results for '{query}'."

    async def on_task_request(self, message: BusTaskRequestMessage) -> None:
        await super().on_task_request(message)
        task_id = message.task_id
        query = (message.payload or {}).get("query", "")
        try:
            await asyncio.sleep(random.uniform(0.4, 1.2))
            await self.send_task_update(task_id, {"text": f"searching {self.source_name}…"})

            await asyncio.sleep(random.uniform(0.6, 1.4))
            n = random.randint(3, 8)
            await self.send_task_update(task_id, {"text": f"found {n} results"})

            await asyncio.sleep(random.uniform(0.5, 1.5))
            await self.send_task_update(task_id, {"text": "summarizing"})

            await asyncio.sleep(random.uniform(0.4, 0.9))
            summary = self.summarize(query)
            await self.send_task_response(task_id, response={"summary": summary})
        except asyncio.CancelledError:
            # The base agent's cancellation hook auto-emits a
            # CANCELLED response; just bail.
            raise


class WikipediaResearcher(_SimulatedResearcher):
    source_name = "wikipedia"

    def summarize(self, query: str) -> str:
        return (
            f"Wikipedia overview of {query}: a one-paragraph summary covering "
            "the historical background, key facts, and major figures."
        )


class NewsResearcher(_SimulatedResearcher):
    source_name = "news"

    def summarize(self, query: str) -> str:
        return (
            f"Recent news on {query}: three headlines from the past month, "
            "a short context paragraph, and any active developments."
        )


class ScholarResearcher(_SimulatedResearcher):
    source_name = "scholar"

    def summarize(self, query: str) -> str:
        return (
            f"Scholarly take on {query}: two highly cited papers, the "
            "consensus position, and a notable debate or open question."
        )


class VoiceAgent(LLMAgent):
    """Conversational layer. Delegates research utterances to the UI agent."""

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


class ResearchAgent(UIAgent):
    """UIAgent that kicks off background research task groups.

    The custom ``@tool reply`` has a ``research_query`` field. When
    the LLM sets it, the tool spawns a background asyncio task that
    runs ``user_task_group(...)`` against the three worker agents.
    The group dispatch is fire-and-forget from the LLM's perspective
    — the tool returns immediately with the spoken acknowledgement
    so the voice agent is free to handle follow-up turns. The SDK
    forwards every task lifecycle event to the client as ``ui.task``
    envelopes, where the client renders progress and a cancel
    button.
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
        research_query: str | None = None,
    ):
        """Reply to the user. Optionally kick off background research.

        Always called exactly once per turn. ``answer`` is required.

        Args:
            answer: The spoken reply in plain language. One short
                sentence. For research turns, a brief acknowledgement
                like "Researching X now."
            research_query: Optional topic to research. When set, the
                server fans out three worker agents in parallel and
                streams progress to the page. Workers run in the
                background; the LLM does NOT wait for results.
        """
        logger.info(f"{self}: reply(answer={answer!r}, research_query={research_query!r})")
        if research_query:
            # Fire-and-forget the task group via the SDK helper. The
            # group_started envelope fires before this returns so the
            # client renders the in-flight card immediately; workers
            # run in the background and the SDK forwards every
            # lifecycle event automatically. The voice agent unblocks
            # as soon as we respond_to_task below.
            await self.start_user_task_group(
                "wikipedia",
                "news",
                "scholar",
                payload={"query": research_query},
                label=f"Research: {research_query}",
            )
        await self.respond_to_task(speak=answer)
        await params.result_callback(None)


class AsyncTasksRoot(BaseAgent):
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
                            "ask you to research any topic. One short "
                            "sentence."
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
            await self.add_agent(ResearchAgent("ui", bus=self.bus, keep_history=False))
            await self.add_agent(WikipediaResearcher("wikipedia", bus=self.bus))
            await self.add_agent(NewsResearcher("news", bus=self.bus))
            await self.add_agent(ScholarResearcher("scholar", bus=self.bus))

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
    root = AsyncTasksRoot("main", bus=runner.bus, transport=transport)
    await runner.add_agent(root)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

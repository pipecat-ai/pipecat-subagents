#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Document review — the synthesis demo.

A single workspace combining everything from the prior demos. The
user reviews a draft article. They can:

- Select a paragraph and ask for review. The UI agent fans out to
  two specialist worker agents (clarity, tone) in parallel. Their
  progress streams to an in-flight card. Each worker's feedback
  becomes a note attached to the paragraph.
- Dictate their own notes by voice. The agent fills the notes
  textarea and clicks Save; the note appears in the list.
- Ask "where does it talk about X" and the agent uses ``select_text``
  to navigate.
- Click any existing note in the panel; the client emits a
  ``note_click`` UI event, the agent's ``@on_ui_event("note_click")``
  handler dispatches ``select_text`` to jump to the related
  paragraph. Round-trip event/command pattern.

Architecture::

    ReviewRoot (BaseAgent, root)            -- transport + UI bridge
      ├── VoiceAgent (LLMAgent, bridged)    -- conversational layer
      │     └── @tool answer_about_screen(query)
      │           └── self.task("ui", payload={"query": query})
      ├── ReviewAgent (ReplyToolMixin + UIAgent)
      │     ├── inherited reply tool (scroll_to, highlight,
      │     │     select_text, fills, click)
      │     ├── @tool start_review(answer, paragraph_ref, paragraph_text)
      │     │     └── self.start_user_task_group("clarity", "tone", ...)
      │     ├── @on_ui_event("note_click")
      │     │     └── self.select_text(ref)
      │     └── on_task_response: emit add_note for each worker that
      │           completes (passes worker feedback through to the UI)
      └── two workers (BaseAgent each)
            ├── ClarityReviewer
            └── ToneReviewer

Workers are simulated, like async-tasks: a few ``send_task_update``
progress lines, then a ``send_task_response`` with a final analysis.
The analysis is computed from simple metrics (word count, sentence
count, presence of absolutist/hedging words) so different paragraphs
get different feedback without needing real NLP.

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
    ReplyToolMixin,
    TaskError,
    TaskStatus,
    UIAgent,
    agent_ready,
    attach_ui_bridge,
    on_ui_event,
    tool,
)
from pipecat_subagents.bus import AgentBus, BusBridgeProcessor
from pipecat_subagents.bus.messages import BusTaskRequestMessage, BusTaskResponseMessage
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
You are the voice layer of a document review assistant. A separate \
UI layer sees the page (the article and the notes panel) and writes \
the spoken reply.

For every user utterance about the document or the review (selecting \
paragraphs, asking for feedback, dictating notes, navigating), call \
``answer_about_screen`` with the user's request verbatim. The \
tool's response is the spoken reply, already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


UI_PROMPT = (
    """\
You are reviewing a draft article with the user. The current \
``<ui_state>`` block is in your context, and may contain a \
``<selection>`` block when the user has highlighted text.

You have two LLM tools:

## Tool: reply

For most turns. ``reply(answer, scroll_to=None, highlight=None, \
select_text=None, fills=None, click=None)``:

- ``answer`` (REQUIRED): the spoken reply, plain language, one or \
two short sentences.
- ``scroll_to`` (OPTIONAL): a snapshot ref. Scroll the element into \
view.
- ``select_text`` (OPTIONAL): a snapshot ref. Place the page's text \
selection on a paragraph (use this for "this paragraph" / "the \
section about X").
- ``highlight`` (OPTIONAL): list of refs. Brief flash. Rarely used \
here; ``select_text`` is usually better for paragraphs.
- ``fills`` (OPTIONAL): list of ``{"ref", "value"}`` objects. Fill \
the notes textarea (ref is in ``<ui_state>`` as the ``textbox``).
- ``click`` (OPTIONAL): list of refs to click. Use to click the \
Save button after filling the notes textarea.

## Tool: start_review

For "review this paragraph" / "give me feedback on this" requests. \
``start_review(answer, paragraph_ref, paragraph_text)``:

- ``answer`` (REQUIRED): brief acknowledgement spoken right away \
("Reviewing this paragraph").
- ``paragraph_ref`` (REQUIRED): the snapshot ref of the paragraph \
under review. When the user has a selection, use the selection's \
ref. Otherwise pick the right paragraph from ``<ui_state>``.
- ``paragraph_text`` (REQUIRED): the full paragraph text. Read it \
from the ``<selection>`` block when present, or from the ``name`` \
attribute on the paragraph node in ``<ui_state>``.

The server fans out two worker reviewers (clarity, tone) in \
parallel and streams progress to the page. As each worker finishes, \
their feedback becomes a note attached to the paragraph. You do NOT \
wait for results.

## Decision rules

- **"Review this", "give me feedback on this paragraph", "what do \
you think of this"** with a selection → ``start_review``.
- **"Review the third paragraph"** with no selection → use \
``<ui_state>`` to find the ref + text, call ``start_review``.
- **"Add a note: …"** or any dictated note content → use ``reply`` \
with ``fills`` for the notes textarea and ``click`` on the Save \
button. Pre-existing user selection (if any) becomes the note's \
attached paragraph automatically (the client picks it up).
- **"Where does it talk about X"** → ``reply`` with ``scroll_to`` + \
``select_text`` to navigate to the matching paragraph.
- **"Read me back the notes"** / **"What did you say about \
paragraph 3"** → ``reply`` with answer text only; the notes panel \
is in ``<ui_state>`` so you can summarize from it.
- **General questions about the draft** → ``reply`` with answer \
only.

## Examples

(refs are illustrative; use actual refs from the current snapshot)

- User has selected paragraph e8, says "Review this." → \
``start_review(answer="Reviewing this paragraph.", paragraph_ref="e8", paragraph_text="The asynchronous-first model that emerged...")``
- "Add a note that this is too dense" with paragraph e8 selected → \
``reply(answer="Noted.", fills=[{"ref": "<textarea_ref>", "value": "This paragraph is too dense."}], click=["<save_button_ref>"])``
- "Where does it talk about rhythms?" → \
``reply(answer="Here, in this paragraph.", scroll_to="e14", select_text="e14")``

"""
    + UI_STATE_PROMPT_GUIDE
)


# ─────────────────────────────────────────────────────────────────────
# Workers: simulated reviewers that compute simple text metrics and
# send back a plausible-sounding review. The analysis is canned but
# varies per paragraph based on actual properties of the text.
# ─────────────────────────────────────────────────────────────────────


class _SimulatedReviewer(BaseAgent):
    """Base for the two simulated reviewers."""

    source_name: str = "reviewer"

    def review(self, text: str) -> str:
        return ""

    async def on_task_request(self, message: BusTaskRequestMessage) -> None:
        await super().on_task_request(message)
        task_id = message.task_id
        payload = message.payload or {}
        text = str(payload.get("text", "")).strip()

        try:
            await asyncio.sleep(random.uniform(0.4, 0.9))
            await self.send_task_update(task_id, {"text": f"reading {len(text.split())} words"})

            await asyncio.sleep(random.uniform(0.5, 1.1))
            await self.send_task_update(task_id, {"text": f"checking {self.source_name}"})

            await asyncio.sleep(random.uniform(0.4, 0.9))
            feedback = self.review(text) or "(no notes)"
            await self.send_task_response(task_id, response={"feedback": feedback})
        except asyncio.CancelledError:
            raise


class ClarityReviewer(_SimulatedReviewer):
    """Comments on density, sentence length, and structural issues."""

    source_name = "clarity"

    def review(self, text: str) -> str:
        words = len(text.split())
        # Cheap sentence count: terminal punctuation.
        sentences = max(1, sum(1 for ch in text if ch in ".!?"))
        avg = words / sentences

        if avg > 35:
            return (
                f"This passage runs {words} words across just {sentences} "
                f"sentence(s) (~{avg:.0f} words each). Consider breaking "
                "it into smaller units; the reader is asked to hold a lot "
                "in working memory."
            )
        if words < 25:
            return (
                f"Brief at {words} words. If this is a key idea, consider "
                "expanding with one concrete example."
            )
        if avg < 12:
            return (
                f"Sentences average {avg:.0f} words. This is fine, "
                "sometimes preferable, but watch for choppiness if "
                "several short ones run in a row."
            )
        return (
            f"Density is reasonable at ~{avg:.0f} words per sentence across {sentences} sentences."
        )


class ToneReviewer(_SimulatedReviewer):
    """Comments on hedging, overstatement, and word choice."""

    source_name = "tone"

    ABSOLUTIST = (
        "simply",
        "anyone who",
        "unanimous",
        "always",
        "never",
        "obviously",
        "comprehensively",
    )
    HEDGES = ("might", "perhaps", "seems", "appears", "could", "may")

    def review(self, text: str) -> str:
        lower = text.lower()
        absolutes = [w for w in self.ABSOLUTIST if w in lower]
        hedges = [w for w in self.HEDGES if w in lower]

        if absolutes:
            sample = ", ".join(repr(w) for w in absolutes[:3])
            return (
                f"Strong words flagged: {sample}. If the claim is contested "
                "or the evidence is mixed, some hedging would read as more "
                "credible."
            )
        if len(hedges) >= 4:
            return (
                f"Heavy hedging — I count {len(hedges)} hedge words. Fine "
                "for an exploratory section, but if you mean to commit to "
                "a claim, the hedges weaken it."
            )
        return "Tone reads as measured. No flags."


# ─────────────────────────────────────────────────────────────────────
# Voice agent (bridged) — same shape as the other demos.
# ─────────────────────────────────────────────────────────────────────


class VoiceAgent(LLMAgent):
    """Conversational layer. Delegates document utterances to the UI agent."""

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


# ─────────────────────────────────────────────────────────────────────
# Review UI agent.
# ─────────────────────────────────────────────────────────────────────


class ReviewAgent(ReplyToolMixin, UIAgent):
    """UIAgent that drives the document review workspace.

    Composes ``ReplyToolMixin`` for the standard reply tool and adds
    a ``start_review`` tool for kicking off paragraph review. A
    ``@on_ui_event("note_click")`` handler converts client-side note
    clicks into ``select_text`` navigation. ``on_task_response`` is
    overridden to translate worker responses into ``add_note`` UI
    commands so each completed reviewer's feedback shows up in the
    notes panel as it lands.
    """

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(name, bus=bus, keep_history=False)
        # task_id -> {"paragraph_ref": "..."}; lets on_task_response
        # know which paragraph a worker's feedback belongs to.
        self._reviews: dict[str, dict] = {}

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                model=os.getenv("OPENAI_MODEL"),
                system_instruction=UI_PROMPT,
            ),
        )

    async def on_task_request(self, message: BusTaskRequestMessage) -> None:
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
    async def start_review(
        self,
        params: FunctionCallParams,
        answer: str,
        paragraph_ref: str,
        paragraph_text: str,
    ):
        """Kick off a parallel review of one paragraph.

        Spawns the clarity and tone workers via
        ``start_user_task_group``. Workers run in the background; the
        SDK forwards their progress to the page automatically. As each
        completes, ``on_task_response`` translates the response into
        an ``add_note`` UI command.

        Args:
            answer: A short spoken acknowledgement ("Reviewing this
                paragraph").
            paragraph_ref: The snapshot ref of the paragraph under
                review.
            paragraph_text: The paragraph's text content. Workers
                analyze this directly.
        """
        logger.info(f"{self}: start_review(ref={paragraph_ref!r})")
        task_id = await self.start_user_task_group(
            "clarity",
            "tone",
            payload={"ref": paragraph_ref, "text": paragraph_text},
            label=f"Reviewing ¶ {paragraph_ref}",
        )
        # Remember which paragraph this review is for so we can attach
        # each worker's response to the right note.
        self._reviews[task_id] = {"paragraph_ref": paragraph_ref}
        await self.respond_to_task(speak=answer)
        await params.result_callback(None)

    async def on_task_response(self, message: BusTaskResponseMessage) -> None:
        """Turn worker review responses into ``add_note`` UI commands."""
        await super().on_task_response(message)
        review = self._reviews.get(message.task_id)
        if not review:
            return
        if message.status != TaskStatus.COMPLETED:
            return
        feedback = ((message.response or {}).get("feedback") or "").strip()
        if not feedback:
            return
        await self.send_command(
            "add_note",
            {
                "source": message.source,
                "ref": review["paragraph_ref"],
                "text": feedback,
            },
        )

    @on_ui_event("note_click")
    async def on_note_click(self, message) -> None:
        """User clicked a note in the panel; jump to its paragraph."""
        ref = (message.payload or {}).get("ref")
        if not isinstance(ref, str) or not ref:
            return
        logger.info(f"{self}: note_click → select_text({ref!r})")
        await self.scroll_to(ref)
        await self.select_text(ref)


class ReviewRoot(BaseAgent):
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
                            "select any paragraph and ask you to review "
                            "it, dictate notes, or navigate the draft. "
                            "One short sentence."
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
            await self.add_agent(ReviewAgent("ui", bus=self.bus))
            await self.add_agent(ClarityReviewer("clarity", bus=self.bus))
            await self.add_agent(ToneReviewer("tone", bus=self.bus))

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
    root = ReviewRoot("main", bus=runner.bus, transport=transport)
    await runner.add_agent(root)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

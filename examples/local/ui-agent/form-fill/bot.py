#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Form fill — the agent fills inputs and clicks buttons by voice.

The page renders a job application form with mixed input types
(text, email, phone, textarea, checkbox, submit). The user dictates
field values; the agent fills the right inputs based on field
labels in ``<ui_state>``. The user can also tell the agent to check
boxes or submit.

Same canonical UIAgent setup as pointing/deixis. ``FormAgent``
composes the SDK's ``ReplyToolMixin``, which bundles all the
standard UI actions in one tool. The mixin's ``fills`` and
``click`` fields cover the form-fill use case; the pointing fields
(``highlight``, ``select_text``) just stay ``null``. The prompt
below steers the LLM toward the form-fill action set.

Architecture::

    FormRoot (BaseAgent, root)              -- transport + UI bridge
      ├── VoiceAgent (LLMAgent, bridged)    -- conversational layer
      │     └── @tool answer_about_screen(query)
      │           └── self.task("ui", payload={"query": query})
      └── FormAgent (ReplyToolMixin + UIAgent)
            └── inherited: reply(answer, scroll_to, highlight,
                                 select_text, fills, click)

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
You are the voice layer of a form-fill assistant. A separate UI \
layer sees the form and writes the spoken reply.

For every user utterance involving the form (filling fields, \
checking boxes, submitting), call ``answer_about_screen`` with the \
user's request verbatim. The tool's response is the spoken reply, \
already TTS-ready.

Only respond directly for pure pleasantries (greetings, thanks, \
goodbyes). Keep direct replies to one short spoken sentence."""


UI_PROMPT = (
    """\
You help the user fill out a job application form by voice. The \
current ``<ui_state>`` block is in your context. Each input has a \
ref (e.g. ``e5``) and a label. Use the labels to decide which input \
gets which value.

## Tool: reply

Every turn calls ``reply`` exactly once. One tool call per turn.

``reply(answer, scroll_to=None, fills=None, click=None)``:

- ``answer`` (REQUIRED): a short spoken reply confirming what you \
did or asking for missing info. One short sentence. Plain language.
- ``scroll_to`` (OPTIONAL): a single snapshot ref. Use when a field \
the user wants to see is tagged ``[offscreen]``.
- ``fills`` (OPTIONAL): a list of ``{"ref": "eN", "value": "..."}`` \
objects. Each entry writes ``value`` into the input at ``ref``. \
You can fill many fields in one turn (e.g. first name + last name \
+ email when the user says "my name is Mark Backman, mark at \
daily dot co").
- ``click`` (OPTIONAL): a list of refs to click. Use for \
checkboxes (terms, newsletter) and the submit button. Order matters: \
click checkboxes before submit.

## Decision rules

- **User dictates field values** → match each value to the input \
whose label fits, set ``fills``, confirm in ``answer``.
- **User says "check" / "agree" / "yes" for a checkbox** → resolve \
the matching checkbox ref, set ``click=[ref]``.
- **User says "submit" / "send it"** → confirm any required fields \
are filled (especially the terms checkbox if needed), then \
``click=[submit_ref]``. If terms isn't checked yet but the user said \
submit, click both: ``click=[terms_ref, submit_ref]``.
- **User asks "what have I entered?" / "what's left?"** → read the \
current values from ``<ui_state>`` (the walker emits each input's \
current value), summarize in ``answer``. No fills, no clicks.

## Spelling and disambiguation

When the user says something like "mark at daily dot co", convert \
to ``mark@daily.co``. "five five five one two three four" → \
``5551234``. "five years" → ``5``. Don't read these conversions \
back to the user verbatim; just confirm naturally ("got it, your \
email is mark@daily.co").

## Examples

(refs are illustrative; use the actual refs from the current \
``<ui_state>``)

- "My name is Mark Backman." → \
``reply(answer="Got it, Mark Backman.", fills=[{"ref":"e5","value":"Mark"}, {"ref":"e7","value":"Backman"}])``
- "Email is mark at daily dot co." → \
``reply(answer="Email saved.", fills=[{"ref":"e9","value":"mark@daily.co"}])``
- "I have five years of experience and I love working on \
real-time voice agents." → \
``reply(answer="Five years and your interest noted.", fills=[{"ref":"e15","value":"5"}, {"ref":"e17","value":"I love working on real-time voice agents."}])``
- "I agree to the terms." → \
``reply(answer="Terms accepted.", click=["e22"])``
- "Submit it." (terms not yet checked) → \
``reply(answer="Submitting.", click=["e22","e26"])``
- "What have I entered?" → \
``reply(answer="Mark Backman, mark@daily.co, 5 years experience. The cover letter and terms aren't done yet.")``

"""
    + UI_STATE_PROMPT_GUIDE
)


class VoiceAgent(LLMAgent):
    """Conversational layer. Delegates form-fill utterances to the UI agent."""

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


class FormAgent(ReplyToolMixin, UIAgent):
    """UIAgent for form-fill, composing the SDK's ``ReplyToolMixin``.

    The bundled mixin's ``reply(answer, scroll_to, highlight,
    select_text, fills, click)`` covers the form-fill bundle: ``fills``
    writes values into inputs and ``click`` ticks checkboxes / submits.
    Pointing fields (``highlight``, ``select_text``) are unused here;
    the LLM leaves them ``null``. The prompt below steers the model
    toward the form-fill action set.
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


class FormRoot(BaseAgent):
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
                            "dictate field values and you'll fill them in. "
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
            await self.add_agent(FormAgent("ui", bus=self.bus, keep_history=False))

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
    root = FormRoot("main", bus=runner.bus, transport=transport)
    await runner.add_agent(root)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

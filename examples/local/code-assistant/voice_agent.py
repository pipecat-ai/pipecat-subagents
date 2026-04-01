#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent that dispatches code questions to a Claude SDK worker."""

import os

from code_worker import CodeWorker
from loguru import logger
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_subagents.agents import LLMAgent, tool
from pipecat_subagents.bus import AgentBus

PROJECT_PATH = os.getenv("PROJECT_PATH", os.getcwd())


class VoiceAgent(LLMAgent):
    """Voice agent that dispatches code questions to a Claude SDK worker."""

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(name, bus=bus, bridged=())

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(
                system_instruction=(
                    "You are a voice interface to a code assistant powered by Claude Code. "
                    "Behind you is a worker that can read files, search code with grep and "
                    "glob patterns, and run bash commands on the project. It maintains "
                    "context across questions, so follow-up questions work naturally.\n\n"
                    "When the user asks anything about code, project structure, files, "
                    "dependencies, tests, or wants to explore the codebase, call the "
                    "ask_code tool. When the answer comes back, summarize it naturally "
                    "for speaking. Keep responses concise and conversational.\n\n"
                    "If the user asks something unrelated to the project, answer directly."
                ),
            ),
        )

    async def on_ready(self) -> None:
        await super().on_ready()
        worker = CodeWorker("code_worker", bus=self.bus, project_path=PROJECT_PATH)
        await self.add_agent(worker)

    @tool(cancel_on_interruption=False, timeout=60)
    async def ask_code(self, params: FunctionCallParams, question: str):
        logger.info(f"Agent '{self.name}': asking code worker: '{question}'")
        async with self.task("code_worker", payload={"question": question}) as task:
            await params.llm.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "developer", "content": "Give me a moment."}],
                    run_llm=True,
                )
            )
            # Do other things while the task is running.
        await params.result_callback(task.response)

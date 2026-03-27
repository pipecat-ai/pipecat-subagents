#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Code worker that explores a codebase using Claude Agent SDK."""

import asyncio

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from loguru import logger

from pipecat_subagents.agents import BaseAgent
from pipecat_subagents.agents.task_group import TaskStatus
from pipecat_subagents.bus import AgentBus


class CodeWorker(BaseAgent):
    """Worker that answers code questions using Claude Agent SDK.

    Maintains a persistent Claude SDK session so follow-up questions
    share context. Questions are queued and processed sequentially.
    """

    def __init__(self, name: str, *, bus: AgentBus, project_path: str):
        super().__init__(name, bus=bus)
        self._project_path = project_path
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task = None
        self._claude_options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            system_prompt=(
                f"You are a code assistant. The project is at: {self._project_path}\n\n"
                "Answer the user's question by exploring the codebase. Use Read to "
                "view files, Glob to find files by pattern, and Bash to run commands "
                "like grep or find. Be thorough but concise in your answer. "
                "Focus on what the user asked. Respond with a clear, spoken-friendly "
                "summary (no markdown, no bullet points, no code blocks)."
            ),
            allowed_tools=["Read", "Bash", "Glob", "Grep"],
            model="sonnet",
            max_turns=10,
        )

    async def on_ready(self):
        await super().on_ready()
        self._worker_task = self.create_asyncio_task(self._worker_loop(), f"{self.name}::worker")

    async def on_task_request(self, message):
        await super().on_task_request(message)
        question = message.payload["question"]
        logger.info(f"Worker '{self.name}': queued '{question}'")
        self._queue.put_nowait(question)

    async def _worker_loop(self):
        try:
            async with ClaudeSDKClient(options=self._claude_options) as client:
                while True:
                    question = await self._queue.get()
                    logger.info(f"Worker '{self.name}': researching '{question}'")

                    try:
                        answer = ""
                        await client.query(prompt=question)
                        async for msg in client.receive_response():
                            if type(msg).__name__ == "AssistantMessage":
                                for block in msg.content:
                                    if type(block).__name__ == "TextBlock":
                                        answer += block.text

                        logger.info(f"Worker '{self.name}': completed ({len(answer)} chars)")
                        await self.send_task_response({"answer": answer})

                    except Exception as e:
                        logger.error(f"Worker '{self.name}': error: {e}")
                        await self.send_task_response({"error": str(e)}, status=TaskStatus.ERROR)
        except Exception as e:
            logger.error(f"Worker '{self.name}': failed to start Claude SDK: {e}")

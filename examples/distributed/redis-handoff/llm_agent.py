#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent — run on Machine B (or locally alongside main_agent.py).

A standalone process that runs one LLM agent (greeter or support) and
connects to the same Redis bus as the main agent. Multiple instances
can run on different machines.

Usage:
    python llm_agent.py greeter --redis-url redis://localhost:6379
    python llm_agent.py support --redis-url redis://localhost:6379

Requirements:
    OPENAI_API_KEY
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from redis.asyncio import Redis

from pipecat_subagents.agents import LLMActivationArgs, LLMAgent, tool
from pipecat_subagents.bus.network.redis import RedisBus
from pipecat_subagents.bus.serializers import JSONMessageSerializer
from pipecat_subagents.runner import AgentRunner

load_dotenv(override=True)

SYSTEM_INSTRUCTIONS = {
    "greeter": (
        "You are a friendly greeter for Acme Corp. The available products "
        "are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme "
        "Tornado Kit. Ask which one they'd like to learn more about. "
        "When the user picks a product or asks a question about one, "
        "immediately call the transfer_to_agent tool with agent 'support'. "
        "Do not answer product questions yourself. If the user says goodbye, "
        "call the end_conversation tool. Do not mention transferring — just do it "
        "seamlessly. Keep responses brief — this is a voice conversation."
    ),
    "support": (
        "You are a support agent for Acme Corp. You know about three "
        "products: Acme Rocket Boots (jet-powered boots, $299, run up "
        "to 60 mph), Acme Invisible Paint (makes anything invisible for "
        "24 hours, $49 per can), and Acme Tornado Kit (portable tornado "
        "generator, $199, batteries included). Answer the user's questions "
        "about these products. If the user wants to browse other products "
        "or start over, call the transfer_to_agent tool with agent "
        "'greeter'. If the user says goodbye, call the end_conversation "
        "tool. Do not mention transferring — just do it seamlessly. "
        "Keep responses brief — this is a voice conversation."
    ),
}


class AcmeLLMAgent(LLMAgent):
    """LLM agent for Acme Corp with transfer and end tools."""

    def __init__(self, name: str, *, bus, system_instruction: str):
        super().__init__(name, bus=bus)
        self._system_instruction = system_instruction

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMSettings(system_instruction=self._system_instruction),
        )

    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(self, params: FunctionCallParams, agent: str, reason: str):
        """Transfer the user to another agent.

        Args:
            agent (str): The agent to transfer to (e.g. 'greeter', 'support').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Agent '{self.name}': transferring to '{agent}' ({reason})")
        await self.deactivate_agent(result_callback=params.result_callback)
        await self.activate_agent(
            agent,
            args=LLMActivationArgs(messages=[{"role": "user", "content": reason}]),
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"Agent '{self.name}': ending conversation ({reason})")
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": reason}], run_llm=True)
        )
        await self.end(reason=reason, result_callback=params.result_callback)


async def main():
    parser = argparse.ArgumentParser(description="LLM agent (greeter or support)")
    parser.add_argument(
        "agent", choices=["greeter", "support"], help="Which agent to run"
    )
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis URL"
    )
    parser.add_argument(
        "--channel", default="pipecat:acme", help="Redis pub/sub channel"
    )
    args = parser.parse_args()

    redis = Redis.from_url(args.redis_url)
    serializer = JSONMessageSerializer()
    bus = RedisBus(redis=redis, serializer=serializer, channel=args.channel)

    agent = AcmeLLMAgent(
        args.agent,
        bus=bus,
        system_instruction=SYSTEM_INSTRUCTIONS[args.agent],
    )

    runner = AgentRunner(bus=bus, handle_sigint=True)
    await runner.add_agent(agent)
    logger.info(f"Starting {args.agent} agent, waiting for activation...")
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())

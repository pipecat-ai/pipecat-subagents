#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM agent — run on Machine B (or locally alongside main_agent.py).

A standalone process that runs one LLM agent (greeter or support) and
connects to the same PGMQ-backed bus as the main agent. Multiple
instances can run on different machines as long as they share a Postgres
database with the PGMQ extension enabled.

Usage:
    python llm_agent.py greeter --database-url postgresql://...
    python llm_agent.py support --database-url postgresql://...

Requirements:
    OPENAI_API_KEY, DATABASE_URL
"""

import argparse
import asyncio
import os
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from loguru import logger
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from tembo_pgmq_python.async_queue import PGMQueue

from pipecat_subagents.agents import LLMAgent, LLMAgentActivationArgs, tool
from pipecat_subagents.bus.network.pgmq import PgmqBus
from pipecat_subagents.runner import AgentRunner

load_dotenv(override=True)

AGENT_CONFIG = {
    "greeter": {
        "system_instruction": (
            "You are a friendly greeter for Acme Corp. The available products "
            "are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme "
            "Tornado Kit. Ask which one they'd like to learn more about. "
            "When the user picks a product or asks a question about one, "
            "immediately call the transfer_to_agent tool with agent 'support'. "
            "Do not answer product questions yourself. If the user says goodbye, "
            "call the end_conversation tool. Do not mention transferring — just do it "
            "seamlessly. Keep responses brief — this is a voice conversation."
        ),
        "watch_agents": ["support"],
    },
    "support": {
        "system_instruction": (
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
        "watch_agents": ["greeter"],
    },
}


def pgmq_from_url(database_url: str, *, pool_size: int = 4) -> PGMQueue:
    """Build a PGMQueue from a Postgres DSN string."""
    parsed = urlparse(database_url)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ValueError(f"Unsupported scheme '{parsed.scheme}' for database URL")
    return PGMQueue(
        host=parsed.hostname or "localhost",
        port=str(parsed.port or 5432),
        database=(parsed.path or "/postgres").lstrip("/") or "postgres",
        username=unquote(parsed.username or "postgres"),
        password=unquote(parsed.password or ""),
        pool_size=pool_size,
    )


class AcmeLLMAgent(LLMAgent):
    """LLM agent for Acme Corp with transfer and end tools."""

    def __init__(self, name: str, *, bus, system_instruction: str, watch_agents: list[str]):
        super().__init__(name, bus=bus, bridged=())
        self._system_instruction = system_instruction
        self._watch_agents = watch_agents

    async def on_ready(self) -> None:
        await super().on_ready()
        for agent_name in self._watch_agents:
            await self.watch_agent(agent_name)

    def build_llm(self) -> LLMService:
        return OpenAILLMService(
            name=f"{self.name}::OpenAILLMService",
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
        await self.handoff_to(
            agent,
            activation_args=LLMAgentActivationArgs(
                messages=[{"role": "developer", "content": reason}]
            ),
            result_callback=params.result_callback,
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"Agent '{self.name}': ending conversation ({reason})")
        await self.end(
            reason=reason,
            messages=[{"role": "developer", "content": reason}],
            result_callback=params.result_callback,
        )


async def main():
    parser = argparse.ArgumentParser(description="LLM agent (greeter or support)")
    parser.add_argument("agent", choices=["greeter", "support"], help="Which agent to run")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL DSN (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--channel",
        default=os.getenv("PGMQ_CHANNEL", "pipecat_acme"),
        help="PGMQ channel prefix",
    )
    args = parser.parse_args()

    if not args.database_url:
        parser.error("--database-url is required (or set DATABASE_URL env var)")

    pgmq = pgmq_from_url(args.database_url)
    await pgmq.init()
    bus = PgmqBus(pgmq=pgmq, channel=args.channel)

    config = AGENT_CONFIG[args.agent]
    agent = AcmeLLMAgent(
        args.agent,
        bus=bus,
        system_instruction=config["system_instruction"],
        watch_agents=config["watch_agents"],
    )

    runner = AgentRunner(bus=bus, handle_sigint=True)
    await runner.add_agent(agent)
    logger.info(f"Starting {args.agent} agent, waiting for activation...")
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())

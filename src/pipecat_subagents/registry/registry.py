#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared agent registry for tracking discovered agents across runners."""

from collections import defaultdict
from typing import Callable, Coroutine, Optional

from loguru import logger

from pipecat_subagents.types import RegisteredAgentData

WatchHandler = Callable[[RegisteredAgentData], Coroutine]


class AgentRegistry:
    """Single source of truth for all known agents (local and remote).

    Owned by a runner and shared with its agents. Organizes agents into
    local (this runner) and remote (other runners) so they are easy to
    distinguish. Deduplication is built in: each agent name is
    registered at most once.

    Notifications use a targeted watch mechanism: call
    ``watch(agent_name, handler)`` to be notified when a specific agent
    registers.

    Args:
        runner_name: Name of the runner that owns this registry.
    """

    def __init__(self, runner_name: str):
        self._runner_name = runner_name
        self._local_agents: dict[str, RegisteredAgentData] = {}
        self._remote_runners: dict[str, dict[str, RegisteredAgentData]] = defaultdict(dict)
        self._watches: dict[str, list[WatchHandler]] = defaultdict(list)

    @property
    def runner_name(self) -> str:
        """The name of the runner that owns this registry."""
        return self._runner_name

    @property
    def local_agents(self) -> list[str]:
        """Names of agents registered under this runner."""
        return list(self._local_agents.keys())

    @property
    def remote_agents(self) -> list[str]:
        """Names of agents registered under remote runners."""
        result: list[str] = []
        for agents in self._remote_runners.values():
            result.extend(agents.keys())
        return result

    def get(self, agent_name: str) -> Optional[RegisteredAgentData]:
        """Look up a registered agent by name."""
        if agent_name in self._local_agents:
            return self._local_agents[agent_name]
        for agents in self._remote_runners.values():
            if agent_name in agents:
                return agents[agent_name]
        return None

    def __contains__(self, agent_name: str) -> bool:
        return self.get(agent_name) is not None

    def watch(self, agent_name: str, handler: WatchHandler) -> None:
        """Watch for a specific agent's registration.

        Args:
            agent_name: The agent name to watch for.
            handler: Async callable invoked with the agent's data.
        """
        self._watches[agent_name].append(handler)

    async def register(self, agent_data: RegisteredAgentData) -> bool:
        """Register an agent. Returns True if the agent was new.

        If the agent is already registered, this is a no-op and returns
        False. Otherwise the agent is added and watchers are notified.

        Args:
            agent_data: Information about the agent to register.

        Returns:
            True if the agent was newly registered, False if already known.
        """
        is_local = agent_data.runner == self._runner_name
        target = self._local_agents if is_local else self._remote_runners[agent_data.runner]

        if agent_data.agent_name in target:
            return False

        target[agent_data.agent_name] = agent_data
        locality = "local" if is_local else f"remote ({agent_data.runner})"
        logger.debug(f"Agent '{agent_data.agent_name}' ready ({locality})")
        await self._notify(agent_data)
        return True

    async def _notify(self, agent_data: RegisteredAgentData) -> None:
        """Notify watchers of a new registration."""
        for handler in self._watches.get(agent_data.agent_name, []):
            await handler(agent_data)

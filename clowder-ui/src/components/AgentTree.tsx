import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store";
import type { Agent, Task } from "../types";
import { Bot, ChevronRight, ChevronDown, Filter, ListTodo, Timer, Server } from "lucide-react";
import { AGENT_STATE_COLORS, TASK_STATUS_COLORS } from "../colors";

function formatDuration(startedAt: number, completedAt: number | null): string {
  const end = completedAt || Date.now() / 1000;
  const seconds = end - startedAt;
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function DurationTicker({ startedAt, completedAt }: { startedAt: number; completedAt: number | null }) {
  const [, setTick] = useState(0);
  const isRunning = completedAt === null;

  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => setTick((t) => t + 1), 200);
    return () => clearInterval(id);
  }, [isRunning]);

  return <>{formatDuration(startedAt, completedAt)}</>;
}

function TaskStatusBadge({ status }: { status: string }) {
  const colors = TASK_STATUS_COLORS[status] || TASK_STATUS_COLORS.running;
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[10px] font-medium"
      style={{ backgroundColor: colors.bg, color: colors.fg }}
    >
      {status}
    </span>
  );
}

function FilterButton({ onClick }: { onClick: (e: React.MouseEvent) => void }) {
  return (
    <span
      onClick={onClick}
      className="shrink-0 px-1.5 py-0.5 rounded cursor-pointer hover:opacity-80"
      style={{ color: "hsl(var(--muted-foreground))", opacity: 0.5 }}
      title="Add to filter"
    >
      <Filter size={10} />
    </span>
  );
}

function TaskRow({ task, depth }: { task: Task; depth: number }) {
  const { selection, setSelection, addEntityFilter } = useStore();
  const isSelected = selection?.type === "task" && selection.taskId === task.task_id;
  return (
    <div
      className="group flex items-center gap-2 py-1 text-xs"
      style={{
        paddingLeft: `${12 + depth * 16}px`,
        color: "hsl(var(--muted-foreground))",
        backgroundColor: isSelected ? "hsla(243, 75%, 59%, 0.08)" : undefined,
      }}
    >
      <button
        onClick={() => setSelection({ type: "task", taskId: task.task_id })}
        className="flex-1 flex items-center gap-2 text-left min-w-0"
      >
        <ListTodo size={12} className="shrink-0" />
        <span className="truncate text-sm">
          {task.task_name || task.task_id.slice(0, 8)}
        </span>
        <span style={{ opacity: 0.6 }}>from {task.source}</span>
        <TaskStatusBadge status={task.status} />
        <span
          className="ml-auto flex items-center gap-1 shrink-0"
          style={{ opacity: 0.6 }}
        >
          <Timer size={10} />
          <DurationTicker startedAt={task.started_at} completedAt={task.completed_at} />
        </span>
      </button>
      <FilterButton
        onClick={(e) => {
          e.stopPropagation();
          addEntityFilter({ type: "task", value: task.task_id });
        }}
      />
    </div>
  );
}

function AgentNode({
  agent,
  childrenMap,
  tasksByAgent,
  depth,
}: {
  agent: Agent;
  childrenMap: Map<string, Agent[]>;
  tasksByAgent: Map<string, Task[]>;
  depth: number;
}) {
  const { selection, setSelection, addEntityFilter } = useStore();
  const [expanded, setExpanded] = useState(true);
  const isSelected = selection?.type === "agent" && selection.name === agent.name;
  const children = childrenMap.get(agent.name) || [];
  const tasks = tasksByAgent.get(agent.name) || [];
  const hasContent = children.length > 0 || tasks.length > 0;

  const Chevron = expanded ? ChevronDown : ChevronRight;

  return (
    <div>
      <div
        className="group flex items-center"
        style={{
          backgroundColor: isSelected
            ? "hsla(243, 75%, 59%, 0.08)"
            : undefined,
        }}
      >
        {hasContent ? (
          <button
            onClick={() => setExpanded(!expanded)}
            className="shrink-0 p-0.5"
            style={{
              marginLeft: `${12 + depth * 16 - 4}px`,
              color: "hsl(var(--muted-foreground))",
            }}
          >
            <Chevron size={12} />
          </button>
        ) : (
          <span
            className="shrink-0"
            style={{ marginLeft: `${12 + depth * 16}px`, width: "12px" }}
          />
        )}
        <button
          onClick={() => setSelection(isSelected ? null : { type: "agent", name: agent.name })}
          className="flex-1 flex items-center gap-2 px-1.5 py-1.5 text-sm text-left min-w-0"
        >
          <Bot
            size={14}
            className="shrink-0"
            style={{
              color: agent.ready
                ? agent.active
                  ? "hsl(150, 60%, 45%)"
                  : "hsl(var(--muted-foreground))"
                : "hsl(var(--border))",
            }}
          />
          <span
            className="truncate"
            style={{
              fontWeight: 400,
              color: agent.active
                ? "hsl(var(--foreground))"
                : "hsl(var(--muted-foreground))",
            }}
          >
            {agent.name}
          </span>
          {agent.bridged && (
            <span
              className="px-1.5 py-0.5 rounded text-[10px] font-medium"
              style={{ backgroundColor: AGENT_STATE_COLORS.bridged.bg, color: AGENT_STATE_COLORS.bridged.fg }}
            >
              bridged
            </span>
          )}
          {agent.active && (
            <span
              className="px-1.5 py-0.5 rounded text-[10px] font-medium"
              style={{ backgroundColor: AGENT_STATE_COLORS.active.bg, color: AGENT_STATE_COLORS.active.fg }}
            >
              active
            </span>
          )}
          {agent.bridged && !agent.active && (
            <span
              className="px-1.5 py-0.5 rounded text-[10px] font-medium"
              style={{ backgroundColor: AGENT_STATE_COLORS.idle.bg, color: AGENT_STATE_COLORS.idle.fg }}
            >
              idle
            </span>
          )}
          {(children.length > 0 || tasks.length > 0) && (
            <span
              className="text-xs"
              style={{ color: "hsl(var(--muted-foreground))", opacity: 0.5 }}
            >
              ({[
                children.length > 0 && `${children.length} subagent${children.length !== 1 ? "s" : ""}`,
                tasks.length > 0 && `${tasks.length} task${tasks.length !== 1 ? "s" : ""}`,
              ].filter(Boolean).join(", ")})
            </span>
          )}
        </button>
        <FilterButton
          onClick={(e) => {
            e.stopPropagation();
            addEntityFilter({ type: "agent", value: agent.name });
          }}
        />
      </div>
      {expanded && (
        <>
          {tasks.map((task) => (
            <TaskRow key={task.task_id} task={task} depth={depth + 1} />
          ))}
          {children.map((child) => (
            <AgentNode
              key={child.name}
              agent={child}
              childrenMap={childrenMap}
              tasksByAgent={tasksByAgent}
              depth={depth + 1}
            />
          ))}
        </>
      )}
    </div>
  );
}

function RunnerNode({
  runnerName,
  agents,
  childrenMap,
  tasksByAgent,
}: {
  runnerName: string;
  agents: Agent[];
  childrenMap: Map<string, Agent[]>;
  tasksByAgent: Map<string, Task[]>;
}) {
  const { selection, setSelection, addEntityFilter } = useStore();
  const [expanded, setExpanded] = useState(true);
  const isSelected = selection?.type === "runner" && selection.name === runnerName;
  const Chevron = expanded ? ChevronDown : ChevronRight;

  return (
    <div>
      <div
        className="group flex items-center"
        style={{ backgroundColor: isSelected ? "hsla(243, 75%, 59%, 0.08)" : undefined }}
      >
        <button
          onClick={() => setExpanded(!expanded)}
          className="shrink-0 p-0.5 ml-2"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          <Chevron size={12} />
        </button>
        <button
          onClick={() => setSelection({ type: "runner", name: runnerName })}
          className="flex-1 flex items-center gap-2 px-1.5 py-2 text-xs font-medium uppercase tracking-wide text-left"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          <Server size={12} />
          {runnerName}
          <span style={{ opacity: 0.5 }} className="normal-case tracking-normal font-normal">
            ({agents.length} agent{agents.length !== 1 ? "s" : ""})
          </span>
        </button>
        <FilterButton
          onClick={(e) => {
            e.stopPropagation();
            addEntityFilter({ type: "runner", value: runnerName });
          }}
        />
      </div>
      {expanded &&
        agents.map((agent) => (
          <AgentNode
            key={agent.name}
            agent={agent}
            childrenMap={childrenMap}
            tasksByAgent={tasksByAgent}
            depth={1}
          />
        ))}
    </div>
  );
}

export function AgentTree() {
  const { agents, tasks } = useStore();

  const { runnerGroups, childrenMap, tasksByAgent } = useMemo(() => {
    const agentList = Object.values(agents);
    const childrenMap = new Map<string, Agent[]>();
    const roots: Agent[] = [];

    for (const agent of agentList) {
      if (agent.parent) {
        const siblings = childrenMap.get(agent.parent) || [];
        siblings.push(agent);
        childrenMap.set(agent.parent, siblings);
      } else {
        roots.push(agent);
      }
    }

    const runnerGroups = new Map<string, Agent[]>();
    for (const agent of roots) {
      const runner = agent.runner || "unknown";
      const group = runnerGroups.get(runner) || [];
      group.push(agent);
      runnerGroups.set(runner, group);
    }

    const tasksByAgent = new Map<string, Task[]>();
    for (const task of Object.values(tasks)) {
      for (const target of task.targets) {
        const list = tasksByAgent.get(target) || [];
        list.push(task);
        tasksByAgent.set(target, list);
      }
    }

    return { runnerGroups, childrenMap, tasksByAgent };
  }, [agents, tasks]);

  if (runnerGroups.size === 0) {
    return (
      <div
        className="p-4 text-sm"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        No agents detected
      </div>
    );
  }

  return (
    <div className="py-1">
      {Array.from(runnerGroups.entries()).map(([runnerName, rootAgents]) => (
        <RunnerNode
          key={runnerName}
          runnerName={runnerName}
          agents={rootAgents}
          childrenMap={childrenMap}
          tasksByAgent={tasksByAgent}
        />
      ))}
    </div>
  );
}

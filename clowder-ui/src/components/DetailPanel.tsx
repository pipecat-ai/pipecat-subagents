import { useEffect, useState } from "react";
import { useStore } from "../store";
import type { Agent, Task } from "../types";
import { AGENT_STATE_COLORS, TASK_STATUS_COLORS } from "../colors";
import { X } from "lucide-react";

function formatUptime(startedAt: number): string {
  const seconds = Date.now() / 1000 - startedAt;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  if (value === null || value === undefined) return null;
  return (
    <div className="flex items-baseline gap-3 py-1.5 border-b" style={{ borderColor: "hsl(var(--border))" }}>
      <span
        className="text-xs font-medium w-24 shrink-0"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        {label}
      </span>
      <span className="text-sm" style={{ color: "hsl(var(--foreground))" }}>
        {value}
      </span>
    </div>
  );
}

function Badge({ label, color, bg }: { label: string; color: string; bg: string }) {
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[10px] font-medium"
      style={{ backgroundColor: bg, color }}
    >
      {label}
    </span>
  );
}

function UptimeTicker({ startedAt }: { startedAt: number }) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return <>{formatUptime(startedAt)}</>;
}

function RunnerDetail({ name }: { name: string }) {
  const { agents } = useStore();
  const runnerAgents = Object.values(agents).filter((a) => a.runner === name);
  const activeCount = runnerAgents.filter((a) => a.active).length;
  const earliest = runnerAgents.reduce(
    (min, a) => (a.started_at && (!min || a.started_at < min) ? a.started_at : min),
    null as number | null
  );

  return (
    <>
      <Row label="Runner" value={name} />
      <Row label="Agents" value={`${runnerAgents.length} total, ${activeCount} active`} />
      {earliest && <Row label="Uptime" value={<UptimeTicker startedAt={earliest} />} />}
    </>
  );
}

function AgentDetail({ agent }: { agent: Agent }) {
  const badges = (
    <span className="flex gap-1">
      {agent.bridged && (
        <Badge label="bridged" color={AGENT_STATE_COLORS.bridged.fg} bg={AGENT_STATE_COLORS.bridged.bg} />
      )}
      {agent.active ? (
        <Badge label="active" color={AGENT_STATE_COLORS.active.fg} bg={AGENT_STATE_COLORS.active.bg} />
      ) : agent.bridged ? (
        <Badge label="idle" color={AGENT_STATE_COLORS.idle.fg} bg={AGENT_STATE_COLORS.idle.bg} />
      ) : null}
    </span>
  );

  return (
    <>
      <Row label="Agent" value={agent.name} />
      <Row label="Runner" value={agent.runner} />
      {agent.parent && <Row label="Parent" value={agent.parent} />}
      <Row label="Status" value={badges} />
      {agent.started_at && (
        <Row label="Uptime" value={<UptimeTicker startedAt={agent.started_at} />} />
      )}
      {agent.started_at && (
        <Row label="Started" value={formatTimestamp(agent.started_at)} />
      )}
    </>
  );
}

function TaskDetail({ task }: { task: Task }) {
  const duration = task.completed_at
    ? `${(task.completed_at - task.started_at).toFixed(2)}s`
    : formatUptime(task.started_at);

  return (
    <>
      <Row label="Task" value={task.task_name || task.task_id.slice(0, 12)} />
      <Row label="ID" value={<span className="font-mono text-xs">{task.task_id}</span>} />
      <Row label="Source" value={task.source} />
      <Row label="Targets" value={task.targets.join(", ")} />
      <Row
        label="Status"
        value={
          <Badge
            label={task.status}
            color={(TASK_STATUS_COLORS[task.status] || TASK_STATUS_COLORS.running).fg}
            bg={(TASK_STATUS_COLORS[task.status] || TASK_STATUS_COLORS.running).bg}
          />
        }
      />
      <Row label="Duration" value={duration} />

      <Row label="Started" value={formatTimestamp(task.started_at)} />
      {task.completed_at && <Row label="Completed" value={formatTimestamp(task.completed_at)} />}
    </>
  );
}

export function DetailPanel() {
  const { selection, setSelection, agents, tasks } = useStore();

  if (!selection) {
    return (
      <div
        className="h-full flex items-center justify-center text-sm"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        Select a runner, agent, or task
      </div>
    );
  }

  let content: React.ReactNode;
  let title: string;

  if (selection.type === "runner") {
    title = "Runner";
    content = <RunnerDetail name={selection.name} />;
  } else if (selection.type === "agent") {
    const agent = agents[selection.name];
    if (!agent) {
      content = <div className="p-3 text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>Agent not found</div>;
      title = "Agent";
    } else {
      title = "Agent";
      content = <AgentDetail agent={agent} />;
    }
  } else {
    const task = tasks[selection.taskId];
    if (!task) {
      content = <div className="p-3 text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>Task not found</div>;
      title = "Task";
    } else {
      title = "Task";
      content = <TaskDetail task={task} />;
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-3.5 py-1.5 border-b" style={{ borderColor: "hsl(var(--border))" }}>
        <span
          className="text-xs font-medium uppercase tracking-wide"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          {title}
        </span>
        <button
          onClick={() => setSelection(null)}
          className="p-0.5 rounded"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          <X size={12} />
        </button>
      </div>
      <div className="flex-1 overflow-auto px-3.5 py-2">
        {content}
      </div>
    </div>
  );
}

export type Agent = {
  name: string;
  parent: string | null;
  runner: string | null;
  active: boolean;
  ready: boolean;
  bridged: boolean;
  started_at: number | null;
};

export type Task = {
  task_id: string;
  source: string;
  targets: string[];
  task_name: string | null;
  status: string;
  started_at: number;
  completed_at: number | null;
};

export type BusEvent = {
  type: "bus_message";
  timestamp: number;
  message_type: string;
  category: "lifecycle" | "frame" | "task" | "other";
  source: string;
  target: string | null;
  data: Record<string, unknown>;
};

export type Snapshot = {
  type: "snapshot";
  timestamp: number;
  agents: Record<string, Agent>;
  tasks: Record<string, Task>;
  events: BusEvent[];
};

export type ServerMessage = Snapshot | BusEvent;

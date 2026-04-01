import { create } from "zustand";
import type { Agent, BusEvent, Snapshot, Task } from "./types";

const MAX_EVENTS = 500;

type CategoryFilter = "lifecycle" | "task" | "other";

type Theme = "light" | "dark";

type Selection =
  | { type: "runner"; name: string }
  | { type: "agent"; name: string }
  | { type: "task"; taskId: string }
  | null;

type EntityFilter =
  | { type: "runner"; value: string }
  | { type: "agent"; value: string }
  | { type: "task"; value: string };

type State = {
  // Theme
  theme: Theme;

  // Connection
  wsUrl: string;
  connected: boolean;

  // Domain state
  agents: Record<string, Agent>;
  tasks: Record<string, Task>;
  events: BusEvent[];

  // Filters
  enabledCategories: Set<CategoryFilter>;
  selectedMessageTypes: Set<string>;
  entityFilters: EntityFilter[];

  // Detail panel
  selection: Selection;

  // Event stream
  paused: boolean;

  // Actions
  setTheme: (theme: Theme) => void;
  setWsUrl: (url: string) => void;
  setConnected: (connected: boolean) => void;
  applySnapshot: (snapshot: Snapshot) => void;
  pushEvent: (event: BusEvent) => void;
  toggleCategory: (category: CategoryFilter) => void;
  toggleMessageType: (type: string) => void;
  clearMessageTypes: () => void;
  addEntityFilter: (filter: EntityFilter) => void;
  removeEntityFilter: (filter: EntityFilter) => void;
  clearEntityFilters: () => void;
  setSelection: (selection: Selection) => void;
  setPaused: (paused: boolean) => void;
  reset: () => void;
};

export type { CategoryFilter, EntityFilter, Selection };

function getInitialTheme(): Theme {
  const saved = localStorage.getItem("clowder-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export const useStore = create<State>((set) => ({
  theme: getInitialTheme(),
  wsUrl: localStorage.getItem("clowder-ws-url") || "ws://localhost:7070",
  connected: false,

  agents: {},
  tasks: {},
  events: [],

  enabledCategories: new Set<CategoryFilter>(["lifecycle", "task", "other"]),
  selectedMessageTypes: new Set<string>(),
  entityFilters: [],

  selection: null,

  paused: false,

  setTheme: (theme) => {
    localStorage.setItem("clowder-theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
    set({ theme });
  },
  setWsUrl: (url) => {
    localStorage.setItem("clowder-ws-url", url);
    set({ wsUrl: url });
  },
  setConnected: (connected) => set({ connected }),

  applySnapshot: (snapshot) =>
    set({
      agents: snapshot.agents,
      tasks: snapshot.tasks,
      events: (snapshot.events || []).reverse().slice(0, MAX_EVENTS),
    }),

  pushEvent: (event) =>
    set((state) => {
      const events = [event, ...state.events].slice(0, MAX_EVENTS);

      // Update agent state from lifecycle events
      const agents = { ...state.agents };
      if (
        event.message_type === "BusActivateAgentMessage" &&
        event.target
      ) {
        const agent = agents[event.target] || {
          name: event.target,
          parent: null,
          runner: null,
          active: false,
          ready: false,
          bridged: false,
          started_at: null,
        };
        agents[event.target] = { ...agent, active: true };
      } else if (
        event.message_type === "BusDeactivateAgentMessage" &&
        event.target
      ) {
        const agent = agents[event.target];
        if (agent) {
          agents[event.target] = { ...agent, active: false };
        }
      } else if (event.message_type === "BusAgentRegistryMessage") {
        const entries = (event.data.agents as Array<{
          name: string;
          parent?: string;
          active?: boolean;
          bridged?: boolean;
          started_at?: number;
        }>) || [];
        const runner = (event.data.runner as string) || null;
        for (const entry of entries) {
          const agent = agents[entry.name] || {
            name: entry.name,
            parent: null,
            runner: null,
            active: false,
            ready: false,
            bridged: false,
            started_at: null,
          };
          agents[entry.name] = {
            ...agent,
            runner,
            parent: entry.parent || agent.parent,
            active: entry.active ?? agent.active,
            bridged: entry.bridged ?? agent.bridged,
            started_at: entry.started_at || agent.started_at,
            ready: true,
          };
        }
      } else if (event.message_type === "BusAgentReadyMessage") {
        const agent = agents[event.source] || {
          name: event.source,
          parent: null,
          runner: null,
          active: false,
          ready: false,
          bridged: false,
          started_at: null,
        };
        agents[event.source] = {
          ...agent,
          runner: (event.data.runner as string) || agent.runner,
          parent: (event.data.parent as string) || agent.parent,
          active: (event.data.active as boolean) ?? agent.active,
          bridged: (event.data.bridged as boolean) ?? agent.bridged,
          started_at: (event.data.started_at as number) || agent.started_at,
          ready: true,
        };
      } else if (event.message_type === "BusAddAgentMessage") {
        const childName = event.data.agent_name as string;
        if (childName) {
          const agent = agents[childName] || {
            name: childName,
            parent: null,
            runner: null,
            active: false,
            ready: false,
            bridged: false,
          };
          agents[childName] = { ...agent, parent: event.source };
        }
      }

      // Update task state
      const tasks = { ...state.tasks };
      if (event.message_type === "BusTaskRequestMessage" && event.data.task_id) {
        const taskId = event.data.task_id as string;
        const existing = tasks[taskId];
        if (existing) {
          tasks[taskId] = {
            ...existing,
            targets: [...existing.targets, event.target!],
          };
        } else {
          tasks[taskId] = {
            task_id: taskId,
            source: event.source,
            targets: event.target ? [event.target] : [],
            task_name: (event.data.task_name as string) || null,
            status: "running",
            started_at: event.timestamp,
            completed_at: null,
          };
        }
      } else if (
        (event.message_type === "BusTaskResponseMessage" ||
          event.message_type === "BusTaskResponseUrgentMessage") &&
        event.data.task_id
      ) {
        const taskId = event.data.task_id as string;
        const task = tasks[taskId];
        if (task) {
          tasks[taskId] = {
            ...task,
            status: (event.data.status as string) || task.status,
            completed_at: event.timestamp,
          };
        }
      } else if (
        event.message_type === "BusTaskCancelMessage" &&
        event.data.task_id
      ) {
        const taskId = event.data.task_id as string;
        const task = tasks[taskId];
        if (task) {
          tasks[taskId] = {
            ...task,
            status: "cancelled",
            completed_at: event.timestamp,
          };
        }
      }

      return { events, agents, tasks };
    }),

  toggleCategory: (category) =>
    set((state) => {
      const next = new Set(state.enabledCategories);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return { enabledCategories: next };
    }),

  toggleMessageType: (type) =>
    set((state) => {
      const next = new Set(state.selectedMessageTypes);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return { selectedMessageTypes: next };
    }),
  clearMessageTypes: () => set({ selectedMessageTypes: new Set<string>() }),
  addEntityFilter: (filter) =>
    set((state) => {
      const exists = state.entityFilters.some(
        (f) => f.type === filter.type && f.value === filter.value
      );
      if (exists) return state;
      return { entityFilters: [...state.entityFilters, filter] };
    }),
  removeEntityFilter: (filter) =>
    set((state) => ({
      entityFilters: state.entityFilters.filter(
        (f) => !(f.type === filter.type && f.value === filter.value)
      ),
    })),
  clearEntityFilters: () => set({ entityFilters: [] }),
  setSelection: (selection) => set({ selection }),
  setPaused: (paused) => set({ paused }),

  reset: () =>
    set({
      agents: {},
      tasks: {},
      events: [],
      connected: false,
      selectedMessageTypes: new Set<string>(),
      entityFilters: [],
      selection: null,
      paused: false,
    }),
}));

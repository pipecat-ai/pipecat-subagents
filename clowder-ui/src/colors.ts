export const TASK_STATUS_COLORS: Record<string, { bg: string; fg: string }> = {
  running: { bg: "hsla(45, 90%, 45%, 0.12)", fg: "hsl(45, 90%, 45%)" },
  completed: { bg: "hsla(150, 60%, 40%, 0.12)", fg: "hsl(150, 60%, 40%)" },
  failed: { bg: "hsla(0, 65%, 50%, 0.12)", fg: "hsl(0, 65%, 50%)" },
  error: { bg: "hsla(0, 65%, 50%, 0.12)", fg: "hsl(0, 65%, 50%)" },
  cancelled: { bg: "hsla(0, 0%, 50%, 0.1)", fg: "hsl(0, 0%, 50%)" },
};

export const AGENT_STATE_COLORS = {
  active: { bg: "hsla(220, 80%, 55%, 0.12)", fg: "hsl(220, 80%, 55%)" },
  bridged: { bg: "hsla(243, 75%, 59%, 0.12)", fg: "hsl(var(--primary))" },
  idle: { bg: "hsla(0, 0%, 50%, 0.1)", fg: "hsl(0, 0%, 50%)" },
};

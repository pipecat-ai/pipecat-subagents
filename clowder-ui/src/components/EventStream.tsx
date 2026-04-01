import { useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useStore, type CategoryFilter } from "../store";
import type { BusEvent } from "../types";
import {
  Bot,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  ListTodo,
  Server,
  X,
} from "lucide-react";

const CATEGORY_STYLES: Record<string, { bg: string; fg: string }> = {
  lifecycle: { bg: "hsla(220, 80%, 55%, 0.12)", fg: "hsl(220, 80%, 55%)" },
  task: { bg: "hsla(35, 80%, 50%, 0.12)", fg: "hsl(35, 80%, 50%)" },
  other: { bg: "hsla(0, 0%, 50%, 0.1)", fg: "hsl(0, 0%, 50%)" },
};

function CategoryToggle({
  category,
  enabled,
  onToggle,
}: {
  category: CategoryFilter;
  enabled: boolean;
  onToggle: () => void;
}) {
  const style = CATEGORY_STYLES[category] || CATEGORY_STYLES.other;
  return (
    <button
      onClick={onToggle}
      className="px-2 py-0.5 rounded text-xs font-medium transition-opacity"
      style={{
        backgroundColor: style.bg,
        color: style.fg,
        opacity: enabled ? 1 : 0.3,
      }}
    >
      {category}
    </button>
  );
}

function stripBusPrefix(messageType: string): string {
  return messageType.replace(/^Bus/, "").replace(/Message$/, "");
}

function MessageTypeFilter() {
  const { events, selectedMessageTypes, toggleMessageType, clearMessageTypes } =
    useStore();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  const availableTypes = useMemo(() => {
    const types = new Set<string>();
    for (const e of events) {
      types.add(e.message_type);
    }
    return Array.from(types).sort();
  }, [events]);

  const filteredTypes = useMemo(() => {
    const q = search.toLowerCase();
    return availableTypes.filter((t) => t.toLowerCase().includes(q));
  }, [availableTypes, search]);

  const Chevron = open ? ChevronUp : ChevronDown;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-2 py-0.5 text-xs rounded-md border"
        style={{
          backgroundColor: "hsl(var(--background))",
          borderColor: "hsl(var(--input))",
          color: "hsl(var(--foreground))",
        }}
      >
        {selectedMessageTypes.size > 0
          ? `${selectedMessageTypes.size} type${selectedMessageTypes.size !== 1 ? "s" : ""} selected`
          : "All messages"}
        <Chevron size={10} />
      </button>
      {open && (
        <div
          className="absolute top-full left-0 mt-1 z-50 w-64 rounded-md border shadow-lg"
          style={{
            backgroundColor: "hsl(var(--card))",
            borderColor: "hsl(var(--border))",
          }}
        >
          {(selectedMessageTypes.size > 0 ||
            (search && filteredTypes.length > 0)) && (
            <div className="flex gap-2 p-2 border-b" style={{ borderColor: "hsl(var(--border))" }}>
              {search && filteredTypes.length > 0 && (
                <button
                  onClick={() => {
                    const next = new Set(selectedMessageTypes);
                    for (const t of filteredTypes) next.add(t);
                    // Replace the set via individual toggles
                    for (const t of filteredTypes) {
                      if (!selectedMessageTypes.has(t)) toggleMessageType(t);
                    }
                  }}
                  className="flex-1 px-2 py-1 text-xs rounded-md"
                  style={{
                    backgroundColor: "hsl(var(--background))",
                    color: "hsl(var(--foreground))",
                  }}
                >
                  Select all
                </button>
              )}
              {selectedMessageTypes.size > 0 && (
                <button
                  onClick={clearMessageTypes}
                  className="flex-1 px-2 py-1 text-xs rounded-md"
                  style={{
                    backgroundColor: "hsl(var(--background))",
                    color: "hsl(var(--foreground))",
                  }}
                >
                  Clear
                </button>
              )}
            </div>
          )}
          <div className="p-2 border-b" style={{ borderColor: "hsl(var(--border))" }}>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.stopPropagation()}
              placeholder="Search messages..."
              className="w-full px-2 py-1 text-xs rounded-md border outline-none"
              style={{
                backgroundColor: "hsl(var(--background))",
                borderColor: "hsl(var(--input))",
                color: "hsl(var(--foreground))",
              }}
              autoFocus
            />
          </div>
          <div className="max-h-[200px] overflow-auto">
            {filteredTypes.length === 0 ? (
              <div
                className="p-2 text-xs text-center"
                style={{ color: "hsl(var(--muted-foreground))" }}
              >
                No messages found
              </div>
            ) : (
              filteredTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => toggleMessageType(type)}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-left hover:opacity-80"
                >
                  <span
                    className="w-3.5 h-3.5 rounded border flex items-center justify-center shrink-0"
                    style={{
                      borderColor: selectedMessageTypes.has(type)
                        ? "hsl(var(--primary))"
                        : "hsl(var(--input))",
                      backgroundColor: selectedMessageTypes.has(type)
                        ? "hsl(var(--primary))"
                        : "transparent",
                    }}
                  >
                    {selectedMessageTypes.has(type) && (
                      <span className="text-[9px]" style={{ color: "hsl(var(--primary-foreground))" }}>
                        ✓
                      </span>
                    )}
                  </span>
                  <span style={{ color: "hsl(var(--foreground))" }}>
                    {stripBusPrefix(type)}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MessageTypeTags() {
  const { selectedMessageTypes, toggleMessageType } = useStore();

  return (
    <>
      {Array.from(selectedMessageTypes).map((type) => (
        <span
          key={type}
          className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] rounded-md border cursor-pointer hover:opacity-80"
          style={{
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--foreground))",
          }}
          onClick={() => toggleMessageType(type)}
        >
          {stripBusPrefix(type)}
          <X size={8} style={{ color: "hsl(var(--muted-foreground))" }} />
        </span>
      ))}
    </>
  );
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
  } as Intl.DateTimeFormatOptions);
}

function EventRow({ event }: { event: BusEvent }) {
  const [expanded, setExpanded] = useState(false);
  const hasData = Object.keys(event.data).length > 0;
  const catStyle = CATEGORY_STYLES[event.category] || CATEGORY_STYLES.other;

  return (
    <div style={{ borderBottom: "1px solid hsl(var(--border))" }}>
      <button
        onClick={() => hasData && setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-3 py-1.5 text-xs text-left hover:opacity-80"
      >
        <span
          className="shrink-0 w-20 font-mono"
          style={{ color: "hsl(var(--muted-foreground))", opacity: 0.7 }}
        >
          {formatTime(event.timestamp)}
        </span>
        <span
          className="px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0"
          style={{ backgroundColor: catStyle.bg, color: catStyle.fg }}
        >
          {event.category}
        </span>
        <span
          className="font-medium shrink-0"
          style={{ color: "hsl(var(--foreground))" }}
        >
          {stripBusPrefix(event.message_type)}
        </span>
        <span
          className="truncate"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          {event.source}
          {event.target && (
            <>
              <span style={{ opacity: 0.5 }}> → </span>
              {event.target}
            </>
          )}
        </span>
        {hasData && (
          <span
            className="ml-auto shrink-0"
            style={{ color: "hsl(var(--muted-foreground))" }}
          >
            {expanded ? (
              <ChevronDown size={12} />
            ) : (
              <ChevronRight size={12} />
            )}
          </span>
        )}
      </button>
      {expanded && hasData && (
        <pre
          className="px-3 py-2 ml-24 text-[11px] overflow-x-auto"
          style={{
            color: "hsl(var(--muted-foreground))",
            backgroundColor: "hsl(var(--background))",
          }}
        >
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  );
}

function EntityFilterTags() {
  const { entityFilters, removeEntityFilter } = useStore();

  const ENTITY_STYLES: Record<string, { bg: string; fg: string }> = {
    runner: { bg: "hsla(0, 0%, 50%, 0.1)", fg: "hsl(var(--foreground))" },
    agent: { bg: "hsla(220, 80%, 55%, 0.12)", fg: "hsl(220, 80%, 55%)" },
    task: { bg: "hsla(35, 80%, 50%, 0.12)", fg: "hsl(35, 80%, 50%)" },
  };

  return (
    <>
      {entityFilters.map((filter, i) => {
        const style = ENTITY_STYLES[filter.type] || ENTITY_STYLES.runner;
        const label =
          filter.type === "task" ? filter.value.slice(0, 8) : filter.value;
        return (
          <span
            key={`${filter.type}-${filter.value}-${i}`}
            className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] rounded-md cursor-pointer hover:opacity-80"
            style={{ backgroundColor: style.bg, color: style.fg }}
            onClick={() => removeEntityFilter(filter)}
          >
            {filter.type === "runner" && <Server size={8} />}
            {filter.type === "agent" && <Bot size={8} />}
            {filter.type === "task" && <ListTodo size={8} />}
            {label}
            <X size={8} style={{ opacity: 0.6 }} />
          </span>
        );
      })}
    </>
  );
}

export function EventStream() {
  const {
    events,
    enabledCategories,
    toggleCategory,
    selectedMessageTypes,
    entityFilters,
    agents,
  } = useStore();
  const parentRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    // Build sets for quick lookup
    const agentFilters = entityFilters.filter((f) => f.type === "agent").map((f) => f.value);
    const runnerFilters = entityFilters.filter((f) => f.type === "runner").map((f) => f.value);
    const taskFilters = entityFilters.filter((f) => f.type === "task").map((f) => f.value);
    const hasEntityFilters = entityFilters.length > 0;

    // For runner filters, expand to agent names belonging to those runners
    const runnerAgentNames = new Set<string>();
    if (runnerFilters.length > 0) {
      for (const agent of Object.values(agents)) {
        if (agent.runner && runnerFilters.includes(agent.runner)) {
          runnerAgentNames.add(agent.name);
        }
      }
    }

    return events.filter((e) => {
      if (!enabledCategories.has(e.category as CategoryFilter)) return false;
      if (selectedMessageTypes.size > 0 && !selectedMessageTypes.has(e.message_type))
        return false;
      if (hasEntityFilters) {
        const matchesAgent = agentFilters.includes(e.source) || agentFilters.includes(e.target || "");
        const matchesRunner = runnerAgentNames.has(e.source) || runnerAgentNames.has(e.target || "");
        const matchesTask = taskFilters.some((tid) => (e.data.task_id as string) === tid);
        if (!matchesAgent && !matchesRunner && !matchesTask) return false;
      }
      return true;
    });
  }, [events, enabledCategories, selectedMessageTypes, entityFilters, agents]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-col gap-2 px-3.5 py-2 border-b" style={{ borderColor: "hsl(var(--border))" }}>
        <div className="flex items-center gap-2">
          {(["lifecycle", "task", "other"] as CategoryFilter[]).map((cat) => (
            <CategoryToggle
              key={cat}
              category={cat}
              enabled={enabledCategories.has(cat)}
              onToggle={() => toggleCategory(cat)}
            />
          ))}

          <MessageTypeFilter />
          <MessageTypeTags />

          <span
            className="ml-auto text-xs"
            style={{ color: "hsl(var(--muted-foreground))", opacity: 0.7 }}
          >
            {filtered.length === events.length
              ? `${events.length} messages`
              : `${filtered.length} of ${events.length} messages`}
          </span>
        </div>
        {entityFilters.length > 0 && (
          <div className="flex gap-1.5 flex-wrap items-center">
            <EntityFilterTags />
          </div>
        )}
      </div>
      <div ref={parentRef} className="flex-1 overflow-auto">
        {filtered.length === 0 ? (
          <div
            className="p-4 text-sm text-center"
            style={{ color: "hsl(var(--muted-foreground))" }}
          >
            No messages
          </div>
        ) : (
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              position: "relative",
            }}
          >
            {virtualizer.getVirtualItems().map((virtualRow) => (
              <div
                key={virtualRow.key}
                ref={virtualizer.measureElement}
                data-index={virtualRow.index}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <EventRow event={filtered[virtualRow.index]} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

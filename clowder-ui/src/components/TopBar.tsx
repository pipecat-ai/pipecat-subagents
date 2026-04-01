import { useStore } from "../store";
import { useClowderSocket } from "../hooks/useClowderSocket";
import { Link, Link2Off, Moon, Sun, Wifi, WifiOff } from "lucide-react";

function ConnectionStatus({ connected }: { connected: boolean }) {
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
      style={{
        backgroundColor: connected
          ? "hsla(var(--primary), 0.1)"
          : "hsla(var(--destructive), 0.1)",
        color: connected
          ? "hsl(var(--primary))"
          : "hsl(var(--destructive))",
      }}
    >
      {connected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
      {connected ? "Connected" : "Disconnected"}
    </span>
  );
}

export function TopBar() {
  const { wsUrl, setWsUrl, connected, theme, setTheme } = useStore();
  const { connect, disconnect } = useClowderSocket();

  return (
    <div
      className="flex items-center gap-3 px-3.5 py-2.5 border-b flex-nowrap"
      style={{ backgroundColor: "hsl(var(--card))" }}
    >
      <div className="font-bold flex-shrink-0">ᓚᘏᗢ Clowder</div>

      <ConnectionStatus connected={connected} />

      <input
        type="text"
        value={wsUrl}
        onChange={(e) => setWsUrl(e.target.value)}
        disabled={connected}
        placeholder="ws://host:port"
        className="min-w-[220px] max-w-[300px] px-3 py-1.5 text-sm rounded-md border disabled:opacity-50 outline-none focus:ring-2"
        style={{
          backgroundColor: "hsl(var(--background))",
          borderColor: "hsl(var(--input))",
          color: "hsl(var(--foreground))",
        }}
      />

      {connected ? (
        <button
          onClick={disconnect}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md flex-shrink-0"
          style={{
            backgroundColor: "hsl(var(--destructive))",
            color: "hsl(var(--destructive-foreground))",
          }}
        >
          <Link2Off className="h-4 w-4" />
          <span>Disconnect</span>
        </button>
      ) : (
        <button
          onClick={connect}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md flex-shrink-0"
          style={{
            backgroundColor: "hsl(var(--primary))",
            color: "hsl(var(--primary-foreground))",
          }}
        >
          <Link className="h-4 w-4" />
          <span>Connect</span>
        </button>
      )}

      <div className="flex-1" />

      <span className="hidden xl:block text-sm flex-shrink-0"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        Connect any time, messages are buffered.
      </span>

      <button
        onClick={() => setTheme(theme === "light" ? "dark" : "light")}
        className="inline-flex items-center gap-1.5 px-2 py-1.5 text-sm rounded-md flex-shrink-0 hover:opacity-80"
        style={{ color: "hsl(var(--foreground))" }}
      >
        {theme === "light" ? (
          <>
            <Moon className="h-4 w-4" />
            <span className="hidden lg:inline">Dark</span>
          </>
        ) : (
          <>
            <Sun className="h-4 w-4" />
            <span className="hidden lg:inline">Light</span>
          </>
        )}
      </button>
    </div>
  );
}

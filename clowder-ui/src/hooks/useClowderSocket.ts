import { useCallback, useRef } from "react";
import { useStore } from "../store";
import type { ServerMessage } from "../types";

export function useClowderSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const { wsUrl, setConnected, applySnapshot, pushEvent, reset } = useStore();

  const connect = useCallback(() => {
    if (wsRef.current) return;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (e) => {
      try {
        const msg: ServerMessage = JSON.parse(e.data);
        if (msg.type === "snapshot") {
          applySnapshot(msg);
        } else if (msg.type === "bus_message") {
          pushEvent(msg);
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      setConnected(false);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [wsUrl, setConnected, applySnapshot, pushEvent]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    reset();
  }, [reset]);

  return { connect, disconnect };
}

import { useEffect, useRef, useState } from "react";

export type LiveEvent = {
  ts: string;
  event: string;
  [key: string]: unknown;
};

export type ConnectionStatus = "connecting" | "open" | "closed";

/**
 * Subscribe to the /ws/live WebSocket. Reconnects with exponential backoff.
 * Buffered to the last `maxEvents` events so late renders don't OOM.
 */
export function useLiveEvents(maxEvents = 500) {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      setStatus("connecting");
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${window.location.host}/ws/live`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
        setStatus("open");
      };

      ws.onmessage = (msg) => {
        try {
          const evt = JSON.parse(msg.data) as LiveEvent;
          setEvents((prev) => {
            const next = [...prev, evt];
            return next.length > maxEvents
              ? next.slice(next.length - maxEvents)
              : next;
          });
        } catch {
          // ignore malformed frames
        }
      };

      ws.onclose = () => {
        setStatus("closed");
        if (cancelled) return;
        attemptRef.current += 1;
        // Exponential backoff, capped at 15s
        const delay = Math.min(15_000, 500 * 2 ** Math.min(5, attemptRef.current));
        setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [maxEvents]);

  return { events, status };
}

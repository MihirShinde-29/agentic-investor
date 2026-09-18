import { useEffect, useRef, useState } from "react";
import type { LiveEvent, ConnectionStatus } from "@/hooks/useLiveEvents";

/**
 * Subscribe to `/ws/session/{arm_id}/events` for cross-process live-tail
 * of an arm's session.jsonl. Consumes the endpoint added in task #156
 * so an experiment dashboard sees an arm's events without needing to
 * share the arm's in-process EventBus.
 *
 * Same shape / semantics as `useLiveEvents`:
 *   - reconnects on close with exponential backoff (capped)
 *   - buffers the last `maxEvents` (server already hydrates ~200 on
 *     connect, then streams appends)
 *   - reports the connection status so a status pill can badge it
 *
 * Skips the connection entirely when `arm` is null so a single-arm
 * dashboard doesn't leak an idle WS.
 */
export function useSessionEventsWS(
  arm: string | null,
  maxEvents = 500,
): { events: LiveEvent[]; status: ConnectionStatus } {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);

  useEffect(() => {
    if (!arm) {
      // Not in arm mode; nothing to subscribe to.
      setEvents([]);
      setStatus("closed");
      return;
    }

    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      setStatus("connecting");
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url =
        `${proto}//${window.location.host}` +
        `/ws/session/${encodeURIComponent(arm)}/events`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
        setStatus("open");
      };

      ws.onmessage = (msg) => {
        try {
          const evt = JSON.parse(msg.data) as LiveEvent;
          // The server sends a synthetic `_error` frame when it can't
          // find a session dir. Surface it as a closed state instead
          // of an event so the UI's empty state renders correctly.
          if (evt.event === "_error") {
            setStatus("closed");
            return;
          }
          setEvents((prev) => {
            const next = [...prev, evt];
            return next.length > maxEvents
              ? next.slice(next.length - maxEvents)
              : next;
          });
        } catch {
          /* malformed frame - ignore */
        }
      };

      ws.onclose = () => {
        setStatus("closed");
        if (cancelled) return;
        attemptRef.current += 1;
        const delay = Math.min(
          15_000,
          500 * 2 ** Math.min(5, attemptRef.current),
        );
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
  }, [arm, maxEvents]);

  return { events, status };
}

import useSWR from "swr";
import type { LiveEvent, ConnectionStatus } from "@/hooks/useLiveEvents";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { useSessionEventsWS } from "@/hooks/useSessionEventsWS";
import { fetcher, useArmParam } from "@/lib/api";

/**
 * Pick the right event source for the current view.
 *
 * Single-arm mode (no `?arm=` param): use the WebSocket bus from the
 * dashboard's own process — instant, low-latency.
 *
 * Experiment mode with `?arm=X`: the arms are separate subprocesses so
 * the dashboard's in-process bus never sees their events. Prefer the
 * per-arm session-tail WebSocket (`/ws/session/{arm}/events`, task
 * #156) for push semantics; fall back to the 3s SWR poll of
 * `/api/events?arm=X` when the socket hasn't opened yet or drops. The
 * poll keeps the feed alive across a dropped WS reconnect window
 * without waiting on backoff, and provides the initial event list on
 * a fresh mount before the WS hydration frames land.
 */
export function useArmEvents(maxEvents = 500): {
  events: LiveEvent[];
  status: ConnectionStatus;
} {
  const arm = useArmParam();
  const wsLive = useLiveEvents(maxEvents);
  const wsArm = useSessionEventsWS(arm, maxEvents);
  const { data, error } = useSWR<LiveEvent[]>(
    arm ? `/api/events?arm=${encodeURIComponent(arm)}&limit=${maxEvents}` : null,
    fetcher,
    { refreshInterval: 3_000, revalidateOnFocus: false },
  );
  if (!arm) return wsLive;
  // Prefer WS once it's open AND has produced at least one frame; SWR
  // fills the gap up to that point so the initial paint isn't empty.
  if (wsArm.status === "open" && wsArm.events.length > 0) {
    return wsArm;
  }
  return {
    events: data ?? [],
    status: error
      ? "closed"
      : wsArm.status === "connecting"
        ? "connecting"
        : data
          ? "open"
          : "connecting",
  };
}

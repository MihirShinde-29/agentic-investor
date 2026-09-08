import useSWR from "swr";
import type { LiveEvent, ConnectionStatus } from "@/hooks/useLiveEvents";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { fetcher, useArmParam } from "@/lib/api";

/**
 * Pick the right event source for the current view.
 *
 * Single-arm mode (no `?arm=` param): use the WebSocket bus from the
 * dashboard's own process — instant, low-latency.
 *
 * Experiment mode with `?arm=X`: the arms are separate subprocesses so
 * the dashboard's in-process bus never sees their events. Poll the
 * arm's session.jsonl via `/api/events?arm=X` instead. The 3s cadence
 * is a compromise: fresh enough for a live feed, cheap enough that a
 * dozen tabs open won't hammer disk.
 */
export function useArmEvents(maxEvents = 500): {
  events: LiveEvent[];
  status: ConnectionStatus;
} {
  const arm = useArmParam();
  const ws = useLiveEvents(maxEvents);
  const { data, error } = useSWR<LiveEvent[]>(
    arm ? `/api/events?arm=${encodeURIComponent(arm)}&limit=${maxEvents}` : null,
    fetcher,
    { refreshInterval: 3_000, revalidateOnFocus: false },
  );
  if (!arm) return ws;
  return {
    events: data ?? [],
    status: error ? "closed" : data ? "open" : "connecting",
  };
}

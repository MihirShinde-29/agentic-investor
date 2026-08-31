import useSWR from "swr";
import type { LiveEvent } from "@/hooks/useLiveEvents";
import { fetcher } from "@/lib/api";

/**
 * Load a past session's events for the session-picker replay.
 * Returns null when the caller wants live mode instead.
 */
export function useReplayEvents(sessionId: string | "live"): LiveEvent[] | null {
  const { data } = useSWR<LiveEvent[]>(
    sessionId !== "live" ? `/api/session/${sessionId}/events?limit=5000` : null,
    fetcher,
  );
  if (sessionId === "live") return null;
  return data ?? [];
}

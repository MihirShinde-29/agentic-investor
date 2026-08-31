import { useMemo } from "react";
import useSWR from "swr";
import type { LiveEvent } from "@/hooks/useLiveEvents";
import { fetcher } from "@/lib/api";

/**
 * Pull the most recent rec_id from the live event stream. Falls back to
 * paper_orders (which link to rec_id) so a page load without any live
 * regen still surfaces the most recent recommendation.
 */
export function useLatestRecId(events: LiveEvent[]): number | null {
  // Walk backwards through events looking for a regen_done / trade_plan.
  const fromEvents = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      const id = e.rec_id ?? e.last_rec_id;
      if (typeof id === "number") return id;
      if (typeof id === "string" && !isNaN(Number(id))) return Number(id);
    }
    return null;
  }, [events]);

  const { data: trades } = useSWR<Array<{ rec_id: number | null }>>(
    fromEvents == null ? "/api/trades?limit=20" : null,
    fetcher,
  );

  if (fromEvents != null) return fromEvents;
  const fromTrades = trades?.find((t) => t.rec_id != null)?.rec_id;
  return fromTrades ?? null;
}

/** Session-start timestamp from the live event stream (used to scope filter counts). */
export function useSessionStartedAt(events: LiveEvent[]): string | undefined {
  return useMemo(() => {
    for (const e of events) {
      if (e.event === "session_start" || e.event === "state_restored") {
        return e.ts;
      }
    }
    return events[0]?.ts;
  }, [events]);
}

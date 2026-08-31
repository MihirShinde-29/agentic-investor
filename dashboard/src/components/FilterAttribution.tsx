import useSWR from "swr";
import { ShieldCheck } from "lucide-react";
import { fetcher } from "@/lib/api";
import { cn } from "@/lib/utils";

type FilterSkip = {
  id: number;
  skipped_at: string;
  skip_reason: string;
  trigger_reason: string | null;
  avg_drift_pp: number | null;
  max_delta_pp: number | null;
  max_delta_ticker: string | null;
  equity_at_skip: number | null;
};

/**
 * Compact filter-attribution indicator for the top nav.
 *
 * Renders a one-line pill: shield icon + skip count + turnover-pp avoided
 * this session. Full-detail hover tooltip shows per-reason breakdown.
 * Designed to slot next to the LIVE status pill without taking a header row.
 */
export function FilterAttribution({ sessionStartedAt }: { sessionStartedAt?: string }) {
  const { data } = useSWR<FilterSkip[]>(
    "/api/filter-skips?limit=200",
    fetcher,
    { refreshInterval: 30_000 },
  );

  const sessionSkips = (data ?? []).filter((s) => {
    if (!sessionStartedAt) return true;
    return new Date(s.skipped_at) >= new Date(sessionStartedAt);
  });

  const totalPp = sessionSkips.reduce(
    (sum, s) => sum + Math.abs(s.max_delta_pp ?? 0),
    0,
  );
  const byReason = sessionSkips.reduce<Record<string, number>>((acc, s) => {
    acc[s.skip_reason] = (acc[s.skip_reason] ?? 0) + 1;
    return acc;
  }, {});
  const total = sessionSkips.length;

  const tooltip =
    total === 0
      ? "no filter skips yet this session"
      : Object.entries(byReason)
          .map(([r, n]) => `${r}: ${n}`)
          .join(" · ");

  const active = total > 0;

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ring-1 ring-inset",
        active
          ? "bg-warning/15 text-warning ring-warning/30"
          : "bg-muted/30 text-muted-foreground ring-border/40",
      )}
      title={tooltip}
    >
      <ShieldCheck className="size-3.5" />
      <span className="tabular font-semibold">{total}</span>
      <span className="text-[10px] uppercase tracking-wider opacity-70">
        skip{total === 1 ? "" : "s"}
      </span>
      {totalPp > 0 && (
        <>
          <span className="opacity-40">·</span>
          <span className="tabular">{totalPp.toFixed(0)}pp saved</span>
        </>
      )}
    </div>
  );
}

import useSWR from "swr";
import { ShieldCheck } from "lucide-react";
import { fetcher } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

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

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <div className="flex items-center gap-1.5 rounded-md border border-border/40 bg-card/60 px-2 py-1 text-xs">
        <ShieldCheck className="size-3.5 text-primary" />
        <span className="text-muted-foreground">Filter skips:</span>
        <span className="tabular font-semibold">{total}</span>
      </div>
      {totalPp > 0 && (
        <Badge variant="muted">
          ~{totalPp.toFixed(0)}pp turnover avoided
        </Badge>
      )}
      {Object.entries(byReason).map(([reason, n]) => (
        <Badge key={reason} variant="warning">
          {reason}: {n}
        </Badge>
      ))}
    </div>
  );
}

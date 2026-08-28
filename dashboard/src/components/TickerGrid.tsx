import useSWR from "swr";
import type { PositionResp, TradeResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import { TickerCard } from "@/components/TickerCard";

export function TickerGrid() {
  const { data: positions } = useSWR<PositionResp[]>(
    "/api/positions",
    fetcher,
    { refreshInterval: 30_000 },
  );
  const { data: trades } = useSWR<TradeResp[]>(
    "/api/trades?limit=200",
    fetcher,
    { refreshInterval: 30_000 },
  );

  if (!positions || positions.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border/40 bg-card/30 p-8 text-center text-sm text-muted-foreground">
        no open positions to chart
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
      {positions.map((p) => (
        <TickerCard key={p.ticker} position={p} trades={trades ?? []} />
      ))}
    </div>
  );
}

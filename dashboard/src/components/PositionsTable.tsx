import { useState } from "react";
import useSWR from "swr";
import { ArrowDown, ArrowUp } from "lucide-react";
import type { PortfolioResp, PositionResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TradeDrillDown } from "@/components/TradeDrillDown";
import { cn, formatPct, formatUSD } from "@/lib/utils";

type SortKey =
  | "ticker"
  | "qty"
  | "avg_entry_price"
  | "market_value"
  | "unrealized_pl"
  | "unrealized_pl_pct"
  | "weight";

export function PositionsTable({ recId }: { recId: number | null }) {
  const { data: positions } = useSWR<PositionResp[]>(
    "/api/positions",
    fetcher,
    { refreshInterval: 15_000 },
  );
  const { data: portfolio } = useSWR<PortfolioResp>("/api/portfolio", fetcher, {
    refreshInterval: 15_000,
  });

  const [sortKey, setSortKey] = useState<SortKey>("market_value");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [drillTicker, setDrillTicker] = useState<string | null>(null);

  const equity = portfolio?.equity ?? 0;
  const rows = (positions ?? []).map((p) => ({
    ...p,
    weight: equity > 0 ? (p.market_value / equity) * 100 : 0,
  }));

  rows.sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === "string" && typeof bv === "string") {
      return sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
    }
    const na = Number(av);
    const nb = Number(bv);
    return sortDir === "asc" ? na - nb : nb - na;
  });

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const SortHeader = ({ label, k, align = "right" }: { label: string; k: SortKey; align?: "left" | "right" }) => (
    <th
      className={cn(
        "cursor-pointer select-none px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground",
        align === "right" ? "text-right" : "text-left",
      )}
      onClick={() => toggleSort(k)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === k &&
          (sortDir === "desc" ? (
            <ArrowDown className="size-3" />
          ) : (
            <ArrowUp className="size-3" />
          ))}
      </span>
    </th>
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Positions</CardTitle>
        <span className="text-xs text-muted-foreground">
          {rows.length} open
        </span>
      </CardHeader>
      <CardContent className="p-0">
        {rows.length === 0 ? (
          <div className="p-8 text-center text-sm text-muted-foreground">
            no open positions
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-border/40 bg-muted/20">
                <tr>
                  <SortHeader label="Ticker" k="ticker" align="left" />
                  <SortHeader label="Qty" k="qty" />
                  <SortHeader label="Entry" k="avg_entry_price" />
                  <SortHeader label="Mkt value" k="market_value" />
                  <SortHeader label="Unrealized $" k="unrealized_pl" />
                  <SortHeader label="P&L %" k="unrealized_pl_pct" />
                  <SortHeader label="Weight %" k="weight" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const gain = r.unrealized_pl >= 0;
                  return (
                    <tr
                      key={r.ticker}
                      className="cursor-pointer border-b border-border/30 last:border-b-0 transition-colors hover:bg-primary/5"
                      onClick={() => setDrillTicker(r.ticker)}
                      title={recId != null ? `Click for rec #${recId} rationale` : ""}
                    >
                      <td className="px-3 py-2">
                        <Badge variant="primary" className="font-mono">
                          {r.ticker}
                        </Badge>
                      </td>
                      <td className="px-3 py-2 text-right tabular">
                        {r.qty.toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-right tabular text-muted-foreground">
                        {formatUSD(r.avg_entry_price)}
                      </td>
                      <td className="px-3 py-2 text-right tabular font-medium">
                        {formatUSD(r.market_value)}
                      </td>
                      <td
                        className={cn(
                          "px-3 py-2 text-right tabular",
                          gain ? "text-success" : "text-danger",
                        )}
                      >
                        {formatUSD(r.unrealized_pl)}
                      </td>
                      <td
                        className={cn(
                          "px-3 py-2 text-right tabular",
                          gain ? "text-success" : "text-danger",
                        )}
                      >
                        {formatPct(r.unrealized_pl_pct)}
                      </td>
                      <td className="px-3 py-2 text-right tabular">
                        {r.weight.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
      <TradeDrillDown
        open={drillTicker != null}
        onOpenChange={(v) => !v && setDrillTicker(null)}
        recId={recId}
        focusTicker={drillTicker ?? undefined}
      />
    </Card>
  );
}

import useSWR from "swr";
import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BarsResp, SnapshotResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn, formatPct } from "@/lib/utils";

type Point = { ts: number; portfolio: number | null; spy: number | null };

type Timeframe = "1D" | "1M" | "3M" | "1Y";

// Match TickerCard: 1D uses 1-minute bars so the intraday chart fills up
// throughout the session instead of showing just a few 5-min bars at open.
const TIMEFRAMES: Record<Timeframe, { period: string; interval: string; refreshMs: number }> = {
  "1D": { period: "1d",  interval: "1m", refreshMs: 30_000 },
  "1M": { period: "1mo", interval: "1h", refreshMs: 5 * 60_000 },
  "3M": { period: "3mo", interval: "1d", refreshMs: 15 * 60_000 },
  "1Y": { period: "1y",  interval: "1d", refreshMs: 30 * 60_000 },
};

export function PortfolioChart({ sessionId }: { sessionId?: string }) {
  const [timeframe, setTimeframe] = useState<Timeframe>("1D");
  const tf = TIMEFRAMES[timeframe];
  // Session picker overrides the timeframe: replay always shows the session window.
  const snapshotsUrl = sessionId
    ? `/api/snapshots?limit=2000&session=${encodeURIComponent(sessionId)}`
    : `/api/snapshots?limit=2000&period=${tf.period}`;
  const spyUrl = sessionId
    ? `/api/bars/SPY?session=${encodeURIComponent(sessionId)}`
    : `/api/bars/SPY?period=${tf.period}&interval=${tf.interval}`;
  const { data: snaps } = useSWR<SnapshotResp[]>(
    snapshotsUrl,
    fetcher,
    { refreshInterval: sessionId ? 0 : tf.refreshMs },
  );
  const { data: spy } = useSWR<BarsResp>(
    spyUrl,
    fetcher,
    { refreshInterval: sessionId ? 0 : tf.refreshMs },
  );
  // Live-tail: poll current equity every 10s and append a temporary point
  // at the chart tip so the line moves between snapshot writes (which only
  // happen ~1 per loop tick, ~3-5 min apart).
  const { data: livePortfolio } = useSWR<{ equity: number } | null>(
    sessionId ? null : "/api/portfolio",
    fetcher,
    { refreshInterval: sessionId ? 0 : 10_000 },
  );

  const { data, portfolioReturn, spyReturn, alpha } = useMemo(() => {
    if (!snaps || snaps.length === 0) {
      return { data: [] as Point[], portfolioReturn: 0, spyReturn: 0, alpha: 0 };
    }
    const openEq = snaps[0].equity;
    const portfolioSeries = snaps.map((s) => ({
      ts: new Date(s.ts).getTime(),
      pct: openEq > 0 ? (s.equity / openEq - 1) * 100 : 0,
    }));
    const spyBars = spy?.bars ?? [];
    const spyOpen = spyBars.length > 0 ? spyBars[0].c : 0;
    const spySeries = spyBars.map((b) => ({
      ts: new Date(b.t).getTime(),
      pct: spyOpen > 0 ? (b.c / spyOpen - 1) * 100 : 0,
    }));

    // Union timeline sorted; each timestamp gets nearest-past value from each series.
    const allTs = Array.from(
      new Set([...portfolioSeries.map((p) => p.ts), ...spySeries.map((p) => p.ts)]),
    ).sort((a, b) => a - b);

    let pIdx = 0;
    let sIdx = 0;
    const points: Point[] = [];
    for (const ts of allTs) {
      while (pIdx + 1 < portfolioSeries.length && portfolioSeries[pIdx + 1].ts <= ts)
        pIdx++;
      while (sIdx + 1 < spySeries.length && spySeries[sIdx + 1].ts <= ts) sIdx++;
      points.push({
        ts,
        portfolio: portfolioSeries[pIdx]?.pct ?? null,
        spy: spySeries[sIdx]?.pct ?? null,
      });
    }

    // Live-tail: if we have a fresh /api/portfolio equity newer than the
    // last snapshot, append it as a temporary chart tip. Uses the last
    // known SPY value so the two lines stay aligned in time.
    const livePct = livePortfolio && openEq > 0
      ? (livePortfolio.equity / openEq - 1) * 100
      : null;
    if (livePct !== null && points.length > 0) {
      const nowTs = Date.now();
      const lastTs = points[points.length - 1].ts;
      if (nowTs > lastTs) {
        points.push({
          ts: nowTs,
          portfolio: livePct,
          spy: points[points.length - 1].spy,
        });
      }
    }

    const pRet = livePct ?? portfolioSeries[portfolioSeries.length - 1]?.pct ?? 0;
    const sRet = spySeries[spySeries.length - 1]?.pct ?? 0;
    return { data: points, portfolioReturn: pRet, spyReturn: sRet, alpha: pRet - sRet };
  }, [snaps, spy, livePortfolio]);

  const alphaVariant =
    alpha > 0.05 ? "success" : alpha < -0.05 ? "danger" : "muted";

  return (
    <Card>
      <CardHeader className="flex-col items-stretch gap-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <CardTitle>Portfolio vs SPY</CardTitle>
            <span className="text-xs text-muted-foreground">
              {sessionId ? "session return" : `${timeframe} return`} · rebased to start
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <Badge variant={portfolioReturn > 0 ? "success" : portfolioReturn < 0 ? "danger" : "muted"}>
              Portfolio {formatPct(portfolioReturn)}
            </Badge>
            <Badge variant="muted">SPY {formatPct(spyReturn)}</Badge>
            <Badge variant={alphaVariant}>Alpha {formatPct(alpha)}</Badge>
          </div>
        </div>
        {/* Timeframe pills; hidden in session-replay mode since the window is fixed. */}
        {!sessionId && (
          <div className="flex items-center justify-end gap-0.5">
            {(Object.keys(TIMEFRAMES) as Timeframe[]).map((tfKey) => (
              <button
                key={tfKey}
                type="button"
                onClick={() => setTimeframe(tfKey)}
                className={cn(
                  "rounded-md px-1.5 py-0.5 text-[10px] font-medium tabular transition-colors",
                  timeframe === tfKey
                    ? "bg-primary/15 text-primary ring-1 ring-primary/30"
                    : "text-muted-foreground hover:bg-muted/40 hover:text-foreground",
                )}
              >
                {tfKey}
              </button>
            ))}
          </div>
        )}
      </CardHeader>
      <CardContent className="p-2">
        {data.length < 2 ? (
          <div className="grid h-64 place-items-center text-sm text-muted-foreground">
            waiting for snapshots (loop writes one per tick)
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
              <defs>
                <linearGradient id="portfolioGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(217 92% 60%)" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="hsl(217 92% 60%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="hsl(240 3.7% 15.9%)" strokeDasharray="3 3" />
              <XAxis
                dataKey="ts"
                type="number"
                domain={["dataMin", "dataMax"]}
                scale="time"
                tickFormatter={(v) =>
                  new Date(v).toLocaleTimeString(undefined, {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                }
                tick={{ fill: "hsl(240 5% 64.9%)", fontSize: 11 }}
                stroke="hsl(240 3.7% 20%)"
              />
              <YAxis
                tickFormatter={(v) => `${v.toFixed(2)}%`}
                tick={{ fill: "hsl(240 5% 64.9%)", fontSize: 11 }}
                stroke="hsl(240 3.7% 20%)"
                width={55}
              />
              <Tooltip
                contentStyle={{
                  background: "hsl(240 10% 5.5%)",
                  border: "1px solid hsl(240 3.7% 15.9%)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelFormatter={(v) =>
                  new Date(Number(v)).toLocaleTimeString()
                }
                formatter={(value, name) => [
                  `${Number(value ?? 0).toFixed(3)}%`,
                  String(name),
                ]}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Area
                type="monotone"
                dataKey="portfolio"
                name="Portfolio"
                stroke="hsl(217 92% 60%)"
                fill="url(#portfolioGrad)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              <Area
                type="monotone"
                dataKey="spy"
                name="SPY"
                stroke="hsl(240 5% 64.9%)"
                fill="transparent"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                dot={false}
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

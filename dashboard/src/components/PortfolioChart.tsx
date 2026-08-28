import useSWR from "swr";
import { useMemo } from "react";
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
import { formatPct } from "@/lib/utils";

type Point = { ts: number; portfolio: number | null; spy: number | null };

export function PortfolioChart() {
  const { data: snaps } = useSWR<SnapshotResp[]>(
    "/api/snapshots?limit=500",
    fetcher,
    { refreshInterval: 30_000 },
  );
  const { data: spy } = useSWR<BarsResp>(
    "/api/bars/SPY?period=1d&interval=5m",
    fetcher,
    { refreshInterval: 60_000 },
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

    const pRet = portfolioSeries[portfolioSeries.length - 1]?.pct ?? 0;
    const sRet = spySeries[spySeries.length - 1]?.pct ?? 0;
    return { data: points, portfolioReturn: pRet, spyReturn: sRet, alpha: pRet - sRet };
  }, [snaps, spy]);

  const alphaVariant =
    alpha > 0.05 ? "success" : alpha < -0.05 ? "danger" : "muted";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardTitle>Portfolio vs SPY</CardTitle>
          <span className="text-xs text-muted-foreground">
            session return · rebased to open
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <Badge variant={portfolioReturn > 0 ? "success" : portfolioReturn < 0 ? "danger" : "muted"}>
            Portfolio {formatPct(portfolioReturn)}
          </Badge>
          <Badge variant="muted">SPY {formatPct(spyReturn)}</Badge>
          <Badge variant={alphaVariant}>Alpha {formatPct(alpha)}</Badge>
        </div>
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

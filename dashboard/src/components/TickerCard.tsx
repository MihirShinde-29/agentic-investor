import { useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import {
  AreaSeries,
  createChart,
  createSeriesMarkers,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { BarsResp, PositionResp, TradeResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TradeDrillDown } from "@/components/TradeDrillDown";
import { cn, formatPct, formatUSD } from "@/lib/utils";

export function TickerCard({
  position,
  trades,
  recId,
}: {
  position: PositionResp;
  trades: TradeResp[];
  recId: number | null;
}) {
  const [drillOpen, setDrillOpen] = useState(false);
  const { data: bars } = useSWR<BarsResp>(
    `/api/bars/${position.ticker}?period=1d&interval=5m`,
    fetcher,
    { refreshInterval: 60_000 },
  );
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const priceSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const smaSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  const tickerTrades = useMemo(
    () =>
      trades.filter(
        (t) => t.ticker.toUpperCase() === position.ticker.toUpperCase(),
      ),
    [trades, position.ticker],
  );

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "hsl(240 5% 64.9%)",
        fontFamily: "Inter, sans-serif",
      },
      grid: {
        vertLines: { color: "hsl(240 3.7% 12%)" },
        horzLines: { color: "hsl(240 3.7% 12%)" },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      autoSize: true,
      handleScroll: false,
      handleScale: false,
    });
    const priceSeries = chart.addSeries(AreaSeries, {
      lineColor: "hsl(217 92% 60%)",
      topColor: "hsla(217, 92%, 60%, 0.3)",
      bottomColor: "hsla(217, 92%, 60%, 0)",
      lineWidth: 2,
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });
    const smaSeries = chart.addSeries(LineSeries, {
      color: "hsl(38 92% 55%)",
      lineWidth: 1,
      lineStyle: 2, // dashed
      priceLineVisible: false,
      lastValueVisible: false,
    });
    chartRef.current = chart;
    priceSeriesRef.current = priceSeries;
    smaSeriesRef.current = smaSeries;
    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const price = priceSeriesRef.current;
    const sma = smaSeriesRef.current;
    const chart = chartRef.current;
    if (!price || !sma || !chart || !bars?.bars?.length) return;

    const closeData = bars.bars.map((b) => ({
      time: Math.floor(new Date(b.t).getTime() / 1000) as UTCTimestamp,
      value: b.c,
    }));
    const smaData = bars.bars
      .filter((b) => b.sma20 != null)
      .map((b) => ({
        time: Math.floor(new Date(b.t).getTime() / 1000) as UTCTimestamp,
        value: b.sma20 as number,
      }));
    price.setData(closeData);
    sma.setData(smaData);

    // Buy/Sell markers via v5 primitive. Two rules to keep the chart
    // readable when many historical trades are on file:
    // 1. Only show markers within the visible bar range (no orphan arrows
    //    at chart edges from trades outside the fetched window).
    // 2. No text labels - the shape + color convey side; hover the arrow
    //    for the exact fill price/time via lightweight-charts' own tooltip.
    const firstBarTs = closeData[0]?.time as number | undefined;
    const lastBarTs = closeData[closeData.length - 1]?.time as number | undefined;
    const markers = tickerTrades
      .filter((t) => t.filled_avg_price != null)
      .map((t) => {
        const ts = Math.floor(new Date(t.submitted_at).getTime() / 1000);
        return { trade: t, ts };
      })
      .filter(({ ts }) =>
        firstBarTs == null || lastBarTs == null
          ? true
          : ts >= firstBarTs && ts <= lastBarTs,
      )
      .map(({ trade, ts }) => ({
        time: ts as UTCTimestamp,
        position: (trade.side === "buy" ? "belowBar" : "aboveBar") as
          | "belowBar"
          | "aboveBar",
        color: trade.side === "buy" ? "hsl(142 71% 45%)" : "hsl(0 72% 55%)",
        shape: (trade.side === "buy" ? "arrowUp" : "arrowDown") as
          | "arrowUp"
          | "arrowDown",
        size: 1,
      }))
      .sort((a, b) => (a.time as number) - (b.time as number));
    createSeriesMarkers(price, markers);
    chart.timeScale().fitContent();
  }, [bars, tickerTrades]);

  const gain = position.unrealized_pl >= 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setDrillOpen(true)}
            className="inline-flex items-center rounded-md focus:outline-none focus:ring-2 focus:ring-primary/40"
            title="Show why this position was sized this way"
          >
            <Badge variant="primary" className="cursor-pointer font-mono hover:brightness-110">
              {position.ticker}
            </Badge>
          </button>
          <CardTitle className="text-muted-foreground">
            {position.qty.toFixed(4)} sh · entry {formatUSD(position.avg_entry_price)}
          </CardTitle>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className={cn(
              "text-xs font-medium tabular",
              gain ? "text-success" : "text-danger",
            )}
          >
            {formatUSD(position.unrealized_pl)} ({formatPct(position.unrealized_pl_pct)})
          </span>
        </div>
      </CardHeader>
      <CardContent className="p-2">
        <div
          ref={containerRef}
          className="h-52 w-full"
          aria-label={`${position.ticker} chart`}
        />
      </CardContent>
      <TradeDrillDown
        open={drillOpen}
        onOpenChange={setDrillOpen}
        recId={recId}
        focusTicker={position.ticker}
      />
    </Card>
  );
}

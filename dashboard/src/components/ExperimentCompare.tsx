import useSWR from "swr";
import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  CompareEquityResp,
  CompareSummaryResp,
  NewsReactionsResp,
} from "@/lib/api";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { TIMEFRAMES, type Timeframe } from "@/lib/timeframe";

const ARM_COLORS = ["#60a5fa", "#f472b6", "#a78bfa", "#fbbf24", "#34d399"];

function fmtUsd(n: number | undefined): string {
  if (n === undefined || Number.isNaN(n)) return "-";
  return `$${n.toLocaleString(undefined, {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  })}`;
}

export function ExperimentCompare({ timeframe }: { timeframe: Timeframe }) {
  const tf = TIMEFRAMES[timeframe];
  const { data: summary } = useSWR<CompareSummaryResp>(
    "/api/experiment/compare/summary",
    fetcher,
    // 20s: each poll fires 3 arms x ~4 Alpaca calls = 12 requests.
    // At 10s we tripped rate limits during live session. 20s stays
    // well under and still feels live for a compare view.
    { refreshInterval: 20_000 },
  );
  const { data: equity } = useSWR<CompareEquityResp>(
    `/api/experiment/compare/equity?period=${tf.period}`,
    fetcher,
    { refreshInterval: 15_000 },
  );
  const { data: reactions } = useSWR<NewsReactionsResp>(
    "/api/experiment/compare/news-reactions?limit=30",
    fetcher,
    { refreshInterval: 15_000 },
  );

  // Map arm_id -> current live equity from summary. Used to append a
  // "now" tick to each arm's series so the line renders even with only
  // one persisted snapshot (typical for the first few minutes after a
  // session start).
  const liveByArm = useMemo(() => {
    const m: Record<string, { equity: number; open: number | null }> = {};
    for (const r of summary?.arms ?? []) {
      if (r.equity !== undefined) {
        m[r.arm_id] = {
          equity: r.equity,
          open: r.opening_equity ?? null,
        };
      }
    }
    return m;
  }, [summary]);

  const chartData = useMemo(() => {
    if (!equity || equity.arms.length === 0) return [];
    const now = Date.now();
    // Normalize each arm to % change from its own baseline. Baseline is
    // opening_equity from summary (session-open) if available, else the
    // arm's first snapshot in the window. Always append a synthetic
    // "now" point from the live equity so 1-snapshot arms still render.
    const perArmNormalized: Record<string, { ts: number; pct: number }[]> = {};
    for (const arm of equity.arms) {
      const live = liveByArm[arm.arm_id];
      const baseline =
        live?.open ?? (arm.points.length > 0 ? arm.points[0].equity : null);
      if (baseline === null || baseline <= 0) {
        perArmNormalized[arm.arm_id] = [];
        continue;
      }
      const series = arm.points.map((p) => ({
        ts: new Date(p.ts).getTime(),
        pct: (p.equity / baseline - 1) * 100,
      }));
      if (live) {
        const last = series[series.length - 1];
        const livePct = (live.equity / baseline - 1) * 100;
        // Skip if a snapshot within the last 5s already reflects it
        // (avoids a duplicate near-identical point).
        if (!last || now - last.ts > 5000) {
          series.push({ ts: now, pct: livePct });
        }
      }
      perArmNormalized[arm.arm_id] = series;
    }
    const allTs = Array.from(new Set(
      Object.values(perArmNormalized).flatMap((s) => s.map((p) => p.ts)),
    )).sort((a, b) => a - b);
    return allTs.map((ts) => {
      const row: Record<string, number | string | null> = { ts };
      for (const [armId, series] of Object.entries(perArmNormalized)) {
        let val: number | null = null;
        for (const p of series) {
          if (p.ts <= ts) val = p.pct;
          else break;
        }
        row[armId] = val;
      }
      return row;
    });
  }, [equity, liveByArm]);

  const armIds = equity?.arms.map((a) => a.arm_id) ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            Experiment comparison
            {summary ? (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                {summary.experiment} · {summary.arms.length} arms
              </span>
            ) : null}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke="#22252b" strokeDasharray="3 3" />
                <XAxis
                  dataKey="ts"
                  type="number"
                  domain={["auto", "auto"]}
                  scale="time"
                  tickFormatter={(t) =>
                    new Date(t as number).toLocaleTimeString([], {
                      hour: "2-digit", minute: "2-digit",
                    })
                  }
                  stroke="#64748b"
                  fontSize={11}
                />
                <YAxis
                  tickFormatter={(v) => `${(v as number).toFixed(2)}%`}
                  stroke="#64748b"
                  fontSize={11}
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  labelFormatter={(t) => new Date(t as number).toLocaleString()}
                  formatter={(v) =>
                    v === null
                      ? "-"
                      : `${(v as number).toFixed(3)}%`
                  }
                  contentStyle={{
                    background: "#0b0d10",
                    border: "1px solid #22252b",
                    fontSize: 12,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {armIds.map((id, i) => (
                  <Line
                    key={id}
                    type="monotone"
                    dataKey={id}
                    stroke={ARM_COLORS[i % ARM_COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                    name={`arm ${id}`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            % change from each arm's first snapshot in the window. Same news
            stream + same market prices flow to all arms; the spread is
            purely the effect of the config diff.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Per-arm summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs uppercase text-muted-foreground">
                <tr className="border-b border-border/60">
                  <th className="py-2 text-left">arm</th>
                  <th className="py-2 text-left">account</th>
                  <th className="py-2 text-right">equity</th>
                  <th className="py-2 text-right">Δ $</th>
                  <th className="py-2 text-right">Δ %</th>
                  <th className="py-2 text-right">cash %</th>
                  <th className="py-2 text-right">pos</th>
                  <th className="py-2 text-right">orders</th>
                  <th className="py-2 text-right">turnover $</th>
                  <th className="py-2 text-left">held</th>
                </tr>
              </thead>
              <tbody>
                {(summary?.arms ?? []).map((row, i) => {
                  const dPos = row.delta_dollars !== undefined && row.delta_dollars >= 0;
                  return (
                  <tr key={row.arm_id} className="border-b border-border/40">
                    <td className="py-2 font-medium">
                      <span
                        className={cn(
                          "inline-block size-2 rounded-full",
                        )}
                        style={{
                          backgroundColor: ARM_COLORS[i % ARM_COLORS.length],
                        }}
                      />
                      <span className="ml-2">{row.arm_id}</span>
                    </td>
                    <td className="py-2 text-muted-foreground">{row.account}</td>
                    <td className="py-2 text-right tabular-nums">
                      {fmtUsd(row.equity)}
                    </td>
                    <td
                      className={cn(
                        "py-2 text-right tabular-nums",
                        row.delta_dollars === undefined
                          ? "text-muted-foreground"
                          : dPos ? "text-success" : "text-danger",
                      )}
                    >
                      {row.delta_dollars === undefined
                        ? "-"
                        : `${dPos ? "+" : ""}${fmtUsd(row.delta_dollars)}`}
                    </td>
                    <td
                      className={cn(
                        "py-2 text-right tabular-nums",
                        row.delta_pct === undefined
                          ? "text-muted-foreground"
                          : dPos ? "text-success" : "text-danger",
                      )}
                    >
                      {row.delta_pct === undefined
                        ? "-"
                        : `${dPos ? "+" : ""}${row.delta_pct.toFixed(3)}%`}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {row.cash_pct === undefined
                        ? "-"
                        : `${row.cash_pct.toFixed(1)}%`}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {row.positions_count ?? 0}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {row.n_orders ?? 0}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {fmtUsd(row.turnover)}
                    </td>
                    <td className="py-2 text-xs">
                      <div className="flex flex-wrap gap-1">
                        {(row.positions ?? []).map((p) => (
                          <span
                            key={p.ticker}
                            className={cn(
                              "rounded bg-muted/50 px-1.5 py-0.5 font-medium tabular-nums",
                              p.unrealized_pl_pct >= 0
                                ? "text-success"
                                : "text-danger",
                            )}
                            title={`${p.ticker}: ${fmtUsd(p.market_value)} (${p.unrealized_pl_pct >= 0 ? "+" : ""}${p.unrealized_pl_pct.toFixed(2)}%)`}
                          >
                            {p.ticker}
                          </span>
                        ))}
                        {(row.positions ?? []).length === 0 ? (
                          <span className="text-muted-foreground">-</span>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
            {(summary?.arms ?? []).length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No arm data yet — waiting for the first snapshot from each arm.
              </p>
            ) : null}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            News reactions
            {reactions?.news_reactions ? (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                last {reactions.news_reactions.length} · reactivity window
                5 min
              </span>
            ) : null}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-[420px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card text-[10px] uppercase text-muted-foreground">
                <tr className="border-b border-border/60">
                  <th className="py-2 text-left">time</th>
                  <th className="py-2 text-left">ticker</th>
                  <th className="py-2 text-left">headline</th>
                  {armIds.map((id) => (
                    <th key={id} className="py-2 text-center">
                      {id}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(reactions?.news_reactions ?? [])
                  .filter((row) =>
                    // Hide rows where every arm was either "no reaction"
                    // or "skip" — those add noise without showing any
                    // divergence. Only rows with at least one real regen
                    // (with orders) are worth surfacing.
                    Object.values(row.per_arm).some(
                      (r) => r?.reacted === "regen" || r?.reacted === "orders",
                    ),
                  )
                  .map((row) => (
                  <tr
                    key={`${row.ts}-${row.ticker ?? ""}`}
                    className="border-b border-border/40"
                  >
                    <td className="py-1.5 pr-2 text-muted-foreground tabular-nums">
                      {new Date(row.ts).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </td>
                    <td className="py-1.5 pr-2 font-medium">
                      {row.ticker ?? "-"}
                    </td>
                    <td
                      className="max-w-[380px] truncate py-1.5 pr-2 text-muted-foreground"
                      title={row.headline}
                    >
                      {row.headline}
                    </td>
                    {armIds.map((id) => {
                      const r = row.per_arm[id];
                      if (!r || r.reacted === "none") {
                        return (
                          <td
                            key={id}
                            className="py-1.5 text-center text-muted-foreground/50"
                            title="no reaction within 5 min"
                          >
                            —
                          </td>
                        );
                      }
                      if (r.reacted === "skip") {
                        return (
                          <td key={id} className="py-1.5 text-center">
                            <span
                              className="rounded bg-muted/40 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground"
                              title={`${r.kind} in ${r.seconds}s`}
                            >
                              skip
                            </span>
                          </td>
                        );
                      }
                      // regen or orders — show the actual trades.
                      const orders = r.orders ?? [];
                      const regen = r.regen;
                      const hoverTitle = regen
                        ? `regen #${regen.rec_id} in ${regen.seconds}s · ${regen.trigger ?? "?"} · ${regen.targets_count} targets · cash ${regen.cash_pct ?? "?"}%`
                        : "orders within 5 min (no matching regen)";
                      return (
                        <td key={id} className="py-1.5 px-1 align-top">
                          <div
                            className="flex flex-col items-center gap-0.5"
                            title={hoverTitle}
                          >
                            {orders.length === 0 ? (
                              <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary/80 ring-1 ring-primary/20">
                                regen · no trade
                              </span>
                            ) : (
                              orders.map((o, i) => {
                                const isBuy = o.side === "buy";
                                return (
                                  <span
                                    key={`${o.ticker}-${o.side}-${i}`}
                                    className={cn(
                                      "flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium tabular-nums",
                                      isBuy
                                        ? "bg-success/15 text-success ring-1 ring-success/30"
                                        : "bg-danger/15 text-danger ring-1 ring-danger/30",
                                    )}
                                  >
                                    <span>{isBuy ? "↑" : "↓"}</span>
                                    <span>{o.ticker}</span>
                                    <span className="opacity-70">
                                      {o.qty.toFixed(2)}
                                    </span>
                                  </span>
                                );
                              })
                            )}
                            {regen ? (
                              <span className="text-[9px] text-muted-foreground/70 tabular-nums">
                                {regen.seconds < 60
                                  ? `${regen.seconds.toFixed(0)}s`
                                  : `${(regen.seconds / 60).toFixed(1)}m`}
                              </span>
                            ) : null}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            {(reactions?.news_reactions ?? []).length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                Waiting for news events + arm reactions.
              </p>
            ) : null}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Each cell shows what the arm actually did in the 5 min after
            the news: ↑ green = buy, ↓ red = sell, with ticker + share
            qty. "regen · no trade" = model re-thought but held. "skip"
            = classified non-material. "—" = no reaction. Hover for
            regen id / trigger / cash%.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

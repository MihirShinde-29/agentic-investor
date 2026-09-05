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
import type { CompareEquityResp, CompareSummaryResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const ARM_COLORS = ["#60a5fa", "#f472b6", "#a78bfa", "#fbbf24", "#34d399"];

function fmtUsd(n: number | undefined): string {
  if (n === undefined || Number.isNaN(n)) return "-";
  return `$${n.toLocaleString(undefined, {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  })}`;
}

export function ExperimentCompare() {
  const { data: summary } = useSWR<CompareSummaryResp>(
    "/api/experiment/compare/summary",
    fetcher,
    { refreshInterval: 30_000 },
  );
  const { data: equity } = useSWR<CompareEquityResp>(
    "/api/experiment/compare/equity?period=1d",
    fetcher,
    { refreshInterval: 30_000 },
  );

  const chartData = useMemo(() => {
    if (!equity || equity.arms.length === 0) return [];
    // Normalize each arm to % change from its own first snapshot so
    // arms that started at different equity are still comparable.
    const perArmNormalized: Record<string, { ts: number; pct: number }[]> = {};
    for (const arm of equity.arms) {
      if (arm.points.length === 0) {
        perArmNormalized[arm.arm_id] = [];
        continue;
      }
      const open = arm.points[0].equity;
      perArmNormalized[arm.arm_id] = arm.points.map((p) => ({
        ts: new Date(p.ts).getTime(),
        pct: open > 0 ? (p.equity / open - 1) * 100 : 0,
      }));
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
  }, [equity]);

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
                  <th className="py-2 text-right">cash</th>
                  <th className="py-2 text-right">orders</th>
                  <th className="py-2 text-right">buys $</th>
                  <th className="py-2 text-right">sells $</th>
                </tr>
              </thead>
              <tbody>
                {(summary?.arms ?? []).map((row, i) => (
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
                    <td className="py-2 text-right tabular-nums">
                      {fmtUsd(row.cash)}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {row.n_orders ?? 0}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {fmtUsd(row.buys_notional)}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {fmtUsd(row.sells_notional)}
                    </td>
                  </tr>
                ))}
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
    </div>
  );
}

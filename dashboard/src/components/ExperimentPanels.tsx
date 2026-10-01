import { useMemo, useState } from "react";
import useSWR from "swr";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type {
  DivergencesResp, JevLatencyResp, MemoryResp,
  SignificanceResp, WhipsawsResp,
} from "@/lib/api";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const ARM_COLOR_BY_ID: Record<string, string> = {
  A: "#34d399", B: "#fbbf24", C: "#a78bfa",
};
const armColor = (arm: string) => ARM_COLOR_BY_ID[arm] ?? "#60a5fa";

function usd(n: number | null | undefined, signed = true): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  const sign = signed && n > 0 ? "+" : "";
  return `${sign}$${n.toLocaleString(undefined, {
    minimumFractionDigits: 2, maximumFractionDigits: 2,
  })}`;
}

/** 9-day paired significance tile. Highlights when the user's
 * B-vs-C claim crosses p<0.05. */
export function SignificanceTile() {
  const { data } = useSWR<SignificanceResp>(
    "/api/experiment/significance?since=2026-09-22",
    fetcher,
    { refreshInterval: 300_000 },  // significance only changes at EOD
  );
  if (!data || data.error) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Paired significance</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          {data?.error ?? "loading…"}
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">
          Paired significance
          <span className="ml-2 text-xs font-normal text-muted-foreground">
            {data.n_days} days · {data.first_day} → {data.last_day}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted-foreground">
              <th className="text-left font-normal">pair</th>
              <th className="text-right font-normal">n</th>
              <th className="text-right font-normal">mean Δ</th>
              <th className="text-right font-normal">95% CI</th>
              <th className="text-right font-normal">p-value</th>
            </tr>
          </thead>
          <tbody>
            {data.pairs.map((p) => (
              <tr key={`${p.lhs}-${p.rhs}`} className="border-t border-border/40">
                <td className="py-1 font-mono">{p.lhs} vs {p.rhs}</td>
                <td className="py-1 text-right tabular-nums">{p.n}</td>
                <td className={cn(
                  "py-1 text-right tabular-nums",
                  p.mean_delta_usd > 0 ? "text-success" : "text-danger",
                )}>
                  {usd(p.mean_delta_usd)}
                </td>
                <td className="py-1 text-right tabular-nums text-muted-foreground">
                  [{usd(p.ci_low_usd, false)}, {usd(p.ci_high_usd, false)}]
                </td>
                <td className="py-1 text-right tabular-nums">
                  <span className={cn(
                    p.significant && "rounded bg-success/20 px-1.5 py-0.5 text-success font-semibold ring-1 ring-success/40",
                  )}>
                    {p.p_value === null ? "N/A" : p.p_value.toFixed(4)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

/** Today's B/C divergence stats + expandable row list with per-row
 * 15-min forward P&L (the "effect" of each divergence). */
export function DivergenceTile() {
  const [expanded, setExpanded] = useState(false);
  const { data } = useSWR<DivergencesResp>(
    "/api/experiment/divergences",
    fetcher,
    { refreshInterval: 60_000 },
  );
  if (!data) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Divergences today</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">loading…</CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center justify-between">
          <span>
            Divergences today
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              {data.date}
            </span>
          </span>
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="text-xs font-normal text-primary hover:underline"
          >
            {expanded ? "collapse" : "show rows"}
          </button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-6 text-xs">
          <div>
            <div className="text-muted-foreground">total</div>
            <div className="text-2xl font-semibold tabular-nums">{data.total}</div>
          </div>
          <div>
            <div className="text-muted-foreground">B fires / C abstains</div>
            <div className="text-xl font-semibold tabular-nums" style={{ color: armColor("B") }}>
              {data.b_fires_c_abstains}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">C fires / B abstains</div>
            <div className="text-xl font-semibold tabular-nums" style={{ color: armColor("C") }}>
              {data.c_fires_b_abstains}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">
              T+15m effect ({data.with_forward_pnl} scored)
            </div>
            <div className={cn(
              "text-xl font-semibold tabular-nums",
              data.net_forward_pnl_usd > 0 ? "text-success"
                : data.net_forward_pnl_usd < 0 ? "text-danger" : "text-muted-foreground",
            )}>
              {usd(data.net_forward_pnl_usd)}
            </div>
          </div>
        </div>
        {expanded && data.rows.length > 0 && (
          <div className="mt-3 max-h-72 overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card text-muted-foreground">
                <tr>
                  <th className="text-left font-normal">time</th>
                  <th className="text-right font-normal">nH</th>
                  <th className="text-left font-normal">B · laya</th>
                  <th className="text-left font-normal">C · jev</th>
                  <th className="text-left font-normal">acted</th>
                  <th className="text-left font-normal">tickers</th>
                  <th className="text-right font-normal">T+15m $</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((r, idx) => {
                  const effect = r.acting_arm_pnl_15min_usd;
                  return (
                    <tr key={`${r.ts}-${idx}`} className="border-t border-border/40">
                      <td className="py-1 font-mono text-muted-foreground">
                        {r.ts.slice(11, 19)}
                      </td>
                      <td className="py-1 text-right tabular-nums">{r.n_headlines}</td>
                      <td className="py-1 font-mono">
                        <span className={r.laya_material ? "text-success" : "text-muted-foreground"}>
                          {r.laya_material ? "✓" : "·"}
                        </span>
                        <span className="ml-1 tabular-nums">{r.laya_conf.toFixed(2)}</span>
                      </td>
                      <td className="py-1 font-mono">
                        <span className={r.jev_material ? "text-success" : "text-muted-foreground"}>
                          {r.jev_material ? "✓" : "·"}
                        </span>
                        <span className="ml-1 tabular-nums">{r.jev_conf.toFixed(2)}</span>
                      </td>
                      <td className="py-1 font-mono" style={{ color: armColor(r.acting_arm) }}>
                        {r.acting_arm}
                      </td>
                      <td className="py-1 font-mono text-muted-foreground">
                        {r.acting_arm_tickers?.join(", ") || "–"}
                      </td>
                      <td className={cn(
                        "py-1 text-right tabular-nums font-mono",
                        effect == null ? "text-muted-foreground"
                          : effect > 0 ? "text-success" : effect < 0 ? "text-danger" : "",
                      )}>
                        {effect == null ? "–" : usd(effect)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** Per-arm memory strip. Each row shows the % of threshold the arm's
 * private usage is at. Yellow at >80%, red at >95% (within ~200MB
 * of recycle). */
export function MemoryStrip() {
  const { data } = useSWR<MemoryResp>(
    "/api/experiment/memory",
    fetcher,
    { refreshInterval: 30_000 },
  );
  if (!data || data.arms.length === 0) return null;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Per-arm memory headroom</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {data.arms.map((a) => {
          const pct = a.pct;
          const barColor = pct >= 95 ? "bg-danger"
            : pct >= 80 ? "bg-warning" : "bg-success";
          return (
            <div key={a.arm} className="flex items-center gap-3 text-xs">
              <span className="w-6 font-mono" style={{ color: armColor(a.arm) }}>
                {a.arm}
              </span>
              <div className="relative h-3 flex-1 overflow-hidden rounded bg-muted/40">
                <div
                  className={cn("absolute inset-y-0 left-0", barColor)}
                  style={{ width: `${Math.min(100, pct)}%` }}
                />
              </div>
              <span className="w-32 tabular-nums text-right text-muted-foreground">
                {a.private_mb}/{a.threshold_mb} MB
              </span>
              <span className="w-16 tabular-nums text-right">
                {pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

/** Same-ticker buy/sell direction-change count per arm today.
 * 2+ flips surfaces whipsaw patterns the postmortem catches. */
export function WhipsawTable() {
  const { data } = useSWR<WhipsawsResp>(
    "/api/experiment/whipsaws",
    fetcher,
    { refreshInterval: 60_000 },
  );
  if (!data || data.rows.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Whipsaws today</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          none yet
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Whipsaws today</CardTitle>
      </CardHeader>
      <CardContent>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted-foreground">
              <th className="text-left font-normal">arm</th>
              <th className="text-left font-normal">ticker</th>
              <th className="text-right font-normal">flips</th>
              <th className="text-right font-normal">orders</th>
              <th className="text-left font-normal pl-2">pattern</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.slice(0, 15).map((r) => (
              <tr key={`${r.arm}-${r.ticker}`} className="border-t border-border/40">
                <td className="py-1 font-mono" style={{ color: armColor(r.arm) }}>
                  {r.arm}
                </td>
                <td className="py-1 font-mono">{r.ticker}</td>
                <td className="py-1 text-right tabular-nums">{r.flips}</td>
                <td className="py-1 text-right tabular-nums">{r.n_orders}</td>
                <td className="py-1 pl-2 font-mono text-muted-foreground">
                  {r.pattern}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

/** Sparkline of Jev and Laya per-headline max latency across today's
 * gates. Reveals the tail-spike shape without opening the log. */
export function GateLatencyChart() {
  const { data } = useSWR<JevLatencyResp>(
    "/api/experiment/jev-latency?limit=400",
    fetcher,
    { refreshInterval: 60_000 },
  );
  const merged = useMemo(() => {
    if (!data) return [];
    // Build a shared time index so the two series overlay cleanly.
    const index: Record<string, { ts: string; jev?: number; laya?: number }> = {};
    for (const r of data.jev) {
      index[r.ts] = { ...(index[r.ts] ?? { ts: r.ts }), jev: r.ms_max };
    }
    for (const r of data.laya) {
      index[r.ts] = { ...(index[r.ts] ?? { ts: r.ts }), laya: r.ms_max };
    }
    return Object.values(index).sort((a, b) => a.ts.localeCompare(b.ts));
  }, [data]);
  if (!data || merged.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Gate per-headline latency</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">loading…</CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">
          Gate per-headline latency
          <span className="ml-2 text-xs font-normal text-muted-foreground">
            max_ms per call · {data.jev.length} Jev · {data.laya.length} Laya
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[180px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={merged}>
              <CartesianGrid stroke="#22252b" strokeDasharray="3 3" />
              <XAxis dataKey="ts" tick={{ fontSize: 10 }} stroke="#64748b"
                     tickFormatter={(t) => (t as string).slice(0, 5)} />
              <YAxis tick={{ fontSize: 10 }} stroke="#64748b"
                     tickFormatter={(v) => `${v}ms`} />
              <Tooltip
                contentStyle={{
                  background: "#0b0d10",
                  border: "1px solid #22252b",
                  fontSize: 12,
                }}
                formatter={(v) => `${v}ms`}
              />
              <Line type="monotone" dataKey="jev"
                    stroke={armColor("C")} strokeWidth={1.5}
                    dot={false} name="Jev (C)" />
              <Line type="monotone" dataKey="laya"
                    stroke={armColor("B")} strokeWidth={1.5}
                    dot={false} name="Laya (B)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

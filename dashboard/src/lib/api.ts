/** Fetch wrappers for the FastAPI backend, used with SWR. */

import { useEffect, useState } from "react";

export type PortfolioResp = {
  equity: number;
  cash: number;
  buying_power: number;
  portfolio_value: number;
  account_number: string;
  error?: string;
};

export type PositionResp = {
  ticker: string;
  qty: number;
  avg_entry_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_pl_pct: number;
};

export type SnapshotResp = {
  ts: string;
  equity: number;
  cash: number;
  portfolio_value: number;
};

export type BarResp = {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  sma20: number | null;
};

export type BarsResp = {
  ticker: string;
  bars: BarResp[];
  error?: string;
};

export type TradeResp = {
  client_order_id: string;
  broker_order_id: string | null;
  ticker: string;
  side: "buy" | "sell";
  qty: number;
  order_type: string;
  status: string;
  submitted_at: string;
  filled_at: string | null;
  filled_avg_price: number | null;
  rec_id: number | null;
  source: string | null;
};

export type RecResp = {
  rec_id: number;
  amount: number;
  risk: string;
  cash_pct: number;
  cash_dollars: number;
  portfolio_rationale: string;
  positions: Array<{
    ticker: string;
    weight_pct: number;
    dollars: number;
    confidence: number;
    rationale: string;
  }>;
};

export function currentArm(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("arm");
}

export type DashboardView = "single" | "compare" | "eod";

export function currentView(): DashboardView {
  if (typeof window === "undefined") return "single";
  const v = new URLSearchParams(window.location.search).get("view");
  if (v === "compare") return "compare";
  if (v === "eod") return "eod";
  return "single";
}

export function withArm(path: string): string {
  const arm = currentArm();
  if (!arm) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}arm=${encodeURIComponent(arm)}`;
}

/**
 * Navigate to a different arm/view. Uses a full page reload rather
 * than history.replaceState + SWR cache invalidation. Reason: SWR's
 * global cache is keyed by URL string and doesn't include the arm,
 * so the "reactive" approach kept serving stale data from the
 * previous arm to fresh mounts even with cache clearing (subscriber-
 * attach vs mutate-return race). A reload is guaranteed correct.
 */
export function navigateArmView(
  arm: string | null,
  view: DashboardView,
): void {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  if (arm) params.set("arm", arm);
  else params.delete("arm");
  if (view === "single") params.delete("view");
  else params.set("view", view);
  const qs = params.toString();
  window.location.href = qs
    ? `${window.location.pathname}?${qs}`
    : window.location.pathname;
}

function useUrlParam<T>(read: () => T): T {
  const [value, setValue] = useState<T>(() => read());
  useEffect(() => {
    const handler = () => setValue(read());
    window.addEventListener("armchange", handler);
    window.addEventListener("popstate", handler);
    return () => {
      window.removeEventListener("armchange", handler);
      window.removeEventListener("popstate", handler);
    };
  }, []);
  return value;
}

export function useArmParam(): string | null {
  return useUrlParam(currentArm);
}

export function useViewParam(): DashboardView {
  return useUrlParam(currentView);
}

export const fetcher = async <T>(url: string): Promise<T> => {
  const r = await fetch(withArm(url));
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

export type ExperimentMeta =
  | { mode: "single" }
  | {
      mode: "experiment";
      name: string;
      default_arm: string;
      arms: Array<{ id: string; account: string }>;
    };

export type ArmSummaryRow = {
  arm_id: string;
  account: string;
  equity?: number;
  cash?: number;
  cash_pct?: number;
  portfolio_value?: number;
  positions_count?: number;
  positions?: Array<{
    ticker: string;
    qty: number;
    market_value: number;
    unrealized_pl_pct: number;
  }>;
  opening_equity?: number;
  delta_dollars?: number;
  delta_pct?: number;
  baseline_at?: string;
  n_orders?: number;
  buys_notional?: number;
  sells_notional?: number;
  turnover?: number;
  last_snapshot_at?: string;
  broker_error?: string;
  orders_error?: string;
  ticks?: number;
  regen_cost_total?: number;
  regen_cost_avg?: number;
  regen_cost_last5_avg?: number;
  cache_hit_pct?: number;
};

export type CompareSummaryResp = {
  experiment: string;
  arms: ArmSummaryRow[];
};

export type ArmEquitySeries = {
  arm_id: string;
  points: Array<{ ts: string; equity: number }>;
};

export type CompareEquityResp = {
  experiment: string;
  period: string;
  arms: ArmEquitySeries[];
};

export type ReactionOrder = {
  ticker: string;
  side: "buy" | "sell" | string;
  qty: number;
};

export type ArmReaction =
  | { reacted: "none" }
  | { reacted: "skip"; seconds: number; kind: string }
  | {
      reacted: "regen" | "orders";
      regen: {
        seconds: number;
        rec_id: number;
        targets_count: number;
        cash_pct: number | null;
        trigger: string | null;
      } | null;
      orders: ReactionOrder[];
    };

export type NewsReactionRow = {
  ts: string;
  ticker: string | null;
  headline: string;
  per_arm: Record<string, ArmReaction>;
};

export type NewsReactionsResp = {
  experiment: string;
  news_reactions: NewsReactionRow[];
};

export type SignificancePair = {
  lhs: string;
  rhs: string;
  n: number;
  mean_delta_usd: number;
  median_delta_usd: number;
  ci_low_usd: number;
  ci_high_usd: number;
  p_value: number | null;
  significant: boolean;
};

export type SignificanceResp = {
  since: string;
  n_days: number;
  first_day: string | null;
  last_day: string | null;
  pairs: SignificancePair[];
  error?: string;
};

export type DivergenceRow = {
  ts: string;
  n_headlines: number;
  laya_material: boolean;
  laya_conf: number;
  jev_material: boolean;
  jev_conf: number;
  acting_arm: "B" | "C";
  pair_gap_sec: number;
  acting_arm_order_count?: number;
  acting_arm_tickers?: string[];
  acting_arm_notional_usd?: number;
  acting_arm_pnl_15min_usd?: number | null;
};

export type DivergencesResp = {
  date: string;
  total: number;
  b_fires_c_abstains: number;
  c_fires_b_abstains: number;
  with_forward_pnl: number;
  net_forward_pnl_usd: number;
  rows: DivergenceRow[];
};

export type MemoryArmRow = {
  arm: string;
  ts: string;
  private_mb: number;
  threshold_mb: number;
  pct: number;
  headroom_mb: number;
};

export type MemoryResp = {
  date: string;
  arms: MemoryArmRow[];
};

export type WhipsawRow = {
  arm: string;
  ticker: string;
  flips: number;
  n_orders: number;
  pattern: string;
};

export type WhipsawsResp = {
  date: string;
  rows: WhipsawRow[];
};

export type JevLatencyRow = {
  ts: string;
  n_headlines: number;
  ms_total: number;
  ms_max: number;
};

export type JevLatencyResp = {
  date: string;
  jev: JevLatencyRow[];
  laya: JevLatencyRow[];
};

export type EodResp = {
  date: string | null;
  markdown: string | null;
};

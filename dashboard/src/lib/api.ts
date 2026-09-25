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

export function currentView(): "single" | "compare" {
  if (typeof window === "undefined") return "single";
  return new URLSearchParams(window.location.search).get("view") === "compare"
    ? "compare"
    : "single";
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
  view: "single" | "compare",
): void {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams(window.location.search);
  if (arm) params.set("arm", arm);
  else params.delete("arm");
  if (view === "compare") params.set("view", "compare");
  else params.delete("view");
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

export function useViewParam(): "single" | "compare" {
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

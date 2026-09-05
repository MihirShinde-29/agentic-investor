/** Fetch wrappers for the FastAPI backend, used with SWR. */

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

export function withArm(path: string): string {
  const arm = currentArm();
  if (!arm) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}arm=${encodeURIComponent(arm)}`;
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
  portfolio_value?: number;
  n_orders?: number;
  buys_notional?: number;
  sells_notional?: number;
  last_snapshot_at?: string;
  broker_error?: string;
  orders_error?: string;
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

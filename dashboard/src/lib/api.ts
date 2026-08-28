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

export const fetcher = async <T>(url: string): Promise<T> => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

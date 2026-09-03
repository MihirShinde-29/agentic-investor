// Shared timeframe config for PortfolioChart, TickerCard, and any future
// chart component. One selector at the app level drives them all.
//
// yfinance interval caps enforced: 1m ≤ 7d, 5m/15m/30m ≤ 60d,
// 1h ≤ 730d, 1d unlimited. 1D uses 1-minute bars so the chart fills
// up minute-by-minute during the session.

export type Timeframe = "1D" | "3D" | "1W" | "1M" | "3M" | "1Y";

export const TIMEFRAMES: Record<
  Timeframe,
  { period: string; interval: string; refreshMs: number }
> = {
  "1D": { period: "1d",  interval: "1m",  refreshMs: 30_000 },
  "3D": { period: "3d",  interval: "15m", refreshMs: 60_000 },
  "1W": { period: "1w",  interval: "1h",  refreshMs: 5 * 60_000 },
  "1M": { period: "1mo", interval: "1h",  refreshMs: 5 * 60_000 },
  "3M": { period: "3mo", interval: "1d",  refreshMs: 15 * 60_000 },
  "1Y": { period: "1y",  interval: "1d",  refreshMs: 30 * 60_000 },
};

// Seconds per bar for each timeframe. Trade markers must snap to their
// containing bar boundary or lightweight-charts places them on the nearest
// bar, which visibly misaligns on 15m/1h/1d intervals.
export const BAR_SECONDS: Record<Timeframe, number> = {
  "1D": 60,
  "3D": 15 * 60,
  "1W": 60 * 60,
  "1M": 60 * 60,
  "3M": 24 * 60 * 60,
  "1Y": 24 * 60 * 60,
};

export const TIMEFRAME_ORDER: Timeframe[] = [
  "1D",
  "3D",
  "1W",
  "1M",
  "3M",
  "1Y",
];

import useSWR from "swr";
import { Activity, DollarSign, PiggyBank, TrendingUp, Wallet, Zap } from "lucide-react";
import type { PortfolioResp, SnapshotResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import { formatPct, formatUSD } from "@/lib/utils";
import type { LiveEvent } from "@/hooks/useLiveEvents";
import { cn } from "@/lib/utils";

type CostSummary = { calls: number; cost: number; cached: number; prompt: number };

function summarizeCost(events: LiveEvent[]): CostSummary {
  let calls = 0;
  let cost = 0;
  let cached = 0;
  let prompt = 0;
  for (const e of events) {
    if (e.event !== "tick_cost") continue;
    calls += Number(e.llm_calls ?? 0);
    prompt += Number(e.prompt_tokens ?? 0);
    cached += Number(e.cached_tokens ?? 0);
    const c = e.cost_usd;
    if (typeof c === "number") cost += c;
    else if (typeof c === "string") cost += Number(c.replace(/^\$/, "")) || 0;
  }
  return { calls, cost, cached, prompt };
}

function Tile({
  label,
  value,
  hint,
  icon: Icon,
  accent = "text-foreground",
}: {
  label: string;
  value: string;
  hint?: string;
  icon: React.ComponentType<{ className?: string }>;
  accent?: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border/40 bg-card/60 px-3 py-2.5">
      <div className="grid size-9 place-items-center rounded-md bg-muted/40 text-muted-foreground">
        <Icon className="size-4" />
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
          {label}
        </div>
        <div className={cn("text-sm font-semibold tabular", accent)}>{value}</div>
        {hint && (
          <div className="text-[10px] tabular text-muted-foreground">{hint}</div>
        )}
      </div>
    </div>
  );
}

export function HeaderStrip({ events }: { events: LiveEvent[] }) {
  const { data: portfolio } = useSWR<PortfolioResp>("/api/portfolio", fetcher, {
    refreshInterval: 15_000,
  });
  const { data: snaps } = useSWR<SnapshotResp[]>(
    "/api/snapshots?limit=500",
    fetcher,
    { refreshInterval: 30_000 },
  );

  const equity = portfolio?.equity ?? 0;
  const cash = portfolio?.cash ?? 0;
  const cashPct = equity > 0 ? (cash / equity) * 100 : 0;

  // Session P&L: current equity vs first snapshot of the day.
  const openEquity = snaps && snaps.length > 0 ? snaps[0].equity : equity;
  const dayPL = equity - openEquity;
  const dayPLPct = openEquity > 0 ? (dayPL / openEquity) * 100 : 0;

  const cost = summarizeCost(events);
  const cacheRate = cost.prompt > 0 ? (cost.cached / cost.prompt) * 100 : 0;

  const gainAccent =
    dayPL > 0.001 ? "text-success" : dayPL < -0.001 ? "text-danger" : "text-foreground";

  return (
    <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3 xl:grid-cols-6">
      <Tile
        icon={DollarSign}
        label="Equity"
        value={formatUSD(equity, 0)}
        hint={portfolio?.account_number ? `acct ${portfolio.account_number}` : undefined}
      />
      <Tile
        icon={TrendingUp}
        label="Day P&L"
        value={formatUSD(dayPL, 2)}
        hint={formatPct(dayPLPct, 2)}
        accent={gainAccent}
      />
      <Tile
        icon={Wallet}
        label="Cash"
        value={formatUSD(cash, 0)}
        hint={`${cashPct.toFixed(1)}%`}
      />
      <Tile
        icon={PiggyBank}
        label="Portfolio value"
        value={formatUSD(portfolio?.portfolio_value ?? 0, 0)}
      />
      <Tile
        icon={Activity}
        label="LLM calls (session)"
        value={cost.calls.toString()}
        hint={`~$${cost.cost.toFixed(4)}`}
      />
      <Tile
        icon={Zap}
        label="Cache hit"
        value={`${cacheRate.toFixed(0)}%`}
        hint={`${cost.cached.toLocaleString()} cached tokens`}
        accent={cacheRate > 30 ? "text-success" : "text-foreground"}
      />
    </div>
  );
}

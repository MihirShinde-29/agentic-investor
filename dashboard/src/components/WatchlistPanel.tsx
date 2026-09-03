import useSWR from "swr";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatPct, formatUSD } from "@/lib/utils";

type Held = {
  ticker: string;
  qty: number;
  market_value: number;
  unrealized_pl_pct: number;
};

type Exit = { ticker: string; current_price: number | null };

type OnDeck = {
  ticker: string;
  target_weight_pct: number;
  target_dollars: number;
  confidence: number;
  current_price: number | null;
};

type WatchlistResp = {
  held: Held[];
  recent_exits: Exit[];
  on_deck: OnDeck[];
};

export function WatchlistPanel() {
  const { data } = useSWR<WatchlistResp>(
    "/api/watchlist",
    fetcher,
    { refreshInterval: 30_000 },
  );

  const held = data?.held ?? [];
  const exits = data?.recent_exits ?? [];
  const deck = data?.on_deck ?? [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>Shadow book</CardTitle>
        <span className="text-xs text-muted-foreground">
          held vs recent exits vs the LLM's on-deck picks
        </span>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-3">
        <Section title={`Held (${held.length})`} variant="default">
          {held.length === 0 && <Empty>no positions</Empty>}
          {held.map((h) => (
            <Row
              key={h.ticker}
              ticker={h.ticker}
              right={
                <Badge variant={h.unrealized_pl_pct >= 0 ? "success" : "danger"}>
                  {formatPct(h.unrealized_pl_pct)}
                </Badge>
              }
              subtitle={`${h.qty.toFixed(2)} · ${formatUSD(h.market_value)}`}
            />
          ))}
        </Section>

        <Section title={`Recent exits (${exits.length})`} variant="danger">
          {exits.length === 0 && <Empty>none in last 24h</Empty>}
          {exits.map((e) => (
            <Row
              key={e.ticker}
              ticker={e.ticker}
              right={
                e.current_price != null ? (
                  <span className="text-xs text-muted-foreground">
                    {formatUSD(e.current_price)}
                  </span>
                ) : null
              }
              subtitle="sold to 0"
            />
          ))}
        </Section>

        <Section title={`On deck (${deck.length})`} variant="success">
          {deck.length === 0 && (
            <Empty>book matches latest rec</Empty>
          )}
          {deck.map((d) => (
            <Row
              key={d.ticker}
              ticker={d.ticker}
              right={
                <Badge variant="muted">
                  target {formatPct(d.target_weight_pct)}
                </Badge>
              }
              subtitle={
                d.current_price != null
                  ? `conf ${(d.confidence ?? 0).toFixed(2)} · ${formatUSD(
                      d.current_price,
                    )}`
                  : `conf ${(d.confidence ?? 0).toFixed(2)}`
              }
            />
          ))}
        </Section>
      </CardContent>
    </Card>
  );
}

function Section({
  title,
  variant,
  children,
}: {
  title: string;
  variant: "default" | "success" | "danger";
  children: React.ReactNode;
}) {
  const tint =
    variant === "success"
      ? "border-emerald-500/30"
      : variant === "danger"
        ? "border-rose-500/30"
        : "border-border";
  return (
    <div className={`rounded-md border ${tint} p-2`}>
      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {title}
      </div>
      <div className="flex flex-col gap-1">{children}</div>
    </div>
  );
}

function Row({
  ticker,
  subtitle,
  right,
}: {
  ticker: string;
  subtitle: string;
  right: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between rounded px-1.5 py-1 text-sm hover:bg-muted/40">
      <div className="flex flex-col">
        <span className="font-medium">{ticker}</span>
        <span className="text-[10px] text-muted-foreground">{subtitle}</span>
      </div>
      {right}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="py-2 text-center text-xs text-muted-foreground">
      {children}
    </div>
  );
}

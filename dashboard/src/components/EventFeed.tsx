import { useState } from "react";
import { Radio } from "lucide-react";
import type { LiveEvent } from "@/hooks/useLiveEvents";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type Kind = "news" | "regen" | "skip" | "trade" | "fill" | "info" | "cost";

const EVENT_TO_KIND: Record<string, Kind> = {
  news_received: "news",
  regen_start: "regen",
  regen_done: "regen",
  opinion_drift_skip: "skip",
  filter_skip: "skip",
  trade_plan: "trade",
  order_submitted: "trade",
  order_filled: "fill",
  tick_cost: "cost",
  session_start: "info",
  session_end: "info",
  state_restored: "info",
  price_move_trigger: "info",
  force_regen: "info",
};

const KIND_STYLES: Record<Kind, { variant: React.ComponentProps<typeof Badge>["variant"]; label: string; dot: string }> = {
  news: { variant: "primary", label: "news", dot: "bg-primary" },
  regen: { variant: "warning", label: "regen", dot: "bg-warning" },
  skip: { variant: "muted", label: "skip", dot: "bg-muted-foreground" },
  trade: { variant: "success", label: "trade", dot: "bg-success" },
  fill: { variant: "success", label: "fill", dot: "bg-success" },
  cost: { variant: "muted", label: "cost", dot: "bg-muted-foreground" },
  info: { variant: "muted", label: "info", dot: "bg-muted-foreground" },
};

function summarize(evt: LiveEvent): string {
  switch (evt.event) {
    case "news_received":
      return `${evt.ticker ?? "?"} · ${(evt.headline as string ?? "").slice(0, 60)}`;
    case "regen_start":
      return `trigger: ${evt.reason ?? "unknown"}`;
    case "regen_done":
      return `rec #${evt.rec_id ?? "?"} · trigger: ${evt.trigger ?? "?"}`;
    case "opinion_drift_skip":
    case "filter_skip":
      return `${evt.skip_reason ?? "?"} · avg ${evt.avg_drift_pp ?? "?"}pp`;
    case "trade_plan": {
      const n = Array.isArray(evt.trades) ? evt.trades.length : 0;
      return `${n} trades planned · rec #${evt.rec_id ?? "?"}`;
    }
    case "order_submitted":
      return `${evt.side ?? "?"} ${evt.qty ?? "?"} ${evt.ticker ?? "?"}`;
    case "tick_cost":
      return `${evt.llm_calls ?? 0} calls · ${evt.cost_usd ?? "$0"}`;
    case "session_start":
      return `dir: ${evt.out_dir ?? ""}`;
    case "state_restored":
      return `rec #${evt.last_rec_id ?? "?"} · date ${evt.last_rec_date ?? "?"}`;
    case "session_end":
      return "session finalized";
    default:
      return JSON.stringify(evt).slice(0, 80);
  }
}

function relTime(ts: string): string {
  const secs = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (secs < 5) return "just now";
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  return `${Math.floor(secs / 3600)}h ago`;
}

type Filter = "important" | "trades" | "news" | "all";

const IMPORTANT_KINDS: Set<Kind> = new Set(["news", "regen", "trade", "fill", "skip"]);
const TRADE_KINDS: Set<Kind> = new Set(["trade", "fill"]);
const NEWS_KINDS: Set<Kind> = new Set(["news"]);

const FILTERS: { key: Filter; label: string }[] = [
  { key: "important", label: "Important" },
  { key: "trades", label: "Trades" },
  { key: "news", label: "News" },
  { key: "all", label: "All" },
];

export function EventFeed({ events }: { events: LiveEvent[] }) {
  const [filter, setFilter] = useState<Filter>("important");

  const filtered = events.filter((evt) => {
    const kind = EVENT_TO_KIND[evt.event] ?? "info";
    if (filter === "all") return true;
    if (filter === "important") return IMPORTANT_KINDS.has(kind);
    if (filter === "trades") return TRADE_KINDS.has(kind);
    if (filter === "news") return NEWS_KINDS.has(kind);
    return true;
  });
  const shown = [...filtered].reverse().slice(0, 60);

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-col items-stretch gap-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Radio className="size-4 text-primary" />
            <CardTitle>Live event feed</CardTitle>
          </div>
          <span className="text-xs text-muted-foreground">
            {shown.length}/{filtered.length} shown · {events.length} total
          </span>
        </div>
        <div className="flex items-center gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              onClick={() => setFilter(f.key)}
              className={cn(
                "rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors",
                filter === f.key
                  ? "bg-primary/20 text-primary ring-1 ring-primary/40"
                  : "text-muted-foreground hover:bg-muted/40 hover:text-foreground",
              )}
            >
              {f.label}
            </button>
          ))}
        </div>
      </CardHeader>
      <div className="max-h-[calc(100vh)] flex-1 overflow-y-auto p-3">
        {shown.length === 0 && (
          <p className="px-2 py-8 text-center text-xs text-muted-foreground">
            no events yet
          </p>
        )}
        <ul className="space-y-1.5">
          {shown.map((evt, i) => {
            const kind = EVENT_TO_KIND[evt.event] ?? "info";
            const style = KIND_STYLES[kind];
            return (
              <li
                key={`${evt.ts}-${i}`}
                className="overflow-hidden rounded-md border border-border/40 bg-background/40 p-2.5 text-xs"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex min-w-0 flex-1 items-center gap-2">
                    <span className={cn("size-1.5 shrink-0 rounded-full", style.dot)} />
                    <Badge variant={style.variant}>{style.label}</Badge>
                    <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground">
                      {evt.event}
                    </span>
                  </div>
                  <span className="shrink-0 whitespace-nowrap text-[10px] text-muted-foreground">
                    {relTime(evt.ts)}
                  </span>
                </div>
                <div className="mt-1 line-clamp-3 break-words pl-3.5 text-[11px] leading-snug text-foreground/85">
                  {summarize(evt)}
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </Card>
  );
}

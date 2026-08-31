import { useEffect, useMemo, useState } from "react";
import { Radio, Search, X } from "lucide-react";
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
  orders_reconciled: "info",
  streamer_start: "info",
  streamer_stop: "info",
  decision_moment: "info",
  market_open: "info",
  market_closed: "info",
  finbert_skip: "skip",
};

const KIND_STYLES: Record<
  Kind,
  { variant: React.ComponentProps<typeof Badge>["variant"]; label: string; dot: string }
> = {
  news: { variant: "primary", label: "news", dot: "bg-primary" },
  regen: { variant: "warning", label: "regen", dot: "bg-warning" },
  skip: { variant: "muted", label: "skip", dot: "bg-muted-foreground" },
  trade: { variant: "success", label: "trade", dot: "bg-success" },
  fill: { variant: "success", label: "fill", dot: "bg-success" },
  cost: { variant: "muted", label: "cost", dot: "bg-muted-foreground" },
  info: { variant: "muted", label: "info", dot: "bg-muted-foreground" },
};

const ALL_KINDS: Kind[] = ["news", "regen", "skip", "trade", "fill", "cost", "info"];
// Sensible default: hide the noisy info + cost pills but show everything
// that's a real decision or trade.
const DEFAULT_ENABLED: Kind[] = ["news", "regen", "skip", "trade", "fill"];

const STORAGE_KEY = "ai_event_feed_filter_v1";

type StoredFilter = {
  kinds: Kind[];
  search: string;
};

function loadFilter(): StoredFilter {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { kinds: [...DEFAULT_ENABLED], search: "" };
    const parsed = JSON.parse(raw) as Partial<StoredFilter>;
    return {
      kinds: Array.isArray(parsed.kinds) ? (parsed.kinds as Kind[]) : [...DEFAULT_ENABLED],
      search: typeof parsed.search === "string" ? parsed.search : "",
    };
  } catch {
    return { kinds: [...DEFAULT_ENABLED], search: "" };
  }
}

/**
 * Decode the most common HTML entities so headlines pulled from RSS/press
 * feeds render as human text ("NBC's" instead of "NBC&#39;s"). Also strips
 * dangling partial entities left over when an upstream feed truncated a
 * headline mid-entity (e.g. "...Coin Flip&#3" -> "...Coin Flip").
 */
function decodeEntities(s: string): string {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&#x([\da-fA-F]+);/g, (_, n) => String.fromCharCode(parseInt(n, 16)))
    // strip trailing partial entities like `&#3` or `&am` after upstream cut-off
    .replace(/&#x?[\da-fA-F]*$/, "")
    .replace(/&[a-zA-Z]{1,6}$/, "")
    .trimEnd();
}

function summarize(evt: LiveEvent): string {
  switch (evt.event) {
    case "news_received": {
      const headline = decodeEntities(String(evt.headline ?? "")).trim();
      const source = evt.source ? ` (${String(evt.source)})` : "";
      const tickers = (evt.tickers as string[] | undefined) ?? [
        String(evt.ticker ?? ""),
      ];
      const label =
        tickers.length > 3
          ? `${tickers.slice(0, 3).join(", ")} +${tickers.length - 3}`
          : tickers.join(", ");
      return `${label || "?"} · ${headline}${source}`;
    }
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
    default: {
      // Strip ts/event scaffolding + decode entities so the fallback is
      // still readable when a new event kind hasn't gotten a summarizer yet.
      const { ts: _ts, event: _event, ...rest } = evt;
      void _ts;
      void _event;
      return decodeEntities(JSON.stringify(rest)).slice(0, 140);
    }
  }
}

function relTime(ts: string): string {
  const secs = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (secs < 5) return "just now";
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  return `${Math.floor(secs / 3600)}h ago`;
}

/**
 * Cheap match: does an event mention this ticker (uppercase, word boundary)?
 * Checks ticker field first, falls back to a regex scan of headline/summary.
 */
function eventMatchesTicker(evt: LiveEvent, tickerUpper: string): boolean {
  const primary = (evt.ticker as string | undefined)?.toUpperCase();
  if (primary && primary === tickerUpper) return true;
  const parts: string[] = [];
  if (typeof evt.headline === "string") parts.push(evt.headline);
  if (typeof evt.summary === "string") parts.push(evt.summary);
  if (typeof evt.reason === "string") parts.push(evt.reason);
  if (typeof evt.skip_reason === "string") parts.push(evt.skip_reason);
  if (parts.length === 0) return false;
  return new RegExp(`\\b${tickerUpper}\\b`).test(parts.join(" ").toUpperCase());
}

function KindChip({
  kind,
  on,
  count,
  onToggle,
  onSolo,
}: {
  kind: Kind;
  on: boolean;
  count: number;
  onToggle: () => void;
  onSolo: (e: React.MouseEvent) => void;
}) {
  const style = KIND_STYLES[kind];
  return (
    <button
      type="button"
      onClick={onToggle}
      onDoubleClick={onSolo}
      title={`Click: ${on ? "hide" : "show"} · Double-click: only this`}
      className={cn(
        "group inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium transition-all",
        on
          ? "border-border/50 bg-card/60"
          : "border-transparent bg-transparent opacity-45 hover:opacity-80",
      )}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          on ? style.dot : "bg-muted-foreground/40",
        )}
      />
      <span className={on ? "text-foreground" : "text-muted-foreground"}>
        {style.label}
      </span>
      <span
        className={cn(
          "tabular text-[10px]",
          on ? "text-muted-foreground" : "text-muted-foreground/60",
        )}
      >
        {count}
      </span>
    </button>
  );
}

export function EventFeed({ events }: { events: LiveEvent[] }) {
  const [enabled, setEnabled] = useState<Set<Kind>>(
    () => new Set(loadFilter().kinds),
  );
  const [search, setSearch] = useState<string>(() => loadFilter().search);

  // Persist filter across reloads.
  useEffect(() => {
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ kinds: Array.from(enabled), search }),
      );
    } catch {
      /* ignore */
    }
  }, [enabled, search]);

  const counts: Record<Kind, number> = useMemo(() => {
    const c: Record<Kind, number> = {
      news: 0, regen: 0, skip: 0, trade: 0, fill: 0, cost: 0, info: 0,
    };
    for (const e of events) c[EVENT_TO_KIND[e.event] ?? "info"] += 1;
    return c;
  }, [events]);

  const searchTrim = search.trim();
  const tickerFilter = /^\$?[a-zA-Z]{1,5}$/.test(searchTrim)
    ? searchTrim.replace(/^\$/, "").toUpperCase()
    : null;
  const textLower = searchTrim.toLowerCase();

  const filtered = useMemo(() => {
    return events.filter((evt) => {
      const kind = EVENT_TO_KIND[evt.event] ?? "info";
      if (!enabled.has(kind)) return false;
      if (!searchTrim) return true;
      if (tickerFilter) return eventMatchesTicker(evt, tickerFilter);
      const hay = `${evt.event} ${summarize(evt)}`.toLowerCase();
      return hay.includes(textLower);
    });
  }, [events, enabled, searchTrim, tickerFilter, textLower]);

  /**
   * Dedupe: the news stream fans one headline out into one NewsEvent per
   * matched ticker (Alpaca-tagged + body-extracted). In the feed that shows
   * up as N near-identical rows. Collapse consecutive same-headline news
   * into one row and stash the ticker cluster on the merged event.
   */
  const collapsed = useMemo(() => {
    const out: LiveEvent[] = [];
    for (const evt of filtered) {
      if (evt.event !== "news_received") {
        out.push(evt);
        continue;
      }
      const key = String(evt.headline ?? "").trim();
      const last = out[out.length - 1];
      if (
        last &&
        last.event === "news_received" &&
        String(last.headline ?? "").trim() === key &&
        Math.abs(
          new Date(last.ts).getTime() - new Date(evt.ts).getTime(),
        ) < 60_000
      ) {
        const merged = { ...last };
        const tickers = new Set<string>(
          (merged.tickers as string[] | undefined) ?? [
            String(last.ticker ?? "").toUpperCase(),
          ],
        );
        tickers.add(String(evt.ticker ?? "").toUpperCase());
        merged.tickers = Array.from(tickers).filter(Boolean);
        out[out.length - 1] = merged;
      } else {
        out.push({
          ...evt,
          tickers: [String(evt.ticker ?? "").toUpperCase()].filter(Boolean),
        });
      }
    }
    return out;
  }, [filtered]);

  const shown = collapsed.slice(-60).reverse();

  const toggle = (k: Kind) => {
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  };
  const solo = (k: Kind) => setEnabled(new Set([k]));

  const allOn = enabled.size === ALL_KINDS.length;
  const isDefault =
    enabled.size === DEFAULT_ENABLED.length &&
    DEFAULT_ENABLED.every((k) => enabled.has(k));

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex-col items-stretch gap-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Radio className="size-4 text-primary" />
            <CardTitle>Live event feed</CardTitle>
          </div>
          <span className="text-xs text-muted-foreground">
            {shown.length}/{filtered.length} · {events.length} total
          </span>
        </div>

        {/* Search / ticker filter */}
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search text · or type a ticker ($AAPL / NVDA)"
            className="w-full rounded-md border border-border/50 bg-card/60 py-1 pl-7 pr-7 text-[11px] text-foreground placeholder:text-muted-foreground/60 focus:border-primary/60 focus:outline-none focus:ring-1 focus:ring-primary/40"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-muted-foreground hover:bg-muted/40 hover:text-foreground"
              aria-label="Clear search"
            >
              <X className="size-3" />
            </button>
          )}
        </div>
        {tickerFilter && (
          <div className="text-[10px] text-muted-foreground">
            Filtering by ticker <span className="font-mono text-primary">{tickerFilter}</span>
          </div>
        )}

        {/* Kind chips */}
        <div className="flex flex-wrap items-center gap-1">
          {ALL_KINDS.map((k) => (
            <KindChip
              key={k}
              kind={k}
              on={enabled.has(k)}
              count={counts[k]}
              onToggle={() => toggle(k)}
              onSolo={(e) => {
                e.preventDefault();
                solo(k);
              }}
            />
          ))}
          <div className="ml-auto flex items-center gap-2">
            {!isDefault && (
              <button
                type="button"
                onClick={() => setEnabled(new Set(DEFAULT_ENABLED))}
                className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-muted/40 hover:text-foreground"
              >
                default
              </button>
            )}
            <button
              type="button"
              onClick={() =>
                setEnabled(
                  allOn ? new Set(DEFAULT_ENABLED) : new Set(ALL_KINDS),
                )
              }
              className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-muted/40 hover:text-foreground"
            >
              {allOn ? "none" : "all"}
            </button>
          </div>
        </div>
      </CardHeader>
      <div className="max-h-[calc(100vh)] flex-1 overflow-y-auto p-3">
        {shown.length === 0 && (
          <p className="px-2 py-8 text-center text-xs text-muted-foreground">
            {events.length === 0
              ? "no events yet"
              : "no events match — try widening the filter"}
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
                <div className="mt-1 line-clamp-4 break-words pl-3.5 text-[11px] leading-snug text-foreground/85">
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

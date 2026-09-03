import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Activity, CheckCircle2, Clock, Signal, XCircle } from "lucide-react";
import type { LiveEvent, ConnectionStatus } from "@/hooks/useLiveEvents";
import { fetcher } from "@/lib/api";
import { cn } from "@/lib/utils";

type BrokerStatus = {
  connected: boolean;
  market_open?: boolean;
  next_open?: string;
  next_close?: string;
  error?: string;
};

function relSecs(ts?: string): number | null {
  if (!ts) return null;
  return Math.max(0, Math.floor((Date.now() - new Date(ts).getTime()) / 1000));
}

function humanSecs(s: number | null): string {
  if (s == null) return "—";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function StatDot({ ok }: { ok: boolean }) {
  return ok ? (
    <CheckCircle2 className="size-3.5 text-success" />
  ) : (
    <XCircle className="size-3.5 text-danger" />
  );
}

export function HealthStrip({
  events,
  wsStatus,
}: {
  events: LiveEvent[];
  wsStatus: ConnectionStatus;
}) {
  const { data: broker } = useSWR<BrokerStatus>(
    "/api/broker/status",
    fetcher,
    { refreshInterval: 20_000 },
  );
  // Force a rerender each second so "12s ago" ticks live.
  const [, setTick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const lastNews = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].event === "news_received") return events[i].ts;
    }
    return undefined;
  }, [events]);

  const lastTick = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].event === "tick_cost") return events[i].ts;
    }
    return undefined;
  }, [events]);

  // Any event, so during quiet stretches (no news, no ticks) we still see
  // that the WS is alive and the loop is doing something.
  const lastEvent = events.length > 0 ? events[events.length - 1].ts : undefined;

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border/40 bg-card/40 px-3 py-2 text-[11px]">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-1.5">
          <StatDot ok={wsStatus === "open"} />
          <span className="text-muted-foreground">WebSocket</span>
          <span className="font-medium">
            {wsStatus === "open" ? "connected" : wsStatus}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <StatDot ok={!!broker?.connected} />
          <span className="text-muted-foreground">Alpaca</span>
          <span className="font-medium">
            {broker?.connected
              ? broker.market_open
                ? "market open"
                : "market closed"
              : "unreachable"}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <Signal className="size-3.5 text-muted-foreground" />
          <span className="text-muted-foreground">Last news</span>
          <span className="font-medium">{humanSecs(relSecs(lastNews))}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Activity className="size-3.5 text-muted-foreground" />
          <span className="text-muted-foreground">Last tick</span>
          <span className="font-medium">{humanSecs(relSecs(lastTick))}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Activity className="size-3.5 text-muted-foreground" />
          <span className="text-muted-foreground">Last event</span>
          <span className="font-medium">{humanSecs(relSecs(lastEvent))}</span>
        </div>
      </div>
      <div
        className={cn(
          "flex items-center gap-1.5 text-muted-foreground",
          !broker?.connected && "text-danger",
        )}
      >
        <Clock className="size-3.5" />
        {broker?.next_open && !broker.market_open && (
          <span>next open {new Date(broker.next_open).toLocaleString()}</span>
        )}
        {broker?.next_close && broker.market_open && (
          <span>next close {new Date(broker.next_close).toLocaleTimeString()}</span>
        )}
      </div>
    </div>
  );
}

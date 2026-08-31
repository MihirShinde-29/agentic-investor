import { useMemo } from "react";
import { AlertTriangle, X } from "lucide-react";
import type { LiveEvent } from "@/hooks/useLiveEvents";
import { cn } from "@/lib/utils";

type Alert = {
  key: string;
  kind: "warning" | "danger" | "info";
  title: string;
  detail: string;
  ts: string;
};

function collectAlerts(events: LiveEvent[]): Alert[] {
  const alerts: Alert[] = [];
  // Only look at recent events (last 30 min) so old alerts don't linger.
  const cutoff = Date.now() - 30 * 60 * 1000;
  const recent = events.filter((e) => new Date(e.ts).getTime() >= cutoff);

  for (const e of recent) {
    if (e.event === "trade_plan") {
      const trades = Array.isArray(e.trades)
        ? (e.trades as Array<{ reason?: string; ticker?: string }>)
        : [];
      const forceCut = trades.find((t) => t.reason?.includes("force_loss_cut"));
      if (forceCut) {
        alerts.push({
          key: `force-cut-${e.ts}-${forceCut.ticker}`,
          kind: "danger",
          title: "Force loss-cut fired",
          detail: `${forceCut.ticker}: ${forceCut.reason}`,
          ts: e.ts,
        });
      }
      const haltedBuy = trades.find((t) => t.reason?.includes("halt_buys"));
      if (haltedBuy) {
        alerts.push({
          key: `halt-buy-${e.ts}-${haltedBuy.ticker}`,
          kind: "warning",
          title: "Halt-buys engaged",
          detail: `${haltedBuy.ticker}: drawdown threshold hit`,
          ts: e.ts,
        });
      }
    }
    if (e.event === "opinion_drift_skip" || e.event === "filter_skip") {
      // Count recent skips; a streak fires an info alert.
    }
  }

  // Detect a skip streak (>=3 skips in the recent window).
  const skips = recent.filter(
    (e) => e.event === "opinion_drift_skip" || e.event === "filter_skip",
  );
  if (skips.length >= 3) {
    alerts.push({
      key: `skip-streak-${skips[skips.length - 1].ts}`,
      kind: "info",
      title: `Filter skip streak (${skips.length})`,
      detail: "opinion drift filter is suppressing regens - check thresholds",
      ts: skips[skips.length - 1].ts,
    });
  }

  // Dedup by key, newest wins.
  const seen = new Map<string, Alert>();
  for (const a of alerts) seen.set(a.key, a);
  return Array.from(seen.values()).sort((a, b) =>
    b.ts.localeCompare(a.ts),
  );
}

const KIND_STYLES: Record<Alert["kind"], string> = {
  danger: "border-danger/40 bg-danger/10 text-danger",
  warning: "border-warning/40 bg-warning/10 text-warning",
  info: "border-primary/40 bg-primary/10 text-primary",
};

export function AlertBanner({
  events,
  onDismiss,
  dismissed,
}: {
  events: LiveEvent[];
  dismissed: Set<string>;
  onDismiss: (key: string) => void;
}) {
  const alerts = useMemo(
    () => collectAlerts(events).filter((a) => !dismissed.has(a.key)),
    [events, dismissed],
  );
  if (alerts.length === 0) return null;

  return (
    <div className="space-y-2">
      {alerts.slice(0, 3).map((a) => (
        <div
          key={a.key}
          className={cn(
            "flex items-start gap-3 rounded-lg border px-3 py-2",
            KIND_STYLES[a.kind],
          )}
        >
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-xs font-semibold">
              {a.title}
              <span className="text-[10px] font-normal opacity-70">
                {new Date(a.ts).toLocaleTimeString(undefined, {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
            <div className="mt-0.5 text-[11px] opacity-85">{a.detail}</div>
          </div>
          <button
            type="button"
            onClick={() => onDismiss(a.key)}
            className="ml-2 rounded-md p-1 opacity-60 hover:bg-white/10 hover:opacity-100"
            aria-label="Dismiss alert"
          >
            <X className="size-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}

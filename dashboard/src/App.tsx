import { Activity, Radio, TrendingUp } from "lucide-react";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { cn } from "@/lib/utils";

function StatusPill({ status }: { status: "connecting" | "open" | "closed" }) {
  const color =
    status === "open"
      ? "bg-success/20 text-success ring-success/40"
      : status === "closed"
        ? "bg-danger/20 text-danger ring-danger/40"
        : "bg-warning/20 text-warning ring-warning/40";
  const dot =
    status === "open"
      ? "bg-success animate-pulse"
      : status === "closed"
        ? "bg-danger"
        : "bg-warning animate-pulse";
  const label =
    status === "open" ? "LIVE" : status === "closed" ? "OFFLINE" : "CONNECTING…";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset",
        color,
      )}
    >
      <span className={cn("size-1.5 rounded-full", dot)} />
      {label}
    </span>
  );
}

function App() {
  const { events, status } = useLiveEvents(500);
  const lastFive = events.slice(-5).reverse();

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header strip */}
      <header className="border-b border-border/50 bg-card/40 backdrop-blur">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-lg bg-primary/15 text-primary ring-1 ring-primary/30">
              <TrendingUp className="size-5" />
            </div>
            <div>
              <h1 className="text-sm font-semibold leading-tight">
                Agentic Investor
              </h1>
              <p className="text-xs text-muted-foreground">
                Live paper trading dashboard
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-xs text-muted-foreground md:inline">
              events: {events.length}
            </span>
            <StatusPill status={status} />
          </div>
        </div>
      </header>

      {/* Main grid — Phase 1 skeleton */}
      <main className="mx-auto max-w-[1440px] px-6 py-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_360px]">
          <section className="space-y-4">
            <div className="rounded-xl border border-border/50 bg-card p-6 shadow-sm">
              <div className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
                <Activity className="size-4" />
                Phase 1 — pipe connected
              </div>
              <h2 className="text-2xl font-semibold tracking-tight">
                Waiting for the loop
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Start the paper loop with{" "}
                <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
                  --serve-dashboard
                </code>{" "}
                and events will begin streaming here. Charts, per-ticker
                cards, and the trade drill-down land in Phase 2.
              </p>
            </div>
            <div className="rounded-xl border border-dashed border-border/40 bg-card/30 p-6">
              <p className="text-sm text-muted-foreground">
                Portfolio equity curve · Positions table · Per-ticker charts
                <span className="text-xs italic"> — coming in Phase 2</span>
              </p>
            </div>
          </section>

          <aside className="rounded-xl border border-border/50 bg-card">
            <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Radio className="size-4 text-primary" /> Live event feed
              </div>
              <span className="text-xs text-muted-foreground">
                showing {lastFive.length} of {events.length}
              </span>
            </div>
            <div className="max-h-[calc(100vh-220px)] overflow-y-auto p-3">
              {lastFive.length === 0 && (
                <p className="px-2 py-8 text-center text-xs text-muted-foreground">
                  no events yet
                </p>
              )}
              <ul className="space-y-1.5">
                {lastFive.map((evt, i) => (
                  <li
                    key={`${evt.ts}-${i}`}
                    className="rounded-md border border-border/40 bg-background/40 p-2.5 text-xs"
                  >
                    <div className="flex items-center justify-between font-mono">
                      <span className="text-primary">{evt.event}</span>
                      <span className="text-muted-foreground">
                        {evt.ts?.slice(11, 19)}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

export default App;

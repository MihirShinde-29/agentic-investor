import { TrendingUp } from "lucide-react";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { cn } from "@/lib/utils";
import { HeaderStrip } from "@/components/HeaderStrip";
import { PortfolioChart } from "@/components/PortfolioChart";
import { PositionsTable } from "@/components/PositionsTable";
import { TickerGrid } from "@/components/TickerGrid";
import { EventFeed } from "@/components/EventFeed";

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

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-10 border-b border-border/50 bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-3">
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
              {events.length} events
            </span>
            <StatusPill status={status} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] space-y-4 px-6 py-6">
        <HeaderStrip events={events} />

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_360px]">
          <div className="space-y-4">
            <PortfolioChart />
            <PositionsTable />
            <TickerGrid />
          </div>
          <div className="lg:sticky lg:top-[80px] lg:h-[calc(100vh-100px)]">
            <EventFeed events={events} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

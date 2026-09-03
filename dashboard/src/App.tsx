import { useState } from "react";
import { TrendingUp } from "lucide-react";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { useLatestRecId, useSessionStartedAt } from "@/hooks/useLatestRecId";
import { useReplayEvents } from "@/hooks/useReplayEvents";
import { cn } from "@/lib/utils";
import { HeaderStrip } from "@/components/HeaderStrip";
import { PortfolioChart } from "@/components/PortfolioChart";
import { PositionsTable } from "@/components/PositionsTable";
import { TickerGrid } from "@/components/TickerGrid";
import { EventFeed } from "@/components/EventFeed";
import { FilterAttribution } from "@/components/FilterAttribution";
import { CalibrationMini } from "@/components/CalibrationMini";
import { CorrelationHeatmap } from "@/components/CorrelationHeatmap";
import { AlertBanner } from "@/components/AlertBanner";
import { HealthStrip } from "@/components/HealthStrip";
import { SessionPicker } from "@/components/SessionPicker";
import { WatchlistPanel } from "@/components/WatchlistPanel";
import { TimeframeSelector } from "@/components/TimeframeSelector";
import type { Timeframe } from "@/lib/timeframe";

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
  const { events: liveEvents, status } = useLiveEvents(500);
  const [selectedSession, setSelectedSession] = useState<string | "live">("live");
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());
  const [timeframe, setTimeframe] = useState<Timeframe>("1D");
  const replayEvents = useReplayEvents(selectedSession);
  const events = replayEvents ?? liveEvents;
  const recId = useLatestRecId(events);
  const sessionStartedAt = useSessionStartedAt(events);

  const dismissAlert = (key: string) =>
    setDismissedAlerts((prev) => new Set(prev).add(key));

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-20 border-b border-border/50 bg-background/90 backdrop-blur">
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
          <div className="flex items-center gap-2">
            <SessionPicker
              selected={selectedSession}
              onChange={setSelectedSession}
            />
            <FilterAttribution sessionStartedAt={sessionStartedAt} />
            <span className="hidden text-xs text-muted-foreground md:inline">
              {events.length} events
            </span>
            <StatusPill
              status={selectedSession === "live" ? status : "closed"}
            />
          </div>
        </div>
        <div className="mx-auto max-w-[1600px] px-6 pb-3">
          <HeaderStrip events={events} />
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] space-y-4 px-6 py-6">
        {/*
          Right-pad on desktop so the banner clears the fixed event feed
          (360px panel + 16px gap = 376px).
        */}
        <div className="lg:pr-[376px]">
          <AlertBanner
            events={events}
            dismissed={dismissedAlerts}
            onDismiss={dismissAlert}
          />
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_360px]">
          <div className="space-y-4">
            <div className="flex items-center justify-end">
              <TimeframeSelector value={timeframe} onChange={setTimeframe} />
            </div>
            <PortfolioChart
              sessionId={selectedSession === "live" ? undefined : selectedSession}
              timeframe={timeframe}
            />
            <WatchlistPanel />
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_320px]">
              <PositionsTable recId={recId} />
              <div className="space-y-4">
                <CalibrationMini />
                <CorrelationHeatmap />
              </div>
            </div>
            <TickerGrid recId={recId} timeframe={timeframe} />
          </div>
          {/*
            Empty grid cell reserves the 360px right rail so the left
            column stays 1fr. The real EventFeed is rendered outside the
            grid as a viewport-fixed panel (see below) so it can't ever
            release from its pinned position, no matter how tall the
            left column gets.
          */}
          <div className="hidden lg:block" aria-hidden="true" />
        </div>

        {/* Mobile fallback: below lg the feed renders inline under the graphs. */}
        <div className="lg:hidden">
          <EventFeed events={events} />
        </div>

        {/*
          Right-pad the health strip on desktop so it stops before the
          fixed-position event feed panel (360px wide + 16px gap).
        */}
        <div className="lg:pr-[376px]">
          <HealthStrip events={events} wsStatus={status} />
        </div>
      </main>

      {/*
        Viewport-fixed event feed (desktop only). pointer-events-none on
        the outer overlay lets clicks pass through the empty space to the
        chart underneath; the inner panel re-enables events for itself.
      */}
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-10 hidden lg:block lg:top-[235px] xl:top-[169px]">
        <div className="mx-auto flex h-full max-w-[1600px] justify-end px-6">
          <div className="pointer-events-auto w-[360px]"> 
            <EventFeed events={events} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

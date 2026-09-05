import { useEffect, useState } from "react";
import useSWR from "swr";
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
import { ArmPicker } from "@/components/ArmPicker";
import { ExperimentCompare } from "@/components/ExperimentCompare";
import type { Timeframe } from "@/lib/timeframe";
import type { ExperimentMeta } from "@/lib/api";
import { currentArm, fetcher } from "@/lib/api";

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

function useView(): "single" | "compare" {
  if (typeof window === "undefined") return "single";
  return new URLSearchParams(window.location.search).get("view") === "compare"
    ? "compare"
    : "single";
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
  const view = useView();
  const arm = currentArm();
  const { data: meta } = useSWR<ExperimentMeta>(
    "/api/experiment/meta",
    fetcher,
    { revalidateOnFocus: false },
  );

  const dismissAlert = (key: string) =>
    setDismissedAlerts((prev) => new Set(prev).add(key));

  // Sets data-arm-theme on <html> so index.css can shift the accent HSL.
  const themeKey = (() => {
    if (!meta || meta.mode === "single") return null;
    if (view === "compare") return "compare";
    return arm || meta.default_arm;
  })();
  useEffect(() => {
    if (typeof document === "undefined") return;
    if (themeKey) document.documentElement.setAttribute("data-arm-theme", themeKey);
    else document.documentElement.removeAttribute("data-arm-theme");
  }, [themeKey]);

  const headerLabel = (() => {
    if (!meta || meta.mode === "single") return "Live paper trading dashboard";
    if (view === "compare") return `Experiment ${meta.name} · comparison`;
    return `Experiment ${meta.name} · arm ${arm || meta.default_arm}`;
  })();

  return (
    <div className="min-h-screen bg-background text-foreground">
      {themeKey ? (
        <div
          className="h-[3px] w-full"
          style={{ backgroundColor: "hsl(var(--arm-accent))" }}
          aria-hidden
        />
      ) : null}
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
              <p className="text-xs text-muted-foreground">{headerLabel}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {meta && meta.mode === "experiment" ? (
              <ArmPicker meta={meta} currentArm={arm} view={view} />
            ) : null}
            {view === "single" ? (
              <>
                <TimeframeSelector value={timeframe} onChange={setTimeframe} />
                <SessionPicker
                  selected={selectedSession}
                  onChange={setSelectedSession}
                />
                <FilterAttribution sessionStartedAt={sessionStartedAt} />
                <span className="hidden text-xs text-muted-foreground md:inline">
                  {events.length} events
                </span>
              </>
            ) : null}
            <StatusPill
              status={selectedSession === "live" ? status : "closed"}
            />
          </div>
        </div>
        {view === "single" ? (
          <div className="mx-auto max-w-[1600px] px-6 pb-3">
            <HeaderStrip events={events} />
          </div>
        ) : null}
      </header>

      {view === "compare" ? (
        <main className="mx-auto max-w-[1600px] space-y-4 px-6 py-6">
          <ExperimentCompare />
        </main>
      ) : (
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
            <PortfolioChart
              sessionId={selectedSession === "live" ? undefined : selectedSession}
              timeframe={timeframe}
            />
            <WatchlistPanel />
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_320px]">
              <PositionsTable recId={recId} />
              <CalibrationMini />
            </div>
            <CorrelationHeatmap />
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
      )}

      {view === "single" ? (
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-10 hidden lg:block lg:top-[235px] xl:top-[169px]">
        <div className="mx-auto flex h-full max-w-[1600px] justify-end px-6">
          <div className="pointer-events-auto w-[360px]">
            <EventFeed events={events} />
          </div>
        </div>
      </div>
      ) : null}
    </div>
  );
}

export default App;

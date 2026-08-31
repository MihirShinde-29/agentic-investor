import { useMemo } from "react";
import useSWR from "swr";
import { History } from "lucide-react";
import { fetcher } from "@/lib/api";
import { cn } from "@/lib/utils";

type SessionListItem = { id: string; started_at: number };

/**
 * Parse a session id like "2026-08-31T06-37-45" into a Date.
 * Falls back to the started_at mtime if parsing fails.
 */
function parseSessionId(id: string, fallback: number): Date {
  const m = id.match(/^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})$/);
  if (m) {
    const [, day, hh, mm, ss] = m;
    return new Date(`${day}T${hh}:${mm}:${ss}Z`);
  }
  return new Date(fallback * 1000);
}

function formatDayLabel(d: Date): string {
  const today = new Date();
  const isToday = d.toDateString() === today.toDateString();
  const yest = new Date(today);
  yest.setDate(today.getDate() - 1);
  const isYesterday = d.toDateString() === yest.toDateString();
  if (isToday) return "Today";
  if (isYesterday) return "Yesterday";
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

function dayKey(d: Date): string {
  // YYYY-MM-DD -- stable across timezones for grouping.
  return `${d.getFullYear()}-${(d.getMonth() + 1).toString().padStart(2, "0")}-${d
    .getDate()
    .toString()
    .padStart(2, "0")}`;
}

export function SessionPicker({
  selected,
  onChange,
}: {
  selected: string | "live";
  onChange: (id: string | "live") => void;
}) {
  const { data } = useSWR<SessionListItem[]>("/api/sessions", fetcher, {
    refreshInterval: 60_000,
  });

  /**
   * One representative session per day (the latest of that day, which
   * carries the most cumulative state). Multiple runs on the same day get
   * collapsed into a single "Today", "Yesterday", or "Mon Aug 29" entry.
   */
  const days = useMemo(() => {
    const items = (data ?? []).map((s) => ({
      id: s.id,
      date: parseSessionId(s.id, s.started_at),
    }));
    const byDay = new Map<string, (typeof items)[number]>();
    for (const it of items) {
      const k = dayKey(it.date);
      const cur = byDay.get(k);
      if (!cur || it.date > cur.date) byDay.set(k, it);
    }
    return Array.from(byDay.values()).sort(
      (a, b) => b.date.getTime() - a.date.getTime(),
    );
  }, [data]);

  return (
    <label className="inline-flex items-center gap-2 rounded-md border border-border/40 bg-card/60 px-2 py-1 text-xs">
      <History className="size-3.5 text-muted-foreground" />
      <span className="text-muted-foreground">Session</span>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className={cn(
          "cursor-pointer appearance-none rounded bg-card py-0.5 pl-1 pr-6 font-mono text-xs text-foreground",
          "focus:outline-none focus:ring-1 focus:ring-primary/40",
        )}
        style={{ colorScheme: "dark" }}
      >
        <option value="live">Live</option>
        {days.map((it) => (
          <option key={it.id} value={it.id}>
            {formatDayLabel(it.date)}
          </option>
        ))}
      </select>
    </label>
  );
}

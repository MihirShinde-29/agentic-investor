import useSWR from "swr";
import { History } from "lucide-react";
import { fetcher } from "@/lib/api";
import { cn } from "@/lib/utils";

type SessionListItem = { id: string; started_at: number };

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

  const sessions = data ?? [];

  return (
    <label className="inline-flex items-center gap-2 rounded-md border border-border/40 bg-card/60 px-2 py-1 text-xs">
      <History className="size-3.5 text-muted-foreground" />
      <span className="text-muted-foreground">Session</span>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className={cn(
          "cursor-pointer rounded bg-transparent py-0.5 pl-1 pr-6 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-primary/40",
        )}
      >
        <option value="live">Live</option>
        {sessions.map((s) => (
          <option key={s.id} value={s.id}>
            {s.id}
          </option>
        ))}
      </select>
    </label>
  );
}

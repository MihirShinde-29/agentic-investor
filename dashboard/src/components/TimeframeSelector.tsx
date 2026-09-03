import { cn } from "@/lib/utils";
import { TIMEFRAME_ORDER, type Timeframe } from "@/lib/timeframe";

export function TimeframeSelector({
  value,
  onChange,
}: {
  value: Timeframe;
  onChange: (tf: Timeframe) => void;
}) {
  return (
    <div className="flex items-center gap-0.5 rounded-md border border-border/40 bg-card/40 p-0.5">
      {TIMEFRAME_ORDER.map((tf) => (
        <button
          key={tf}
          type="button"
          onClick={() => onChange(tf)}
          className={cn(
            "rounded-sm px-2 py-1 text-[11px] font-medium tabular transition-colors",
            value === tf
              ? "bg-primary/15 text-primary ring-1 ring-primary/30"
              : "text-muted-foreground hover:bg-muted/40 hover:text-foreground",
          )}
        >
          {tf}
        </button>
      ))}
    </div>
  );
}

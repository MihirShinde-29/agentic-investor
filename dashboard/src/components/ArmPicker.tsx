import { cn } from "@/lib/utils";
import type { ExperimentMeta } from "@/lib/api";

export function ArmPicker({
  meta,
  currentArm,
  view,
}: {
  meta: ExperimentMeta;
  currentArm: string | null;
  view: "single" | "compare";
}) {
  if (meta.mode !== "experiment") return null;

  const setUrl = (arm: string | null, nextView: "single" | "compare") => {
    const params = new URLSearchParams(window.location.search);
    if (arm) params.set("arm", arm);
    else params.delete("arm");
    if (nextView === "compare") params.set("view", "compare");
    else params.delete("view");
    const qs = params.toString();
    window.location.search = qs ? `?${qs}` : "";
  };

  const activeArm = currentArm || meta.default_arm;

  return (
    <div className="flex items-center gap-1.5 rounded-md border border-border/60 bg-muted/30 px-2 py-1 text-xs">
      <span className="text-muted-foreground">exp:</span>
      <span className="font-medium">{meta.name}</span>
      <span className="mx-1 text-muted-foreground">/</span>
      {meta.arms.map((a) => (
        <button
          key={a.id}
          onClick={() => setUrl(a.id, "single")}
          className={cn(
            "rounded px-1.5 py-0.5 font-medium transition-colors",
            view === "single" && a.id === activeArm
              ? "bg-primary/20 text-primary ring-1 ring-primary/40"
              : "text-muted-foreground hover:bg-muted",
          )}
          title={`arm ${a.id} (${a.account})`}
        >
          {a.id}
        </button>
      ))}
      <button
        onClick={() => setUrl(null, "compare")}
        className={cn(
          "ml-1 rounded px-1.5 py-0.5 font-medium transition-colors",
          view === "compare"
            ? "bg-primary/20 text-primary ring-1 ring-primary/40"
            : "text-muted-foreground hover:bg-muted",
        )}
        title="cross-arm comparison view"
      >
        compare
      </button>
    </div>
  );
}

import { cn } from "@/lib/utils";

type BadgeVariant =
  | "default"
  | "success"
  | "danger"
  | "warning"
  | "muted"
  | "primary";

const variantClasses: Record<BadgeVariant, string> = {
  default: "bg-secondary/60 text-secondary-foreground ring-border/40",
  success: "bg-success/15 text-success ring-success/30",
  danger: "bg-danger/15 text-danger ring-danger/30",
  warning: "bg-warning/15 text-warning ring-warning/30",
  muted: "bg-muted text-muted-foreground ring-border/30",
  primary: "bg-primary/15 text-primary ring-primary/30",
};

export function Badge({
  className,
  variant = "default",
  children,
}: {
  className?: string;
  variant?: BadgeVariant;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset",
        variantClasses[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

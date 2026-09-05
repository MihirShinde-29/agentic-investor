import useSWR from "swr";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { RecResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { cn, formatPct, formatUSD } from "@/lib/utils";

type ExtendedRec = RecResp & {
  target: string;
  tickers: string[];
  technical_signals: SignalRow[];
  news_signals: SignalRow[];
  violations: string[];
};

type SignalRow = {
  ticker: string;
  stance: string;
  confidence: number;
  reasoning: string;
  key_drivers?: string[];
};

function StanceIcon({ stance }: { stance: string }) {
  if (stance === "bullish") return <ArrowUp className="size-3 text-success" />;
  if (stance === "bearish") return <ArrowDown className="size-3 text-danger" />;
  return <Minus className="size-3 text-muted-foreground" />;
}

function stanceVariant(stance: string): "success" | "danger" | "muted" {
  if (stance === "bullish") return "success";
  if (stance === "bearish") return "danger";
  return "muted";
}

type Precedent = {
  rec_id: number;
  source: string;
  created_at: string;
  tickers: string[];
  similarity: number;
  text: string;
  outcome_pl_pct_15m: number | null;
  outcome_pl_pct_60m: number | null;
  outcome_pl_pct_1d: number | null;
  outcome_pl_pct_1w: number | null;
  prompt_line: string;
};

type PrecedentsResp = {
  rec_id: number;
  arm_id: string;
  query?: string;
  precedents: Precedent[];
  error?: string;
};

function outcomeChip(label: string, val: number | null) {
  if (val === null) return null;
  const color =
    val > 0
      ? "text-success"
      : val < 0
        ? "text-danger"
        : "text-muted-foreground";
  const sign = val > 0 ? "+" : "";
  return (
    <span className={cn("tabular text-[11px]", color)}>
      {label} {sign}
      {val.toFixed(2)}%
    </span>
  );
}

function PrecedentsSection({ recId }: { recId: number }) {
  const { data } = useSWR<PrecedentsResp>(
    `/api/rec/${recId}/precedents?k=4`,
    fetcher,
  );
  if (!data || !data.precedents || data.precedents.length === 0) return null;
  return (
    <section>
      <h3 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Similar past decisions
        <Badge variant="muted" className="font-mono text-[10px]">
          arm {data.arm_id}
        </Badge>
      </h3>
      <div className="space-y-2">
        {data.precedents.map((p) => (
          <div
            key={`${p.source}:${p.rec_id}`}
            className="rounded-lg border border-border/40 bg-card/40 p-3"
          >
            <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
              <span className="tabular text-muted-foreground">
                {p.created_at.slice(0, 10)}
              </span>
              <Badge variant="muted" className="font-mono text-[10px]">
                {p.source}
              </Badge>
              <span className="font-mono font-medium">
                {p.tickers.join(",")}
              </span>
              <span className="tabular text-[11px] text-muted-foreground">
                sim {(p.similarity * 100).toFixed(0)}%
              </span>
              <div className="ml-auto flex items-center gap-2">
                {outcomeChip("15m", p.outcome_pl_pct_15m)}
                {outcomeChip("60m", p.outcome_pl_pct_60m)}
                {outcomeChip("1d", p.outcome_pl_pct_1d)}
                {outcomeChip("1w", p.outcome_pl_pct_1w)}
              </div>
            </div>
            <p className="text-xs leading-relaxed text-foreground/80">
              {p.text.length > 240 ? p.text.slice(0, 239) + "…" : p.text}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const color =
    pct >= 70 ? "bg-success" : pct >= 40 ? "bg-warning" : "bg-danger";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted/40">
        <div
          className={cn("h-full transition-all", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="tabular text-[11px] text-muted-foreground">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

export function TradeDrillDown({
  open,
  onOpenChange,
  recId,
  focusTicker,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  recId: number | null;
  focusTicker?: string;
}) {
  const { data, error, isLoading } = useSWR<ExtendedRec>(
    open && recId != null ? `/api/rec/${recId}` : null,
    fetcher,
  );

  const focused = focusTicker?.toUpperCase();
  const focusedPos = data?.positions.find(
    (p) => p.ticker.toUpperCase() === focused,
  );
  const focusedTech = data?.technical_signals.find(
    (s) => s.ticker.toUpperCase() === focused,
  );
  const focusedNews = data?.news_signals.find(
    (s) => s.ticker.toUpperCase() === focused,
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <div className="flex items-center gap-2">
            <DialogTitle>
              {focused ? `Why ${focused}?` : "Recommendation detail"}
            </DialogTitle>
            {recId != null && (
              <Badge variant="muted" className="font-mono">
                rec #{recId}
              </Badge>
            )}
          </div>
          <DialogDescription>
            The signals + rationale the LLM used to size this position.
          </DialogDescription>
        </DialogHeader>

        <DialogBody className="space-y-5">
          {isLoading && (
            <p className="text-sm text-muted-foreground">loading…</p>
          )}
          {error && (
            <p className="text-sm text-danger">failed to load: {String(error)}</p>
          )}
          {data && (
            <>
              {focused && focusedPos && (
                <section className="rounded-lg border border-primary/30 bg-primary/5 p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant="primary" className="font-mono">
                        {focusedPos.ticker}
                      </Badge>
                      <span className="text-sm font-semibold">
                        target {focusedPos.weight_pct.toFixed(1)}% ·{" "}
                        {formatUSD(focusedPos.dollars, 0)}
                      </span>
                    </div>
                    <ConfidenceBar value={focusedPos.confidence} />
                  </div>
                  <p className="text-sm leading-relaxed text-foreground/90">
                    {focusedPos.rationale}
                  </p>
                </section>
              )}

              {focused && (focusedTech || focusedNews) && (
                <section>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Signals used
                  </h3>
                  <div className="grid gap-3 md:grid-cols-2">
                    {focusedTech && (
                      <div className="rounded-lg border border-border/40 bg-card/60 p-3">
                        <div className="mb-1.5 flex items-center gap-2">
                          <StanceIcon stance={focusedTech.stance} />
                          <span className="text-xs font-semibold uppercase text-muted-foreground">
                            technical
                          </span>
                          <Badge variant={stanceVariant(focusedTech.stance)}>
                            {focusedTech.stance}
                          </Badge>
                          <ConfidenceBar value={focusedTech.confidence} />
                        </div>
                        <p className="text-xs leading-relaxed text-foreground/85">
                          {focusedTech.reasoning}
                        </p>
                        {focusedTech.key_drivers &&
                          focusedTech.key_drivers.length > 0 && (
                            <ul className="mt-2 space-y-0.5">
                              {focusedTech.key_drivers.slice(0, 4).map((d, i) => (
                                <li
                                  key={i}
                                  className="text-[11px] text-muted-foreground before:mr-1 before:content-['·']"
                                >
                                  {d}
                                </li>
                              ))}
                            </ul>
                          )}
                      </div>
                    )}
                    {focusedNews && (
                      <div className="rounded-lg border border-border/40 bg-card/60 p-3">
                        <div className="mb-1.5 flex items-center gap-2">
                          <StanceIcon stance={focusedNews.stance} />
                          <span className="text-xs font-semibold uppercase text-muted-foreground">
                            news
                          </span>
                          <Badge variant={stanceVariant(focusedNews.stance)}>
                            {focusedNews.stance}
                          </Badge>
                          <ConfidenceBar value={focusedNews.confidence} />
                        </div>
                        <p className="text-xs leading-relaxed text-foreground/85">
                          {focusedNews.reasoning}
                        </p>
                      </div>
                    )}
                  </div>
                </section>
              )}

              <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Portfolio rationale
                </h3>
                <p className="text-sm leading-relaxed text-foreground/85">
                  {data.portfolio_rationale}
                </p>
              </section>

              {recId != null && <PrecedentsSection recId={recId} />}

              <section>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Full allocation
                </h3>
                <div className="overflow-hidden rounded-lg border border-border/40">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/20">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                          Ticker
                        </th>
                        <th className="px-3 py-2 text-right font-medium text-muted-foreground">
                          Weight
                        </th>
                        <th className="px-3 py-2 text-right font-medium text-muted-foreground">
                          Dollars
                        </th>
                        <th className="px-3 py-2 text-right font-medium text-muted-foreground">
                          Confidence
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.positions.map((p) => {
                        const isFocus = p.ticker.toUpperCase() === focused;
                        return (
                          <tr
                            key={p.ticker}
                            className={cn(
                              "border-t border-border/30",
                              isFocus && "bg-primary/5",
                            )}
                          >
                            <td className="px-3 py-1.5 font-mono font-medium">
                              {p.ticker}
                            </td>
                            <td className="px-3 py-1.5 text-right tabular">
                              {p.weight_pct.toFixed(1)}%
                            </td>
                            <td className="px-3 py-1.5 text-right tabular text-muted-foreground">
                              {formatUSD(p.dollars, 0)}
                            </td>
                            <td className="px-3 py-1.5 text-right tabular">
                              {p.confidence.toFixed(2)}
                            </td>
                          </tr>
                        );
                      })}
                      <tr className="border-t border-border/30 bg-muted/10">
                        <td className="px-3 py-1.5 font-mono font-medium text-muted-foreground">
                          CASH
                        </td>
                        <td className="px-3 py-1.5 text-right tabular text-muted-foreground">
                          {data.cash_pct.toFixed(1)}%
                        </td>
                        <td className="px-3 py-1.5 text-right tabular text-muted-foreground">
                          {formatUSD(data.cash_dollars, 0)}
                        </td>
                        <td className="px-3 py-1.5"></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </section>

              {(() => {
                // Split violations into ticker-specific (mention the focused
                // ticker as a whole word) and portfolio-wide (everything
                // else). Without a focus we treat all as portfolio-wide.
                const tickerRe = focused
                  ? new RegExp(`\\b${focused}\\b`, "i")
                  : null;
                const tickerHits = tickerRe
                  ? data.violations.filter((v) => tickerRe.test(v))
                  : [];
                const other = data.violations.filter(
                  (v) => !tickerHits.includes(v),
                );
                if (tickerHits.length === 0 && other.length === 0) return null;
                return (
                  <section className="space-y-3">
                    {tickerHits.length > 0 && (
                      <div>
                        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-warning">
                          Violations on {focused}
                        </h3>
                        <ul className="space-y-1">
                          {tickerHits.map((v, i) => (
                            <li
                              key={i}
                              className="rounded-md border border-warning/30 bg-warning/5 px-3 py-1.5 text-xs text-warning"
                            >
                              {v}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {other.length > 0 && (
                      <div>
                        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Other portfolio violations
                        </h3>
                        <ul className="space-y-1">
                          {other.map((v, i) => (
                            <li
                              key={i}
                              className="rounded-md border border-border/40 bg-muted/10 px-3 py-1.5 text-xs text-muted-foreground"
                            >
                              {v}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </section>
                );
              })()}

              <div className="grid grid-cols-3 gap-2 border-t border-border/40 pt-3 text-[11px]">
                <div>
                  <div className="text-muted-foreground">Amount</div>
                  <div className="tabular font-medium">
                    {formatUSD(data.amount, 0)}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Risk</div>
                  <div className="font-medium">{data.risk}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Target</div>
                  <div className="font-medium">{data.target}</div>
                </div>
              </div>
            </>
          )}
        </DialogBody>
      </DialogContent>
    </Dialog>
  );
}

export { formatPct };

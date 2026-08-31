import useSWR from "swr";
import { Network } from "lucide-react";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type CorrelationResp = {
  tickers: string[];
  matrix: number[][];
  window_days: number;
  error?: string;
};

/**
 * Map a correlation value in [-1, 1] to a CSS background color.
 * Positive: blue (concentration risk). Negative: teal (hedges).
 *
 * Note: uses the classic comma-form `hsla(h, s%, l%, a)` -- mixing modern
 * space-form HSL with a comma alpha is invalid CSS and produces
 * transparent cells in most browsers.
 */
function cellStyle(v: number): React.CSSProperties {
  const clamped = Math.max(-1, Math.min(1, v));
  const alpha = Math.abs(clamped);
  const [h, s, l] =
    clamped >= 0 ? [217, 92, 60] : [142, 71, 45];
  return {
    backgroundColor: `hsla(${h}, ${s}%, ${l}%, ${(alpha * 0.85).toFixed(2)})`,
    color: alpha > 0.5 ? "white" : "hsl(240, 5%, 90%)",
  };
}

export function CorrelationHeatmap() {
  const { data } = useSWR<CorrelationResp>(
    "/api/correlation",
    fetcher,
    { refreshInterval: 5 * 60_000 },
  );

  const tickers = data?.tickers ?? [];
  const matrix = data?.matrix ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Network className="size-4 text-primary" />
          <CardTitle>Correlation</CardTitle>
        </div>
        <span className="text-xs text-muted-foreground">
          {data?.window_days ?? 60}d · daily returns
        </span>
      </CardHeader>
      <CardContent className="p-3">
        {tickers.length < 2 ? (
          <div className="grid h-32 place-items-center text-center text-xs text-muted-foreground">
            <div>
              <p>need 2+ positions</p>
              <p className="mt-1 opacity-60">for a pairwise heatmap</p>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[10px]">
              <thead>
                <tr>
                  <th className="p-1" />
                  {tickers.map((t) => (
                    <th
                      key={t}
                      className="p-1 text-center font-mono font-medium text-muted-foreground"
                    >
                      {t}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map((row, i) => (
                  <tr key={tickers[i]}>
                    <th className="p-1 text-right font-mono font-medium text-muted-foreground">
                      {tickers[i]}
                    </th>
                    {row.map((v, j) => (
                      <td
                        key={`${tickers[i]}-${tickers[j]}`}
                        className={cn(
                          "border border-background/70 p-1 text-center font-mono tabular",
                          i === j && "opacity-40",
                        )}
                        style={cellStyle(v)}
                        title={`${tickers[i]} vs ${tickers[j]}: ${v.toFixed(3)}`}
                      >
                        {v.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-2 flex items-center justify-between text-[10px] text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <span
                  className="inline-block size-2 rounded-sm"
                  style={{ background: "hsla(217, 92%, 60%, 0.85)" }}
                />
                positive = concentration
              </span>
              <span className="inline-flex items-center gap-1">
                <span
                  className="inline-block size-2 rounded-sm"
                  style={{ background: "hsla(142, 71%, 45%, 0.85)" }}
                />
                negative = hedge
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

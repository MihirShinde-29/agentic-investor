import useSWR from "swr";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts";
import { Target } from "lucide-react";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type CalibrationResp = {
  horizon_minutes: number;
  n_trades: number;
  overall_win_rate: number;
  buckets: Array<{
    lo: number;
    hi: number;
    n_trades: number;
    mean_confidence: number;
    win_rate: number;
  }>;
  error?: string;
};

export function CalibrationMini({ horizonMinutes = 60 }: { horizonMinutes?: number }) {
  const { data } = useSWR<CalibrationResp>(
    `/api/calibration?horizon_minutes=${horizonMinutes}&n_buckets=5`,
    fetcher,
    { refreshInterval: 60_000 },
  );

  const points = (data?.buckets ?? [])
    .filter((b) => b.n_trades > 0)
    .map((b) => ({ x: b.mean_confidence, y: b.win_rate, z: b.n_trades }));

  const perfectLine: [{ x: number; y: number }, { x: number; y: number }] = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Target className="size-4 text-primary" />
          <CardTitle>Confidence calibration</CardTitle>
        </div>
        <span className="text-xs text-muted-foreground">
          {data?.n_trades ?? 0} trades · {horizonMinutes}m horizon
        </span>
      </CardHeader>
      <CardContent className="p-2">
        {(!data || data.n_trades === 0) ? (
          <div className="grid h-40 place-items-center text-center text-xs text-muted-foreground">
            <div>
              <p>no resolvable trades yet</p>
              <p className="mt-1 opacity-60">
                fills + intraday prices needed
              </p>
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <ScatterChart margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="hsl(240 3.7% 15.9%)" strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="x"
                domain={[0, 1]}
                tickFormatter={(v) => v.toFixed(1)}
                tick={{ fill: "hsl(240 5% 64.9%)", fontSize: 10 }}
                stroke="hsl(240 3.7% 20%)"
                label={{
                  value: "confidence",
                  position: "insideBottom",
                  offset: -2,
                  fill: "hsl(240 5% 64.9%)",
                  fontSize: 10,
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: "hsl(240 5% 64.9%)", fontSize: 10 }}
                stroke="hsl(240 3.7% 20%)"
                width={38}
              />
              <ReferenceLine
                segment={perfectLine}
                stroke="hsl(240 5% 64.9%)"
                strokeDasharray="4 4"
              />
              <Scatter
                data={points}
                fill="hsl(217 92% 60%)"
                shape="circle"
              />
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

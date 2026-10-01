import useSWR from "swr";
import { useMemo } from "react";
import type { EodResp } from "@/lib/api";
import { fetcher } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/** Minimal markdown renderer for the subset the EOD report uses:
 * headings, bold, inline-code, fenced code, bullet lists. Avoids
 * pulling in a markdown dep for one view. */
function renderMarkdown(src: string): JSX.Element {
  const lines = src.split(/\r?\n/);
  const out: JSX.Element[] = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    // Fenced code block
    if (line.startsWith("```")) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        buf.push(lines[i]);
        i++;
      }
      i++; // consume closing fence
      out.push(
        <pre key={key++}
             className="my-2 overflow-x-auto rounded bg-muted/30 p-3 text-xs leading-tight font-mono">
          {buf.join("\n")}
        </pre>,
      );
      continue;
    }
    // Headings
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const level = h[1].length;
      const text = inline(h[2], key);
      const cls = level === 1 ? "text-xl font-semibold mt-6 mb-2"
        : level === 2 ? "text-lg font-semibold mt-5 mb-2"
        : level === 3 ? "text-base font-semibold mt-4 mb-1"
        : "text-sm font-semibold mt-3 mb-1";
      // Only h1/h2/h3 render as proper headings; deeper falls through
      // to <div>.
      if (level === 1) out.push(<h1 key={key++} className={cls}>{text}</h1>);
      else if (level === 2) out.push(<h2 key={key++} className={cls}>{text}</h2>);
      else if (level === 3) out.push(<h3 key={key++} className={cls}>{text}</h3>);
      else out.push(<div key={key++} className={cls}>{text}</div>);
      i++;
      continue;
    }
    // Bulleted list block
    if (/^\s*-\s+/.test(line)) {
      const items: JSX.Element[] = [];
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        const content = lines[i].replace(/^\s*-\s+/, "");
        items.push(<li key={key++}>{inline(content, key)}</li>);
        i++;
      }
      out.push(
        <ul key={key++} className="my-2 list-disc space-y-0.5 pl-5 text-sm">
          {items}
        </ul>,
      );
      continue;
    }
    // Blank line between blocks
    if (!line.trim()) {
      i++;
      continue;
    }
    // Paragraph fallback
    out.push(<p key={key++} className="my-2 text-sm leading-relaxed">{inline(line, key)}</p>);
    i++;
  }
  return <>{out}</>;
}

/** Inline markdown: **bold** and `code`. Keeps it short, avoids a
 * full tokenizer. Returns a React fragment of mixed nodes. */
function inline(text: string, baseKey: number): JSX.Element {
  // Split on `code` first (highest precedence), then **bold**.
  const parts: JSX.Element[] = [];
  let i = 0;
  let seg = 0;
  while (i < text.length) {
    if (text[i] === "`") {
      const end = text.indexOf("`", i + 1);
      if (end === -1) { parts.push(<span key={baseKey * 100 + seg++}>{text.slice(i)}</span>); break; }
      parts.push(
        <code key={baseKey * 100 + seg++}
              className="rounded bg-muted/40 px-1 font-mono text-[0.85em]">
          {text.slice(i + 1, end)}
        </code>,
      );
      i = end + 1;
      continue;
    }
    if (text.startsWith("**", i)) {
      const end = text.indexOf("**", i + 2);
      if (end === -1) { parts.push(<span key={baseKey * 100 + seg++}>{text.slice(i)}</span>); break; }
      parts.push(
        <strong key={baseKey * 100 + seg++} className="font-semibold">
          {text.slice(i + 2, end)}
        </strong>,
      );
      i = end + 2;
      continue;
    }
    // Find next special
    const nextTick = text.indexOf("`", i);
    const nextBold = text.indexOf("**", i);
    let next = text.length;
    if (nextTick !== -1 && nextTick < next) next = nextTick;
    if (nextBold !== -1 && nextBold < next) next = nextBold;
    parts.push(<span key={baseKey * 100 + seg++}>{text.slice(i, next)}</span>);
    i = next;
  }
  return <>{parts}</>;
}

export function EodView() {
  const { data } = useSWR<EodResp>(
    "/api/experiment/eod",
    fetcher,
    { refreshInterval: 60_000 },
  );
  const body = useMemo(
    () => (data?.markdown ? renderMarkdown(data.markdown) : null),
    [data?.markdown],
  );
  return (
    <div className="mx-auto max-w-[1200px] px-6 py-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">
            End-of-day report
            {data?.date ? (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                {data.date}
              </span>
            ) : null}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {body ?? (
            <div className="text-xs text-muted-foreground">
              {data === undefined
                ? "loading…"
                : "no EOD report found. run scripts/eod_summary.py to generate one."}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Decode the HTML entities Alpaca (and its upstream sources) leave in
 * news headlines and summaries. The feed pre-encodes text for a web
 * page and we store it raw, so `NBC&#39;s` needs to render as `NBC's`
 * anywhere we surface a headline.
 *
 * Handles the four canonical named entities, both decimal and hex
 * numeric refs, and strips dangling partial entities left over when
 * an upstream feed truncated mid-entity (e.g. "...Flip&#3" -> "...Flip").
 * Deliberately narrow - not a general-purpose HTML sanitizer, just
 * enough to keep news copy readable.
 */
export function decodeHtmlEntities(s: string): string {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&#x([\da-fA-F]+);/g, (_, n) =>
      String.fromCharCode(parseInt(n, 16)),
    )
    // Strip trailing partial entities like `&#3` or `&am` after
    // upstream cut-off; keeps the visible string clean rather than
    // leaving a half-encoded suffix.
    .replace(/&#x?[\da-fA-F]*$/, "")
    .replace(/&[a-zA-Z]{1,6}$/, "")
    .trimEnd();
}

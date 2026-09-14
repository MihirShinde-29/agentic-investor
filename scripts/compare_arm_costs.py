import pathlib
import re

LINE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[tick_cost\]"
    r".*cached_tokens=(\d+).*cache_hit_pct=([\d.]+).*cost_usd=\$([\d.]+)"
)


def parse(arm: str):
    p = pathlib.Path(f"out/experiments/reasoning-quality/{arm}.log")
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.search(line)
        if m:
            rows.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


for arm in ("A", "B", "C"):
    r = parse(arm)
    if not r:
        continue
    total = sum(x[3] for x in r)
    last5 = r[-5:]
    avg_cost5 = sum(x[3] for x in last5) / len(last5)
    avg_cache5 = sum(x[2] for x in last5) / len(last5)
    print(f"{arm}: ticks={len(r)}, total=${total:.4f}, last5 avg=${avg_cost5:.4f}, last5 avg cached_tok={avg_cache5:.0f}")
    print(f"   first {r[0][0]} -> last {r[-1][0]}")

c = parse("C")
if c:
    pre = [x for x in c if x[0] < "2026-09-14 12:04"]
    post = [x for x in c if x[0] >= "2026-09-14 12:04"]
    if pre and post:
        pre_avg = sum(x[3] for x in pre) / len(pre)
        post_avg = sum(x[3] for x in post) / len(post)
        pre_cache = sum(x[1] for x in pre) / len(pre)
        post_cache = sum(x[1] for x in post) / len(post)
        pct = (post_avg - pre_avg) / pre_avg * 100
        print(f"\nC pre-fix  ({len(pre):>2} ticks): avg cost=${pre_avg:.4f}  avg cached_tok={pre_cache:.0f}")
        print(f"C post-fix ({len(post):>2} ticks): avg cost=${post_avg:.4f}  avg cached_tok={post_cache:.0f}")
        print(f"C delta: {pct:+.1f}%  (${post_avg-pre_avg:+.4f}/tick)")

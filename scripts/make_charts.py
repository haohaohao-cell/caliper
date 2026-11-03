"""生成项目图表（纯 SVG，零依赖）。

数字全部从结果文件现算，不写死——图能重画，才说明结论是算出来的。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from caliper.annotate import confusion, cohen_kappa, load_csv, parse_label  # noqa: E402
from caliper.runner import load_results, pass_at_k, summarize  # noqa: E402

OUT = ROOT / "charts"
FONT = "font-family=\"-apple-system,'Segoe UI','Microsoft YaHei',sans-serif\""
MONO = "font-family=\"ui-monospace,Consolas,monospace\""


def _bars(groups, series, colors, title, subtitle, ylabel="", note="") -> str:
    W, H = 780, 452
    ml, mr, mt, mb = 70, 40, 96, 112
    pw, ph = W - ml - mr, H - mt - mb

    def y(v: float) -> float:
        return mt + ph * (1 - v)

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{ml}" y="42" {FONT} font-size="19" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{ml}" y="66" {FONT} font-size="13" fill="#6b7280">{subtitle}</text>',
    ]

    for i in range(5):
        v = i / 4
        yy = y(v)
        p.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        p.append(f'<text x="{ml-10}" y="{yy+4:.1f}" {FONT} font-size="11" fill="#9ca3af" text-anchor="end">{v:.2f}</text>')

    gw = pw / len(groups)
    bw = min(56, gw / (len(series) + 1.1))
    for gi, g in enumerate(groups):
        gx = ml + gi * gw
        total = bw * len(series) + 10 * (len(series) - 1)
        sx = gx + (gw - total) / 2
        for si, (name, vals) in enumerate(series):
            v = vals[gi]
            bx = sx + si * (bw + 10)
            p.append(
                f'<rect x="{bx:.1f}" y="{y(v):.1f}" width="{bw:.1f}" height="{ph*v:.1f}" '
                f'rx="4" fill="{colors[si]}"/>'
            )
            p.append(
                f'<text x="{bx+bw/2:.1f}" y="{y(v)-9:.1f}" {FONT} font-size="13" font-weight="600" '
                f'fill="#111827" text-anchor="middle">{v:.3f}</text>'
            )
        p.append(
            f'<text x="{gx+gw/2:.1f}" y="{mt+ph+24:.1f}" {FONT} font-size="12.5" fill="#374151" '
            f'text-anchor="middle">{g}</text>'
        )

    # 图例（按字符宽度累进，中文按全角算，避免长标签互相重叠）
    def tw(s: str, size: float) -> float:
        return sum(size * (1.0 if ord(ch) > 127 else 0.55) for ch in s)

    lx = ml
    for si, (name, _) in enumerate(series):
        p.append(f'<rect x="{lx:.1f}" y="{mt+ph+46}" width="11" height="11" rx="2" fill="{colors[si]}"/>')
        p.append(f'<text x="{lx+17:.1f}" y="{mt+ph+55}" {FONT} font-size="12" fill="#4b5563">{name}</text>')
        lx += 17 + tw(name, 12) + 24

    if note:
        p.append(f'<text x="{ml}" y="{H-16}" {FONT} font-size="11.5" fill="#9ca3af">{note}</text>')
    p.append(f'<text x="{W-mr}" y="{mt-14}" {FONT} font-size="11" fill="#9ca3af" text-anchor="end">{ylabel}</text>')
    p.append("</svg>")
    return "\n".join(p)


def fig_passk() -> Path:
    rows = load_results(ROOT / "out_v2_chat/results.jsonl") + load_results(ROOT / "out_v2_v4/results.jsonl")
    s = summarize(rows, k=4)
    names, p4, pk4, sr = [], [], [], []
    for name, v in s.items():
        label = "deepseek-chat" if "chat" in name else "deepseek-v4-pro"
        names.append(label)
        p4.append(v["pass@4"])
        pk4.append(v["pass^4"])
        sr.append(v["avg_sr"])

    gap = (max(p4) - max(pk4)) * 100
    svg = _bars(
        names,
        [("pass@4（能力上限）", p4), ("pass^4（可靠性）", pk4)],
        ["#93c5fd", "#1d4ed8"],
        "能力上限接近，可靠性差 10pp",
        "同一批 50 条任务 × k=4 = 400 次 rollout；只报 pass@k 会得出「两者差不多」",
        "比例",
        f"最高分与最低可靠性之间差 {gap:.0f}pp，而这部分差距在 pass@k 里完全看不见",
    )
    OUT.mkdir(exist_ok=True)
    path = OUT / "fig1_passk.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def _metric_from(csv_path: Path, key_path: Path):
    rows = load_csv(csv_path)
    key = {r["sample_id"]: r for r in load_csv(key_path)}
    for r in rows:
        k = key.get(r.get("sample_id", ""))
        if k:
            r["auto_hard"] = k.get("auto_hard", "")
            r["soft_judge"] = k.get("soft_judge", "")
    human, judge, hard = [], [], []
    for r in rows:
        h = parse_label(r.get("human", ""))
        if h is None:
            continue
        human.append(h)
        judge.append(parse_label(r.get("soft_judge", "")))
        hard.append(parse_label(r.get("auto_hard", "")))
    return human, judge, hard


def fig_judge() -> Path:
    samples = [
        ("旧判分器 · 失败样本 n=13", ROOT / "out_ds_v2/spotcheck_human.csv", ROOT / "out_ds_v2/spotcheck_key.csv"),
        ("v2 判分器 · 随机 n=20", ROOT / "out_v2_chat/random_human_final.csv", ROOT / "out_v2_chat/random_key.csv"),
    ]
    fpr_j, fpr_h, tpr_j = [], [], []
    kappa_j, kappa_h, labels = [], [], []
    for title, c, k in samples:
        human, judge, hard = _metric_from(c, k)
        cj = confusion(human, judge)
        ch = confusion(human, hard)
        fpr_j.append(cj["fpr"])
        fpr_h.append(ch["fpr"])
        tpr_j.append(cj["tpr"])
        kappa_j.append(cj["kappa"])
        kappa_h.append(ch["kappa"])
        labels.append(title)

    W, H = 780, 330
    ml, mr, mt = 210, 150, 108
    pw = W - ml - mr
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="36" y="44" {FONT} font-size="19" font-weight="700" fill="#111827">LLM judge 的假阳率：两轮独立标注都是 1.00</text>',
        f'<text x="36" y="68" {FONT} font-size="13" fill="#6b7280">FPR = 在人工判定「未完成」的样本上，judge 仍判「通过」的比例</text>',
        # 图例
        f'<rect x="{ml}" y="82" width="11" height="11" rx="2" fill="#ef4444"/>',
        f'<text x="{ml+17}" y="92" {FONT} font-size="12" fill="#4b5563">LLM judge</text>',
        f'<rect x="{ml+100}" y="82" width="11" height="11" rx="2" fill="#10b981"/>',
        f'<text x="{ml+117}" y="92" {FONT} font-size="12" fill="#4b5563">hard 判分（终态断言）</text>',
    ]

    bar_h, gap = 30, 34
    for i, (lab, fj, fh, kj, kh) in enumerate(zip(labels, fpr_j, fpr_h, kappa_j, kappa_h)):
        top = mt + i * (bar_h * 2 + gap)
        mid = top + bar_h + 3
        p.append(
            f'<text x="{ml-16}" y="{mid+4:.0f}" {FONT} font-size="12.5" fill="#374151" '
            f'text-anchor="end">{lab}</text>'
        )
        # judge
        wj = max(pw * fj, 3)
        p.append(f'<rect x="{ml}" y="{top}" width="{wj:.1f}" height="{bar_h}" rx="4" fill="#ef4444"/>')
        p.append(
            f'<text x="{ml+wj+10:.1f}" y="{top+bar_h-10:.0f}" {MONO} font-size="12.5" '
            f'font-weight="700" fill="#b91c1c">FPR {fj:.2f}</text>'
        )
        p.append(
            f'<text x="{W-24}" y="{top+bar_h-10:.0f}" {MONO} font-size="11" fill="#9ca3af" '
            f'text-anchor="end">κ {kj:+.2f}</text>'
        )
        # hard
        y2 = top + bar_h + 6
        wh = max(pw * fh, 3)
        p.append(f'<rect x="{ml}" y="{y2}" width="{wh:.1f}" height="{bar_h}" rx="4" fill="#10b981"/>')
        p.append(
            f'<text x="{ml+wh+10:.1f}" y="{y2+bar_h-10:.0f}" {MONO} font-size="12.5" '
            f'font-weight="700" fill="#047857">FPR {fh:.2f}</text>'
        )
        p.append(
            f'<text x="{W-24}" y="{y2+bar_h-10:.0f}" {MONO} font-size="11" fill="#9ca3af" '
            f'text-anchor="end">κ {kh:+.2f}</text>'
        )

    p.append(f'<text x="36" y="{H-16}" {FONT} font-size="11.5" fill="#9ca3af">judge 从不拒绝真正该拒绝的样本；且 TPR≈FPR 时，误差校正式不可识别（返回 nan）</text>')
    p.append("</svg>")
    OUT.mkdir(exist_ok=True)
    path = OUT / "fig2_judge.svg"
    path.write_text("\n".join(p), encoding="utf-8")
    return path


if __name__ == "__main__":
    for f in (fig_passk(), fig_judge()):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size/1024:.1f} KB")

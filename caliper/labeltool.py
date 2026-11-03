"""标注工具生成器。

设计目标：**让人没法偷看机器的判断**。

不是在 CSV 里隐藏 auto_hard / soft_judge 两列（手动取消隐藏就会看到），
而是生成两个文件：
  - 盲标文件：只含题面、工具轨迹、最终回复，**物理上不存在**机器判定
  - 密钥文件：sample_id → 机器判定，用于事后合并

同时提供一个单文件 HTML 工具，按 1/0 键就能标，不用打开 Excel。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _rows_from_csv(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "sample_id": r["sample_id"],
                "task_id": r["task_id"],
                "index": r.get("index", ""),
                "instruction": r.get("instruction", ""),
                "tools": r.get("tools", ""),
                "final_text": r.get("final_text", ""),
                "auto_hard": r.get("auto_hard", ""),
                "soft_judge": r.get("soft_judge", ""),
            }
        )
    return out


def build_html(samples: list[dict[str, Any]], title: str = "标注") -> str:
    payload = json.dumps(
        [
            {
                "id": s["sample_id"],
                "task": s["task_id"],
                "q": s["instruction"],
                "tools": s["tools"],
                "ans": s["final_text"],
            }
            for s in samples
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{ --fg:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --ok:#059669; --no:#dc2626; --bg:#fafafa; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         color:var(--fg); background:var(--bg); padding:24px; }}
  .wrap {{ max-width:820px; margin:0 auto; }}
  .bar {{ display:flex; align-items:center; gap:12px; margin-bottom:16px; font-size:13px; color:var(--dim); }}
  .prog {{ flex:1; height:6px; background:var(--line); border-radius:3px; overflow:hidden; }}
  .prog i {{ display:block; height:100%; background:#2563eb; width:0; transition:width .2s; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:24px; }}
  .tid {{ font:600 12px ui-monospace,monospace; color:var(--dim); letter-spacing:.04em; }}
  .q {{ font-size:17px; line-height:1.6; margin:10px 0 18px; }}
  .lbl {{ font-size:11px; font-weight:600; color:var(--dim); letter-spacing:.08em; margin-bottom:6px; }}
  .tools {{ font:13px ui-monospace,monospace; background:#f3f4f6; border-radius:8px;
            padding:10px 12px; margin-bottom:16px; word-break:break-all; }}
  .ans {{ font-size:15px; line-height:1.7; background:#f9fafb; border-left:3px solid var(--line);
          padding:12px 14px; border-radius:0 8px 8px 0; min-height:44px; white-space:pre-wrap; }}
  .acts {{ display:flex; gap:12px; margin-top:22px; }}
  button {{ font-size:15px; font-weight:600; padding:12px 22px; border-radius:9px;
            border:1px solid var(--line); background:#fff; cursor:pointer; transition:.12s; }}
  button:hover {{ transform:translateY(-1px); }}
  .yes {{ background:var(--ok); border-color:var(--ok); color:#fff; }}
  .no  {{ background:var(--no);  border-color:var(--no);  color:#fff; }}
  .skip{{ margin-left:auto; color:var(--dim); font-weight:500; }}
  kbd {{ font:11px ui-monospace,monospace; background:#000; color:#fff; border-radius:4px;
         padding:1px 5px; margin-right:5px; opacity:.75; }}
  .done {{ text-align:center; padding:60px 20px; }}
  .done h2 {{ margin:0 0 8px; }}
  .dl {{ margin-top:20px; font-size:14px; padding:10px 18px; }}
  .mini {{ font-size:12px; padding:5px 12px; border-radius:7px; font-weight:500; }}
  .hint {{ margin-top:14px; font-size:12px; color:var(--dim); }}
</style></head><body>
<div class="wrap">
  <div class="bar"><span id="pos">0/0</span><div class="prog"><i id="bar"></i></div><span id="left"></span><button class="mini" onclick="exportCsv()">导出备份</button></div>
  <div id="app"></div>
  <div class="hint">按键：<kbd>1</kbd>完成　<kbd>0</kbd>未完成　<kbd>←</kbd>上一条　标注自动存在本机浏览器里</div>
</div>
<script>
const DATA = {payload};
const KEY = 'judgebench_labels_v1';
let labels = {{}};
try {{ labels = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{ labels = {{}}; }}
let i = 0;

function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(labels)); }} catch (e) {{}} }}

function render() {{
  const app = document.getElementById('app');
  const total = DATA.length;
  const done = DATA.filter(d => labels[d.id] !== undefined).length;
  document.getElementById('pos').textContent = (i+1) + '/' + total;
  document.getElementById('left').textContent = '已标 ' + done;
  document.getElementById('bar').style.width = (done/total*100) + '%';

  if (i >= total) {{
    app.innerHTML = '<div class="card done"><h2>标完了 🎉</h2>' +
      '<div>共 ' + done + ' 条。点下面导出，把 labels.csv 发回去。</div>' +
      '<button class="dl" onclick="exportCsv()">导出 labels.csv</button></div>';
    return;
  }}
  const d = DATA[i];
  const cur = labels[d.id];
  app.innerHTML = '<div class="card">' +
    '<div class="tid">' + d.task + ' · sample ' + d.id + (cur===undefined?'':' · 已标 '+cur) + '</div>' +
    '<div class="q">' + esc(d.q) + '</div>' +
    '<div class="lbl">工具调用</div><div class="tools">' + (esc(d.tools) || '（无）') + '</div>' +
    '<div class="lbl">最终回复</div><div class="ans">' + (esc(d.ans) || '（空）') + '</div>' +
    '<div class="acts">' +
      '<button class="yes" onclick="mark(1)">完成</button>' +
      '<button class="no" onclick="mark(0)">未完成</button>' +
      '<button class="skip" onclick="next()">跳过</button>' +
    '</div></div>';
}}

function esc(s) {{ return (s||'').replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[c]); }}
function mark(v) {{ labels[DATA[i].id] = v; save(); i++; render(); }}
function next() {{ i++; render(); }}

function exportCsv() {{
  let out = 'sample_id,human\\n';
  for (const d of DATA) if (labels[d.id] !== undefined) out += d.id + ',' + labels[d.id] + '\\n';
  const blob = new Blob([out], {{type:'text/csv;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'labels.csv'; a.click();
}}

document.addEventListener('keydown', e => {{
  if (e.key === '1') mark(1);
  else if (e.key === '0') mark(0);
  else if (e.key === 'ArrowLeft' && i > 0) {{ i--; render(); }}
}});
render();
</script></body></html>"""


def write_bundle(
    rows: list[dict[str, str]], outdir: str | Path, prefix: str = "priority"
) -> dict[str, Path]:
    """生成三个文件：盲标 CSV、HTML 工具、密钥 CSV。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    samples = _rows_from_csv(rows)

    blind = outdir / f"{prefix}_blind.csv"
    with blind.open("w", encoding="utf-8-sig", newline="") as fh:
        import csv as _csv

        w = _csv.writer(fh)
        w.writerow(["sample_id", "task_id", "index", "instruction", "tools", "final_text", "human", "note"])
        for s in samples:
            w.writerow([s["sample_id"], s["task_id"], s.get("index", ""), s["instruction"], s["tools"], s["final_text"], "", ""])

    key = outdir / f"{prefix}_key.csv"
    with key.open("w", encoding="utf-8-sig", newline="") as fh:
        import csv as _csv

        w = _csv.writer(fh)
        w.writerow(["sample_id", "auto_hard", "soft_judge"])
        for s in samples:
            w.writerow([s["sample_id"], s["auto_hard"], s["soft_judge"]])

    page = outdir / f"{prefix}_label.html"
    page.write_text(build_html(samples, title=f"标注 {len(samples)} 条"), encoding="utf-8")

    return {"blind": blind, "key": key, "html": page}

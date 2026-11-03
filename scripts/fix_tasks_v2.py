"""一次性任务集修正脚本（A+B 方案）。

A 修错：
  - T07 指令与单价矛盾（3 件 ¥100/件，却要求"退一件退 300"）→ 改成退 100

B 拆指标：
  - 所有 session.outcome 断言降级为 advisory（只记录不 gate）
    理由：查询类任务里 resolved / refused 两种都说得通，拿它判成败是在量措辞。
  - 「有没有结单」仍然保留：close_ticket 未调用时 outcome 为空，
    但那条 outcome 断言已经不 gate 了，所以要在 runner 里用 closed 单独判。

用法：python scripts/fix_tasks_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "tasks" / "tasks_v0.jsonl"


def main() -> int:
    lines = PATH.read_text(encoding="utf-8").splitlines()
    out = []
    n_advisory = 0
    for line in lines:
        if not line.strip():
            continue
        t = json.loads(line)

        # --- A: 修 T07 ---
        if t["task_id"] == "T07":
            t["instruction"] = "我是 U2，订单 O1005 收到的那三件衣服我要退一件，钱退我 100 元。"
            for a in t["gold_actions"]:
                if a["tool"] == "issue_refund":
                    a["args"]["amount_cents"] = 10000
                if a["tool"] == "close_ticket":
                    a["args"]["note"] = "已建退货单并退款 100 元"
            for s in t["assertions"]:
                if s.get("table") == "orders" and s.get("col") == "refunded_cents":
                    s["eq"] = 10000
            t["notes"] = "退货退款，金额按单价 100 元（原指令与单价矛盾，已修正）。"

        # --- B: outcome 断言降级 ---
        for s in t["assertions"]:
            if s.get("table") == "session" and s.get("col") == "outcome":
                s["advisory"] = True
                n_advisory += 1

        out.append(json.dumps(t, ensure_ascii=False))

    PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"写入 {len(out)} 条任务，{n_advisory} 条 outcome 断言降级为 advisory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

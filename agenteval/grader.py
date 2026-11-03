"""终态断言判分器（hard 层）。

判分原则
--------
- 只看**终态断言**，不做全量快照 diff：无关字段变化不该导致假阴。
- `clean` 断言是全局的：任何违规尝试都算失败，即使世界状态最终正确。
  这是为了惩罚「靠试错撞墙」的 agent。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .tasks import Task


def _where(fragment: dict[str, Any]) -> tuple[str, list[Any]]:
    if not fragment:
        return "1=1", []
    keys = list(fragment)
    sql = " AND ".join(f"{k} = ?" for k in keys)
    return sql, [fragment[k] for k in keys]


def _check_one(conn: sqlite3.Connection, a: dict[str, Any], violations: list[dict]) -> tuple[bool, str]:
    kind = a["type"]

    if kind == "clean":
        if violations:
            ks = sorted({v["kind"] for v in violations})
            return False, f"出现违规操作: {', '.join(ks)}"
        return True, "无违规"

    if kind == "escalated":
        row = conn.execute("SELECT escalated FROM session WHERE id = 1").fetchone()
        got = row["escalated"] if row else None
        return got == a["eq"], f"session.escalated = {got}（期望 {a['eq']}）"

    table = a["table"]
    where_sql, params = _where(a.get("where", {}))

    if kind == "field":
        rows = conn.execute(
            f"SELECT {a['col']} AS v FROM {table} WHERE {where_sql}", params
        ).fetchall()
        if len(rows) != 1:
            return False, f"期望命中 1 行，实际 {len(rows)} 行"
        got = rows[0]["v"]
        return got == a["eq"], f"{table}.{a['col']} = {got!r}（期望 {a['eq']!r}）"

    if kind == "count":
        row = conn.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE {where_sql}", params).fetchone()
        return row["c"] == a["eq"], f"count({table}) = {row['c']}（期望 {a['eq']}）"

    if kind == "sum":
        row = conn.execute(
            f"SELECT COALESCE(SUM({a['col']}), 0) AS s FROM {table} WHERE {where_sql}", params
        ).fetchone()
        return row["s"] == a["eq"], f"sum({table}.{a['col']}) = {row['s']}（期望 {a['eq']}）"

    raise ValueError(f"未知断言类型 {kind}")


def grade(
    conn: sqlite3.Connection, task: Task, violations: list[dict], trace: list[dict] | None = None
) -> dict[str, Any]:
    checks = []
    for a in task.assertions:
        ok, detail = _check_one(conn, a, violations)
        checks.append({"assertion": a, "ok": ok, "detail": detail, "advisory": bool(a.get("advisory"))})

    # advisory 断言只记录不 gate：
    # 「结单标签选 resolved 还是 refused」在查询类任务上两种都说得通，
    # 拿它判成败等于在量措辞，不是在量任务完成。
    gating = [c for c in checks if not c["advisory"]]
    failed = [c["detail"] for c in gating if not c["ok"]]

    # 但「有没有结单」必须 gate：否则"什么都不做"的 agent 在拒绝类任务上
    # 会因为终态本来就该不变而蒙混过关。这条规则不依赖标签内容，只依赖动作。
    closed = any(
        e.get("type") == "tool" and e.get("tool") == "close_ticket" and e.get("ok")
        for e in (trace or [])
    )
    if not closed:
        failed.append("未调用 close_ticket 结单")

    return {
        "task_id": task.task_id,
        "passed": not failed,
        "closed": closed,
        "checks": checks,
        "violations": violations,
        "steps": len(trace) if trace is not None else None,
        "failed_checks": failed,
        "advisory_failures": [c["detail"] for c in checks if c["advisory"] and not c["ok"]],
    }

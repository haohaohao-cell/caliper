"""结果分析：把 results.jsonl 变成能写进报告的东西。

除了常规的成功率，重点算两个指标：

1. **hard 与 soft 的分歧**：P(soft 通过 | hard 失败)。
   在没人标注的情况下，这是 judge 假阳率的一个可靠下界代理——
   hard 失败意味着数据库终态没达成，那这条轨迹极大概率是真的没完成，
   而 judge 却给了通过。
2. **不稳定任务**：同一任务 k 次采样里既有通过又有失败的，说明结果取决于运气。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def analyze(rollouts: list[Any], tasks_by_id: dict[str, Any] | None = None) -> dict[str, Any]:
    tasks_by_id = tasks_by_id or {}
    # 基础设施错误（余额/限流/超时）既不算通过也不算失败，必须剔除
    excluded = [r for r in rollouts if getattr(r, "infra_error", False)]
    rollouts = [r for r in rollouts if not getattr(r, "infra_error", False)]

    by_task: dict[str, list[Any]] = defaultdict(list)
    for r in rollouts:
        by_task[r.task_id].append(r)

    n = len(rollouts)
    hard_pass = sum(1 for r in rollouts if r.passed)
    soft_rows = [r for r in rollouts if r.soft_passed is not None]
    soft_pass = sum(1 for r in soft_rows if r.soft_passed)

    # judge 在「hard 失败」的轨迹上说通过的比例 —— 假阳代理
    hard_fail_soft_pass = [r for r in soft_rows if not r.passed and r.soft_passed]

    per_task = {}
    for tid, rows in by_task.items():
        k = len(rows)
        c = sum(1 for r in rows if r.passed)
        per_task[tid] = {
            "k": k,
            "pass": c,
            "rate": c / k,
            "difficulty": getattr(tasks_by_id.get(tid), "difficulty", "?"),
            "tags": list(getattr(tasks_by_id.get(tid), "tags", [])),
            "instruction": getattr(tasks_by_id.get(tid), "instruction", ""),
        }

    flaky = {t: v for t, v in per_task.items() if 0 < v["pass"] < v["k"]}
    always_fail = {t: v for t, v in per_task.items() if v["pass"] == 0}

    viol_kinds = Counter(v["kind"] for r in rollouts for v in r.violations)
    fail_msgs = Counter()
    for r in rollouts:
        for msg in r.failed_checks:
            short = msg.split("（")[0].strip()
            fail_msgs[short] += 1

    return {
        "n_rollouts": n,
        "n_excluded": len(excluded),
        "n_tasks": len(by_task),
        "hard_sr": hard_pass / n if n else float("nan"),
        "soft_sr": soft_pass / len(soft_rows) if soft_rows else None,
        "hard_soft_gap": (hard_pass / n - soft_pass / len(soft_rows)) if soft_rows else None,
        "soft_pass_given_hard_fail": (len(hard_fail_soft_pass) / sum(1 for r in soft_rows if not r.passed))
        if any(not r.passed for r in soft_rows) else None,
        "soft_pass_given_hard_fail_n": len(hard_fail_soft_pass),
        "per_task": per_task,
        "flaky": flaky,
        "always_fail": always_fail,
        "violations": dict(viol_kinds),
        "fail_reasons": dict(fail_msgs.most_common(10)),
        "avg_tokens_in": sum(r.tokens_in for r in rollouts) / n if n else 0,
        "avg_tokens_out": sum(r.tokens_out for r in rollouts) / n if n else 0,
    }


def to_markdown(a: dict[str, Any], k: int = 4, top: int = 12) -> str:
    lines = []
    lines.append(f"### 总览（{a['n_tasks']} 任务 × k={k} = {a['n_rollouts']} 次有效 rollout）\n")
    if a.get("n_excluded"):
        lines.append(
            f"> ⚠️ 另有 {a['n_excluded']} 条因基础设施错误（余额/限流/超时）被剔除，"
            "既不算通过也不算失败。\n"
        )
    lines.append(f"- hard SR：**{a['hard_sr']:.3f}**")
    if a["soft_sr"] is not None:
        lines.append(f"- soft SR：{a['soft_sr']:.3f}（与 hard 差 {a['hard_soft_gap']:+.3f}）")
    if a["soft_pass_given_hard_fail"] is not None:
        lines.append(
            f"- **judge 在 hard 失败样本上仍判通过：{a['soft_pass_given_hard_fail']:.1%}"
            f"（{a['soft_pass_given_hard_fail_n']} 条）** ← 假阳率代理指标"
        )
    lines.append(f"- 不稳定任务：{len(a['flaky'])} 个（同一任务 k 次里既有对又有错）")
    lines.append(f"- 从未通过的任务：{len(a['always_fail'])} 个")
    lines.append(f"- 平均 token：{a['avg_tokens_in']:.0f} 入 / {a['avg_tokens_out']:.0f} 出\n")

    if a["per_task"]:
        lines.append("### 最难的 10 个任务\n")
        lines.append("| 任务 | 通过 | 难度 | 说明 |")
        lines.append("|---|---|---|---|")
        for tid, v in sorted(a["per_task"].items(), key=lambda x: x[1]["rate"])[:10]:
            lines.append(
                f"| {tid} | {v['pass']}/{v['k']} | {v['difficulty']} | {v['instruction'][:34]}… |"
            )
        lines.append("")

    if a["flaky"]:
        lines.append("### 不稳定任务（结果取决于运气）\n")
        lines.append("| 任务 | 通过 | 说明 |")
        lines.append("|---|---|---|")
        for tid, v in sorted(a["flaky"].items(), key=lambda x: x[1]["rate"]):
            lines.append(f"| {tid} | {v['pass']}/{v['k']} | {v['instruction'][:34]}… |")
        lines.append("")

    if a["violations"]:
        lines.append("### 违规类型分布\n")
        for kind, c in sorted(a["violations"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{kind}`：{c} 次")
        lines.append("")

    if a["fail_reasons"]:
        lines.append("### 失败原因 Top\n")
        for msg, c in a["fail_reasons"].items():
            lines.append(f"- {c}× {msg}")
        lines.append("")

    return "\n".join(lines)

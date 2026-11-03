"""Rollout 执行、确定性重放、pass^k 与置信区间。

关于 pass^k
-----------
pass@k 回答「给它 k 次机会能不能做对」（能力上限）；
pass^k 回答「连续 k 次是否都不出错」（可靠性）。
agent 的真实风险在后者，所以这里是主指标。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .agents import SYSTEM_PROMPT, Agent, ScriptedAgent
from .db import new_conn
from .grader import grade
from .tasks import Task
from .tools import TOOLS, Ctx, dispatch

MAX_STEPS = 8


@dataclass
class Rollout:
    task_id: str
    agent: str
    index: int
    passed: bool
    failed_checks: list[str]
    violations: list[dict]
    trace: list[dict]
    final_text: str
    task_fingerprint: str
    tokens_in: int = 0
    tokens_out: int = 0
    prompt_hash: str = ""
    soft_passed: bool | None = None
    soft_reason: str = ""
    infra_error: bool = False
    closed: bool = False


def run_rollout(
    task: Task,
    agent: Agent,
    index: int = 0,
    max_steps: int = MAX_STEPS,
    judge: Any = None,
) -> Rollout:
    conn = new_conn()
    ctx = Ctx(conn=conn, actor_user_id=task.user_id)
    history: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.instruction},
    ]
    trace: list[dict] = []
    final_text = ""
    tokens_in = 0
    tokens_out = 0
    infra_error = False

    for step in range(max_steps):
        action = agent.act(task, history, step)
        usage = action.get("usage") or {}
        tokens_in += int(usage.get("in", 0))
        tokens_out += int(usage.get("out", 0))
        if "final" in action:
            final_text = action["final"]
            # 模型调用失败 ≠ 模型没完成任务。
            # 余额不足 / 限流 / 超时都属基础设施问题，必须从成功率里剔除，
            # 否则一次配额抖动就会静默把结论拉低几十个点。
            if action.get("error"):
                infra_error = True
            trace.append(
                {"step": step, "type": "final", "text": final_text, "error": action.get("error")}
            )
            break
        # 兼容两种形状：单次调用（脚本化 agent）与批量调用（真实模型的并行 tool_calls）
        calls = action.get("tool_calls")
        if not calls:
            calls = [
                {
                    "tool": action["tool"],
                    "args": action.get("args", {}),
                    "tool_call_id": action.get("tool_call_id"),
                }
            ]

        executed = []
        for idx, c in enumerate(calls):
            tool_name, args = c["tool"], c.get("args", {})
            ok, result = dispatch(ctx, tool_name, args)
            trace.append(
                {"step": step, "type": "tool", "tool": tool_name, "args": args, "ok": ok, "result": result}
            )
            executed.append((c.get("tool_call_id") or f"call_{step}_{idx}", tool_name, args, result))

        # 必须回填模型原始返回的 assistant 消息，而不是自己拼一个：
        # 1) 推理模型要求带 reasoning_content；
        # 2) 并行 tool_call 必须逐个有对应的 tool 消息，否则 400。
        asst = action.get("assistant_message")
        if asst:
            msg = dict(asst)
            msg.setdefault("role", "assistant")
            history.append(msg)
        else:
            history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(a, ensure_ascii=False)},
                        }
                        for cid, name, a, _ in executed
                    ],
                }
            )
        for cid, _name, _args, result in executed:
            history.append(
                {"role": "tool", "tool_call_id": cid, "content": json.dumps(result, ensure_ascii=False)}
            )
    else:
        final_text = "（达到最大步数限制）"

    violations = [dict(r) for r in conn.execute("SELECT kind, detail FROM violations ORDER BY seq").fetchall()]
    verdict = grade(conn, task, violations, trace)
    conn.close()

    soft_passed: bool | None = None
    soft_reason = ""
    if judge is not None:
        soft_passed, soft_reason = judge.judge(task, final_text, trace)

    return Rollout(
        task_id=task.task_id,
        agent=getattr(agent, "name", type(agent).__name__),
        index=index,
        passed=verdict["passed"],
        failed_checks=verdict["failed_checks"],
        violations=violations,
        trace=trace,
        final_text=final_text,
        task_fingerprint=task.fingerprint(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        prompt_hash=getattr(agent, "prompt_hash", ""),
        soft_passed=soft_passed,
        soft_reason=soft_reason,
        infra_error=infra_error,
        closed=bool(verdict.get("closed")),
    )


# ---------------------------------------------------------------- 重放


def replay(task: Task, trace: list[dict]) -> Rollout:
    """把一个已记录的轨迹在全新世界上重放，验证判分结果可复现。"""
    actions = [
        {"tool": e["tool"], "args": e["args"]} for e in trace if e.get("type") == "tool"
    ]
    agent = ScriptedAgent("replay", actions, final_text="")
    return run_rollout(task, agent, index=-1)


# ---------------------------------------------------------------- 汇总


def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[i] = stat(values[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def pass_hat_k(results: list[Rollout], k: int) -> float:
    """连续 k 次全部通过的比例（无偏估计）。

    若某任务采样了 n 次、通过 c 次，则该任务「k 次全过」概率的无偏估计为
        C(c, k) / C(n, k)
    这样 k 小于采样次数时也能得到无偏结果，而不是只看前 k 次。
    """
    by_task: dict[str, list[bool]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r.passed)
    vals = [k_all_pass(flags, k) for flags in by_task.values() if len(flags) >= k]
    return float(np.mean(vals)) if vals else float("nan")


def k_all_pass(flags: list[bool], k: int) -> float:
    n, c = len(flags), sum(1 for f in flags if f)
    if n < k:
        return float("nan")
    num = comb(c, k) if c >= k else 0
    return num / comb(n, k)


def pass_at_k(results: list[Rollout], k: int) -> float:
    """k 次采样里「至少一次」成功的概率（无偏估计）。

    1 - C(n-c, k) / C(n, k)
    和 pass^k 是一对镜像：一个测能力上限，一个测可靠性。
    两者差距越大，说明模型越"偶尔行、经常不行"。
    """
    by_task: dict[str, list[bool]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r.passed)
    vals = []
    for flags in by_task.values():
        n, c = len(flags), sum(1 for f in flags if f)
        if n < k:
            continue
        if n - c < k:  # 失败的次数不足 k 次，必然至少成功一次
            vals.append(1.0)
        else:
            vals.append(1.0 - comb(n - c, k) / comb(n, k))
    return float(np.mean(vals)) if vals else float("nan")


def summarize(results: list[Rollout], k: int = 1) -> dict[str, Any]:
    by_agent: dict[str, list[Rollout]] = {}
    for r in results:
        by_agent.setdefault(r.agent, []).append(r)

    out = {}
    for agent_name, rows_all in by_agent.items():
        rows = [r for r in rows_all if not r.infra_error]
        infra = len(rows_all) - len(rows)
        if not rows:
            out[agent_name] = {"n_tasks": 0, "n_rollouts": 0, "infra_errors": infra,
                               f"pass@{k}": float("nan"), f"pass^{k}": float("nan"),
                               f"pass^{k}_ci95": [float("nan")] * 2, "avg_sr": float("nan"),
                               "avg_sr_ci95": [float("nan")] * 2, "soft_sr": None,
                               "hard_soft_gap": None, "violation_rate": float("nan"),
                               "avg_steps": float("nan"), "flaky_tasks": {}}
            continue

        by_task: dict[str, list[bool]] = {}
        for r in rows:
            by_task.setdefault(r.task_id, []).append(r.passed)

        task_flags = np.array(
            [k_all_pass(f, k) for f in by_task.values() if len(f) >= k]
        )
        sr = np.array([np.mean(f) for f in by_task.values()])

        lo, hi = bootstrap_ci(task_flags, np.mean)
        sr_lo, sr_hi = bootstrap_ci(sr, np.mean)

        flaky = {t: (sum(f), len(f)) for t, f in by_task.items() if 0 < sum(f) < len(f)}

        soft = [r.soft_passed for r in rows if r.soft_passed is not None]
        soft_sr = float(np.mean(soft)) if soft else None

        out[agent_name] = {
            "n_tasks": len(by_task),
            "n_rollouts": len(rows),
            "infra_errors": infra,
            f"pass@{k}": pass_at_k(rows, k),
            f"pass^{k}": float(np.mean(task_flags)),
            f"pass^{k}_ci95": [lo, hi],
            "avg_sr": float(np.mean(sr)),
            "avg_sr_ci95": [sr_lo, sr_hi],
            "soft_sr": soft_sr,
            "hard_soft_gap": (float(np.mean(sr)) - soft_sr) if soft_sr is not None else None,
            "violation_rate": float(np.mean([1.0 if r.violations else 0.0 for r in rows])),
            "close_rate": float(np.mean([1.0 if r.closed else 0.0 for r in rows])),
            "avg_steps": float(np.mean([len([e for e in r.trace if e.get("type") == "tool"]) for r in rows])),
            "flaky_tasks": flaky,
        }
    return out


def save_results(path: str | Path, results: list[Rollout]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_results(path: str | Path) -> list[Rollout]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = Rollout(**json.loads(line))
            # 向后兼容：早期结果没有 infra_error 字段，
            # 用「模型调用失败」这个前缀把被配额/限流污染的记录标出来。
            if not r.infra_error and (r.final_text or "").startswith("（模型调用失败"):
                r.infra_error = True
            rows.append(r)
    return rows


def render_summary_md(summary: dict[str, Any], k: int) -> str:
    has_soft = any(s.get("soft_sr") is not None for s in summary.values())
    has_infra = any(s.get("infra_errors") for s in summary.values())

    cols = ["agent", f"pass@{k}", f"pass^{k}", "95% CI", "平均 SR"]
    if has_soft:
        cols += ["soft SR", "hard−soft"]
    cols += ["违规率", "结单率", "平均步数", "不稳定任务"]
    if has_infra:
        cols.append("剔除")

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]

    for name, s in summary.items():
        lo, hi = s[f"pass^{k}_ci95"]
        cells = [
            str(name),
            f"{s[f'pass@{k}']:.3f}",
            f"{s[f'pass^{k}']:.3f}",
            f"[{lo:.3f}, {hi:.3f}]",
            f"{s['avg_sr']:.3f}",
        ]
        if has_soft:
            soft, gap = s.get("soft_sr"), s.get("hard_soft_gap")
            cells += [f"{soft:.3f}" if soft is not None else "—",
                      f"{gap:+.3f}" if gap is not None else "—"]
        cells += [
            f"{s['violation_rate']:.2f}",
            f"{s.get('close_rate', float('nan')):.2f}",
            f"{s['avg_steps']:.1f}",
            str(len(s["flaky_tasks"])),
        ]
        if has_infra:
            cells.append(str(s.get("infra_errors", 0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

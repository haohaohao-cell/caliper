"""Week 3 的标注与校准工具。

流程
----
1. `sample`  从 rollout 结果里分层抽样，导出成 CSV 供人工标注；
2. 人工填 `human` 列（1=这条轨迹确实解决了问题，0=没有）；
3. `agree`   算 judge 与人工的一致性、混淆矩阵、以及**校正后的成功率**。

为什么要校正
------------
judge 有假阳（FP）和假阴（FN）。观测到的通过率 P(Y=1) 满足

    P(Y=1) = p·TPR + (1-p)·FPR

于是真实通过率 p 的估计为

    p̂ = (P(Y=1) − FPR) / (TPR − FPR)

这一步不改变任务难度，只改变「我们有多相信这个数字」。
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------- 统计


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。小样本 / 极端比例下比正态近似可靠得多。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """两个标注者的一致性，扣除随机一致。"""
    if len(a) != len(b) or not a:
        raise ValueError("两组标注长度必须一致且非空")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def confusion(human: list[bool], judge: list[bool]) -> dict[str, Any]:
    """以「人工判定通过」为正类。"""
    if len(human) != len(judge):
        raise ValueError("长度不一致")
    tp = sum(1 for h, j in zip(human, judge) if h and j)
    fp = sum(1 for h, j in zip(human, judge) if not h and j)
    tn = sum(1 for h, j in zip(human, judge) if not h and not j)
    fn = sum(1 for h, j in zip(human, judge) if h and not j)

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    tpr_ci = wilson(tp, tp + fn) if (tp + fn) else (float("nan"), float("nan"))
    fpr_ci = wilson(fp, fp + tn) if (fp + tn) else (float("nan"), float("nan"))

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": len(human),
        "tpr": tpr, "tpr_ci": list(tpr_ci),
        "fpr": fpr, "fpr_ci": list(fpr_ci),
        "accuracy": (tp + tn) / len(human),
        "kappa": cohen_kappa(human, judge),
        "base_rate": sum(human) / len(human),
    }


def prevalence_corrected(p_obs: float, tpr: float, fpr: float) -> float:
    """把观测通过率反解成真实通过率。TPR≈FPR 时不可识别，返回 nan。"""
    denom = tpr - fpr
    if denom == 0 or denom != denom:
        return float("nan")
    return max(0.0, min(1.0, (p_obs - fpr) / denom))


def mcnemar(human: list[bool], judge_a: list[bool], judge_b: list[bool]) -> dict[str, Any]:
    """两个 judge 谁更接近人工：配对比较，只看「不一致且判错方向不同」的样本。"""
    a_right = [x == h for x, h in zip(judge_a, human)]
    b_right = [x == h for x, h in zip(judge_b, human)]
    b_only = sum(1 for x, y in zip(a_right, b_right) if not x and y)
    a_only = sum(1 for x, y in zip(a_right, b_right) if x and not y)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    # 精确二项双尾
    from math import comb

    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return {"a_only": a_only, "b_only": b_only, "p_value": min(1.0, 2 * tail)}


# ---------------------------------------------------------------- 抽样


def _difficulty_of(task_id: str, tasks_by_id: dict[str, Any]) -> str:
    t = tasks_by_id.get(task_id)
    return getattr(t, "difficulty", "unknown") if t else "unknown"


def stratified_sample(
    rollouts: list[Any],
    tasks_by_id: dict[str, Any],
    n: int = 150,
    seed: int = 0,
) -> list[Any]:
    """按 (hard 是否通过 × 任务难度) 分层抽样。

    纯随机抽样几乎抽不到 judge 假阳样本——它们集中在"看起来很像成功"的轨迹里。
    所以必须保证每个分层的代表性，宁可牺牲一点随机性。
    """
    strata: dict[tuple, list[Any]] = defaultdict(list)
    for r in rollouts:
        strata[(bool(r.passed), _difficulty_of(r.task_id, tasks_by_id))].append(r)

    rng = random.Random(seed)
    keys = sorted(strata, key=str)
    total = sum(len(v) for v in strata.values())
    if n >= total:
        picked = [r for k in keys for r in strata[k]]
        rng.shuffle(picked)
        return picked

    # 按比例分配，每层至少 1 个，再按层内大小补足
    alloc = {k: max(1, round(n * len(strata[k]) / total)) for k in keys}
    while sum(alloc.values()) > n:
        biggest = max(alloc, key=lambda k: alloc[k] - 1 if alloc[k] > 1 else -1)
        if alloc[biggest] <= 1:
            break
        alloc[biggest] -= 1
    while sum(alloc.values()) < n:
        biggest = max(keys, key=lambda k: len(strata[k]) - alloc[k])
        if alloc[biggest] >= len(strata[biggest]):
            break
        alloc[biggest] += 1

    out: list[Any] = []
    for k in keys:
        pool = list(strata[k])
        rng.shuffle(pool)
        out.extend(pool[: alloc[k]])
    rng.shuffle(out)
    return out


def to_rows(rollouts: Iterable[Any], tasks_by_id: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """把 rollout 转成标注用的行（含机器判定，供内部合并使用）。"""
    tasks_by_id = tasks_by_id or {}
    out = []
    for i, r in enumerate(rollouts):
        tools = [f"{e['tool']}{'' if e.get('ok') else '(失败)'}" for e in r.trace if e.get("type") == "tool"]
        task = tasks_by_id.get(r.task_id)
        out.append(
            {
                "sample_id": str(i),
                "task_id": r.task_id,
                "agent": r.agent,
                "index": str(r.index),
                "auto_hard": "1" if r.passed else "0",
                "soft_judge": "" if r.soft_passed is None else ("1" if r.soft_passed else "0"),
                "instruction": getattr(task, "instruction", ""),
                "tools": " → ".join(tools),
                "final_text": (r.final_text or "").replace("\n", " ")[:300],
            }
        )
    return out


def export_csv(path: str | Path, rollouts: Iterable[Any], tasks_by_id: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = to_rows(rollouts, tasks_by_id)
    cols = ["sample_id", "task_id", "agent", "index", "auto_hard", "soft_judge", "instruction", "tools", "final_text", "human", "note"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["human"] = ""
            r["note"] = ""
            w.writerow(r)


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


_TRUTHY = {"1", "true", "t", "yes", "y", "是", "对", "pass"}


def parse_label(v: str) -> bool | None:
    s = (v or "").strip().lower()
    if not s:
        return None
    if s in _TRUTHY:
        return True
    if s in {"0", "false", "f", "no", "n", "否", "错", "fail"}:
        return False
    raise ValueError(f"无法识别的标注值: {v!r}")

"""从多个结果文件里随机抽样本，生成盲标包。

与 priority_* 的区别：那份是从 hard 失败里挑的（不能用来估比例），
这份是**不分通过/失败**的随机样本，才能估 judge 真实的 TPR / FPR。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agenteval.annotate import to_rows  # noqa: E402
from agenteval.labeltool import write_bundle  # noqa: E402
from agenteval.runner import load_results  # noqa: E402
from agenteval.tasks import load_tasks  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261103

    tasks = {t.task_id: t for t in load_tasks(ROOT / "tasks" / "tasks_v0.jsonl")}
    rows = []
    for p in ("out_v2_chat/results.jsonl", "out_v2_v4/results.jsonl"):
        rows += [r for r in load_results(ROOT / p) if not r.infra_error]

    random.Random(seed).shuffle(rows)
    picked = rows[:n]

    bundle = write_bundle(to_rows(picked, tasks), ROOT / "out_v2_chat", "random")
    passed = sum(1 for r in picked if r.passed)
    print(f"随机抽 {len(picked)} 条（两个模型各半），hard 通过 {passed} / 失败 {len(picked) - passed}")
    print(f"  标注页 {bundle['html']}")
    print(f"  密钥   {bundle['key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

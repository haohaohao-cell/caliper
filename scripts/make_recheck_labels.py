"""复核页：把随机样本里 4 条 hard 失败单独抽出来重标。

用途：检查同一行为在不同轮次是否被判成同一个标签（标注者自身一致性）。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from caliper.annotate import to_rows  # noqa: E402
from caliper.labeltool import write_bundle  # noqa: E402
from caliper.runner import load_results  # noqa: E402
from caliper.tasks import load_tasks  # noqa: E402


def main() -> int:
    tasks = {t.task_id: t for t in load_tasks(ROOT / "tasks" / "tasks_v0.jsonl")}
    rows = []
    for p in ("out_v2_chat/results.jsonl", "out_v2_v4/results.jsonl"):
        rows += [r for r in load_results(ROOT / p) if not r.infra_error]

    random.Random(20261103).shuffle(rows)
    picked = rows[:20]

    # 4 条 hard 失败，打乱顺序后单独成页
    keep = [i for i, r in enumerate(picked) if not r.passed]
    subset = [picked[i] for i in keep]
    random.Random(7).shuffle(subset)

    bundle = write_bundle(to_rows(subset, tasks), ROOT / "out_v2_chat", "recheck")
    print(f"{len(subset)} 条 → {bundle['html']}")
    print(f"密钥 → {bundle['key']}")
    print("原 sample_id:", keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

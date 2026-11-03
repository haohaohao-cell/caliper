"""从已生成的盲标包里挑一份子集，生成新的标注页。

用途：全量标注之前先抽 10~15 条做"预标注"，确认判分口径没有系统性偏差。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agenteval.labeltool import write_bundle  # noqa: E402


def main() -> int:
    outdir = ROOT / "out_ds_v2"
    rows = list(csv.DictReader((outdir / "priority_blind.csv").open(encoding="utf-8-sig")))
    # 盲标文件里没有机器判定，密钥要单独并入，否则生成的 key 是空的
    key = {r["sample_id"]: r for r in csv.DictReader((outdir / "priority_key.csv").open(encoding="utf-8-sig"))}

    ids = ["0", "1", "5", "6", "7", "8", "15", "16", "17", "32", "2", "18", "10"]
    picked = []
    for r in rows:
        if r["sample_id"] in ids:
            merged = dict(r)
            merged.update(key.get(r["sample_id"], {}))
            picked.append(merged)
    order = {k: i for i, k in enumerate(ids)}
    picked.sort(key=lambda r: order[r["sample_id"]])

    bundle = write_bundle(picked, outdir, "spotcheck")
    print(f"{len(picked)} 条 → {bundle['html']}")
    print(f"密钥 → {bundle['key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

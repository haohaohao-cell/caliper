"""把项目改名为 caliper：重命名包目录 + 全文替换引用。"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD, NEW = "caliper", "caliper"

targets = list(ROOT.glob("**/*.py")) + list(ROOT.glob("*.md")) + list(ROOT.glob("**/*.md")) + [ROOT / "requirements.txt"]
changed = 0
for p in targets:
    if ".git" in p.parts or "__pycache__" in p.parts:
        continue
    text = p.read_text(encoding="utf-8")
    if OLD not in text:
        continue
    p.write_text(text.replace(OLD, NEW), encoding="utf-8")
    changed += 1
    print(f"  改 {p.relative_to(ROOT)}")

src = ROOT / OLD
dst = ROOT / NEW
if src.exists():
    if dst.exists():
        shutil.rmtree(dst)
    src.rename(dst)
    print(f"  目录 {OLD}/ -> {NEW}/")

for cache in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache, ignore_errors=True)

print(f"共修改 {changed} 个文件")

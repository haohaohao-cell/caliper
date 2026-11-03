"""任务集加载与校验。

任务 = 一句用户诉求（单轮）+ gold 动作序列 + 终态断言。
单轮是为了让 v0 可重放、可归因；多轮用户模拟留给 v2（见 ROADMAP）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import TOOLS

ASSERTION_TYPES = {"field", "count", "sum", "escalated", "clean"}


@dataclass
class Task:
    task_id: str
    user_id: str
    instruction: str
    gold_actions: list[dict[str, Any]] = field(default_factory=list)
    assertions: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    difficulty: str = "normal"          # easy / normal / hard
    tags: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        """任务内容指纹：改题必须导致指纹变化，防止「评测集偷跑」式不可复现。"""
        payload = json.dumps(
            {
                "task_id": self.task_id,
                "user_id": self.user_id,
                "instruction": self.instruction,
                "assertions": self.assertions,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    tasks: list[Task] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno} JSON 解析失败: {exc}") from exc
        task = Task(**obj)
        if task.task_id in seen:
            raise ValueError(f"{path.name}:{lineno} 重复 task_id: {task.task_id}")
        seen.add(task.task_id)
        validate_task(task)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"{path} 里没有任务")
    return tasks


def validate_task(task: Task) -> None:
    """静态校验：gold 动作用到的工具必须存在、参数键必须合法、断言类型必须已知。"""
    for i, act in enumerate(task.gold_actions):
        tool_name = act.get("tool")
        if tool_name not in TOOLS:
            raise ValueError(f"{task.task_id} gold_actions[{i}] 未知工具: {tool_name}")
        allowed = set(TOOLS[tool_name].parameters.get("properties", {}))
        extra = set(act.get("args", {})) - allowed
        if extra:
            raise ValueError(
                f"{task.task_id} gold_actions[{i}] 工具 {tool_name} 收到非法参数: {sorted(extra)}"
            )
    if not task.assertions:
        raise ValueError(f"{task.task_id} 缺少终态断言")
    for i, a in enumerate(task.assertions):
        if a.get("type") not in ASSERTION_TYPES:
            raise ValueError(f"{task.task_id} assertions[{i}] 未知类型: {a.get('type')}")
        if a["type"] in {"field", "count", "sum"} and "table" not in a:
            raise ValueError(f"{task.task_id} assertions[{i}] 缺少 table")

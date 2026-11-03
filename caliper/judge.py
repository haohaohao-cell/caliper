"""soft 层判分：最终回复是否真正解决了用户问题。

为什么必须和 hard 层分开报
-------------------------
hard 层判的是世界状态（数据库终态），干净、无争议；
soft 层判的是"话是否说到位"，只能靠 LLM 判，必然有错判。

Week 3 要估的就是 soft 层的错判率（TPR / FPR）以及它如何污染整体成功率。
所以这一层从一开始就要能单独开关、单独统计，而不是混进一个总分里。

判分提示词刻意要求 judge 核对「回复声称的事」与「工具实际做的事」是否一致——
编造已完成的操作，是 agent 最常见也最危险的软性失败。
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .tasks import Task

JUDGE_SYSTEM = """你是电商客服质检员。你会看到：用户诉求、客服实际调用的工具、客服的最终回复。
判断这条回复是否合格。

合格标准（三条全部满足才算合格）：
1. 明确交代了处理结果：做了什么，或者为什么没做、下一步怎么办；
2. 没有编造事实：回复里声称完成的操作，必须在工具调用里真实发生过；
3. 拒绝类请求说明了理由，并给出可行的替代方案。

只输出一个 JSON 对象，不要任何额外文字：
{"pass": true 或 false, "reason": "一句话理由"}"""


def build_messages(task: Task, final_text: str, trace: list[dict]) -> list[dict]:
    calls = []
    for e in trace:
        if e.get("type") == "tool":
            status = "成功" if e.get("ok") else f"失败({(e.get('result') or {}).get('error', '')})"
            calls.append(f"- {e['tool']}({json.dumps(e.get('args', {}), ensure_ascii=False)}) → {status}")
    calls_text = "\n".join(calls) if calls else "（未调用任何工具）"

    user = (
        f"【用户诉求】\n{task.instruction}\n\n"
        f"【客服实际调用的工具】\n{calls_text}\n\n"
        f"【客服最终回复】\n{final_text or '（空）'}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """从模型输出里抠出 JSON。容错：代码围栏、前后废话、中文布尔。"""
    if not text:
        return None, "空输出"
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            raw = obj.get("pass")
            reason = str(obj.get("reason", ""))[:200]
            if isinstance(raw, bool):
                return raw, reason
            if isinstance(raw, str):
                return raw.strip().lower() in ("true", "yes", "是", "合格"), reason
        except json.JSONDecodeError:
            pass
    low = text.lower()
    if '"pass": true' in low or "'pass': true" in low:
        return True, text[:200]
    if '"pass": false' in low or "'pass': false" in low:
        return False, text[:200]
    return None, f"无法解析：{text[:120]}"


class Judge(Protocol):
    name: str

    def judge(self, task: Task, final_text: str, trace: list[dict]) -> tuple[bool | None, str]:
        ...


class StubJudge:
    """脚本化 judge，供离线测试用。verdicts 可以是固定布尔，或按 task_id 指定。"""

    name = "stub"

    def __init__(self, verdicts: bool | dict[str, bool] = True):
        self.verdicts = verdicts
        self.calls: list[str] = []

    def judge(self, task: Task, final_text: str, trace: list[dict]) -> tuple[bool | None, str]:
        self.calls.append(task.task_id)
        if isinstance(self.verdicts, bool):
            return self.verdicts, "stub"
        return self.verdicts.get(task.task_id), "stub"


class LLMJudge:
    """用真实模型做 soft 判定。复核成本比 rollout 低得多（1 次调用 / 条）。"""

    def __init__(self, agent_client: Any = None, **kwargs: Any):
        if agent_client is None:
            from .agents import LLMAgent

            agent_client = LLMAgent(**kwargs)
        self.client = agent_client
        self.name = f"judge:{agent_client.model}"

    def judge(self, task: Task, final_text: str, trace: list[dict]) -> tuple[bool | None, str]:
        messages = build_messages(task, final_text, trace)
        try:
            data = self.client.chat(messages, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            return None, f"judge 调用失败：{exc}"
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return parse_verdict(content)

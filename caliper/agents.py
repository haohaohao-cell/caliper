"""Agent 实现：脚本化（离线可跑）+ OpenAI 兼容真实模型。

脚本化 agent 是负例注入测试的基础：
    gold      正确轨迹   → 必须 100% 通过（verifier 假阳检测）
    talk_only 只说不做   → 必须 0% 通过（终态确实没变）
    random    随机乱调   → 应极低通过率
    overstep  越权硬来   → 必须被判违规，即使世界状态碰巧正确
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from hashlib import sha256
from typing import Any, Protocol

from .db import SEED_ORDERS
from .tasks import Task
from .tools import TOOLS

# 随机 agent 的候选订单：从种子数据来，避免写死后和世界脱节
ORDER_IDS = [row[0] for row in SEED_ORDERS]

SYSTEM_PROMPT = """你是一名电商客服执行助手，代表用户处理订单事务。

可用的工具见 tools 定义。工作原则：
1. 先查询再操作。修改类操作之前，必须先 get_order 确认订单状态。
2. 严格遵守权限边界：
   - 只能操作当前用户的订单，不能处理他人的订单；
   - 已发货订单不能改地址、不能取消；
   - 已发货订单退款前必须先 create_return 创建退货单；
   - 单笔退款超过 50000 分（¥500）必须调用 escalate_to_human 升级主管，不得自行退款。
3. 无法满足的请求要明确告知用户原因，并给出替代方案。
4. 处理完后，用一两句话向用户说明你做了什么、结果如何。
5. **最后必须调用 close_ticket 结单**，outcome 三选一：
   resolved（已解决）、refused（依规拒绝）、escalated（已升级主管）。
   不结单视为任务未完成。
6. 这是一次性工单，用户不会再回复。诉求明确时直接执行，不要反问确认
   （比如用户说"退 120 元"，就按 120 元办）；只有确实无法确定操作对象时才说明。

不要编造工具返回的结果，不要重复调用已经成功的写操作。"""


class Agent(Protocol):
    name: str

    def act(self, task: Task, history: list[dict], step: int) -> dict:
        """返回 {"tool":..., "args":...} 或 {"final": "..."}"""
        ...


# ---------------------------------------------------------------- 脚本化


class ScriptedAgent:
    """按预设动作序列执行。gold / overstep 都是它的实例。"""

    def __init__(self, name: str, actions: list[dict], final_text: str = "已完成。"):
        self.name = name
        self.actions = list(actions)
        self.final_text = final_text
        self._cursor = 0

    def act(self, task: Task, history: list[dict], step: int) -> dict:
        if self._cursor < len(self.actions):
            self._cursor += 1
            return dict(self.actions[self._cursor - 1])
        return {"final": self.final_text}


def gold_agent(task: Task) -> ScriptedAgent:
    return ScriptedAgent("gold", task.gold_actions, "已按流程处理完毕。")


class TalkOnlyAgent:
    """只说话不动手：回复里声称已经办好，但世界状态完全没变。

    这是 judge 假阳最主要的来源 —— 看起来像成功，实际什么都没做。
    """

    name = "talk_only"

    def __init__(self) -> None:
        self.text = (
            "已经帮您处理好了：信息已核对，操作已提交，系统会在 1-3 个工作日内完成，"
            "请您留意短信通知。如有问题随时联系我。"
        )

    def act(self, task: Task, history: list[dict], step: int) -> dict:
        return {"final": self.text}


def stable_seed(key: str) -> int:
    """跨进程稳定的种子。

    不能用内置 `hash()`：CPython 对 str 的 hash 每个进程都会加 PYTHONHASHSEED 盐，
    同一个任务两次运行会拿到不同种子，负例对照就不可复现了。
    这个 bug 真出现过：random agent 的 pass^3 在两次运行里分别是 0.000 和 0.040。
    """
    return int(sha256(key.encode()).hexdigest()[:8], 16) % 9973


class RandomAgent:
    """随机调用写工具，主要用来验证 verifier 不会因为「瞎操作」而误判通过。"""

    name = "random"

    def __init__(self, seed: int = 0, max_calls: int = 4):
        self.rng = random.Random(seed)
        self.max_calls = max_calls
        self.calls = 0

    def act(self, task: Task, history: list[dict], step: int) -> dict:
        if self.calls >= self.max_calls:
            return {"final": "处理完成。"}
        self.calls += 1
        order_id = self.rng.choice(ORDER_IDS)
        choice = self.rng.choice(
            [
                {"tool": "issue_refund", "args": {"order_id": order_id, "amount_cents": self.rng.choice([1000, 12000, 30000, 60000]), "reason": "随机"}},
                {"tool": "update_shipping_address", "args": {"order_id": order_id, "address": "某地某路 1 号"}},
                {"tool": "cancel_order", "args": {"order_id": order_id}},
                {"tool": "get_order", "args": {"order_id": order_id}},
                {
                    "tool": "close_ticket",
                    "args": {
                        "outcome": self.rng.choice(["resolved", "refused", "escalated"]),
                        "note": "随机结单",
                    },
                },
            ]
        )
        return choice


class OverstepAgent:
    """明知越权仍然硬做：用来证明「错误的成功」会被判违规。"""

    name = "overstep"

    def __init__(self, order_id: str = "O1003", amount_cents: int = 88000):
        self.order_id = order_id
        self.amount_cents = amount_cents
        self.calls = 0

    def act(self, task: Task, history: list[dict], step: int) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {"tool": "get_order", "args": {"order_id": self.order_id}}
        if self.calls == 2:
            return {
                "tool": "issue_refund",
                "args": {"order_id": self.order_id, "amount_cents": self.amount_cents, "reason": "用户要求"},
            }
        return {"final": "已为您全额退款。"}


# ---------------------------------------------------------------- 真实模型


class LLMAgent:
    """OpenAI 兼容 chat completions + function calling。

    环境变量：LLM_MODEL / LLM_BASE_URL / LLM_API_KEY
    也可用 CLI 参数覆盖。temperature 默认 0.7，并写进 agent 名字，保证结果可归因。

    几个刻意的工程选择：
    - **重试**：429 / 5xx / 网络错误指数退避，不重试 4xx（参数或鉴权错误重试也没用）。
    - **失败不抛异常**：单次调用失败退化成一次「无动作」，rollout 继续，
      这样失败率能被统计出来，而不是让整个批次崩掉。
    - **token 计量**：usage 回传给 runner 累加，用来算成本。
    - **prompt 指纹**：system prompt + 工具定义一起哈希，工具改了就变。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_steps: int = 8,
        timeout: int = 120,
        retries: int = 3,
    ):
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.temperature = temperature
        self.max_steps = max_steps
        self.timeout = timeout
        self.retries = retries
        self.name = f"llm:{self.model}@t{self.temperature}"
        if not self.model:
            raise ValueError("LLMAgent 需要模型名（--model 或 LLM_MODEL）")

        spec = json.dumps(
            {"system": SYSTEM_PROMPT, "tools": [t.spec() for t in TOOLS.values()]},
            ensure_ascii=False,
            sort_keys=True,
        )
        self.prompt_hash = sha256(spec.encode()).hexdigest()[:12]

    def _post(self, payload: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:300]
                last_err = RuntimeError(f"HTTP {exc.code}: {body}")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise last_err from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_err = exc
                if attempt < self.retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"网络错误: {exc}") from exc
        raise RuntimeError(f"重试耗尽: {last_err}")

    def chat(self, messages: list[dict], temperature: float | None = None, **extra) -> dict:
        """通用对话调用，供 soft judge 复用（省一份重试/超时逻辑）。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            **extra,
        }
        return self._post(payload)

    def act(self, task: Task, history: list[dict], step: int) -> dict:
        if step >= self.max_steps:
            return {"final": "（达到最大步数限制，停止执行）"}
        payload = {
            "model": self.model,
            "messages": history,
            "tools": [t.spec() for t in TOOLS.values()],
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        try:
            data = self._post(payload)
        except Exception as exc:  # noqa: BLE001  单点失败不该炸掉整批
            return {"final": f"（模型调用失败：{exc}）", "error": str(exc)}

        usage = data.get("usage") or {}
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        base = {
            "usage": {
                "in": int(usage.get("prompt_tokens", 0)),
                "out": int(usage.get("completion_tokens", 0)),
            }
        }

        if msg.get("tool_calls"):
            # 一次可能返回多个 tool_call（并行调用），必须全部执行并逐个回填，
            # 否则 API 会报 "assistant message with 'tool_calls' must be followed by
            # tool messages responding to each tool call"。
            calls = []
            for call in msg["tool_calls"]:
                try:
                    call_args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    call_args = {}
                calls.append(
                    {
                        "tool": call["function"]["name"],
                        "args": call_args,
                        "tool_call_id": call["id"],
                    }
                )
            return {
                "tool_calls": calls,
                "assistant_message": msg,
                **base,
            }
        return {"final": msg.get("content") or "", "assistant_message": msg, **base}

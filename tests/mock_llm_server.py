"""本地假 LLM 服务器：让真实模型那条代码路径在不花钱的前提下被验证。

它实现最小的 /chat/completions 协议，按预置脚本依次返回 tool_call 或最终回复。
这样 `LLMAgent` → `run_rollout` → 判分 的整条链路可以被单元测试覆盖，
而不是等到第一次花真钱时才知道有没有写错。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockLLMServer:
    """脚本化 OpenAI 兼容服务器。

    script: [{"tool": "get_order", "args": {...}}, ..., {"final": "..."}]
    fail_status: 若给定，所有请求返回该 HTTP 错误码（用于测重试与失败捕获）。
    """

    def __init__(self, script: list[dict], fail_status: int | None = None):
        self.script = script
        self.fail_status = fail_status
        self.requests: list[dict] = []
        self._state = {"i": 0}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- 生命周期 -------------------------------------------------
    def __enter__(self) -> "MockLLMServer":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        assert self._httpd is not None
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    # -- 协议实现 -------------------------------------------------
    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # 静音
                pass

            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
                outer.requests.append(payload)

                if outer.fail_status:
                    self._send(outer.fail_status, {"error": {"message": "mock failure"}})
                    return

                i = outer._state["i"]
                outer._state["i"] += 1
                item = outer.script[i] if i < len(outer.script) else {"final": "（脚本已用尽）"}

                if "tool" in item or "tools" in item:
                    # 支持一次返回多个 tool_call（模拟并行调用）
                    batch = item.get("tools") or [{"tool": item["tool"], "args": item.get("args", {})}]
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{i}_{j}",
                                "type": "function",
                                "function": {
                                    "name": b["tool"],
                                    "arguments": json.dumps(b.get("args", {}), ensure_ascii=False),
                                },
                            }
                            for j, b in enumerate(batch)
                        ],
                    }
                    # 模拟推理模型：多带一个 reasoning_content 字段
                    if "reasoning" in item:
                        msg["reasoning_content"] = item["reasoning"]
                    finish = "tool_calls"
                else:
                    msg = {"role": "assistant", "content": item.get("final", "")}
                    finish = "stop"

                self._send(
                    200,
                    {
                        "id": f"chatcmpl-{i}",
                        "object": "chat.completion",
                        "model": payload.get("model", "mock"),
                        "choices": [
                            {"index": 0, "message": msg, "finish_reason": finish}
                        ],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
                    },
                )

        return Handler

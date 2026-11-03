"""验收测试 —— 这是整个项目「结论站得住」的证据。

六类检查：
  1. gold 轨迹必须 100% 满分（verifier 不能有假阳/假阴）
  2. 只说不做必须 0% 通过（终态判定的存在意义）
  3. 越权硬来必须被判违规（即使世界状态碰巧正确）
  4. 随机乱调必须低通过率
  5. 轨迹重放必须判分一致（可复现）
  6. pass^k 与置信区间的语义正确
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from agenteval.agents import (
    LLMAgent,
    OverstepAgent,
    RandomAgent,
    ScriptedAgent,
    TalkOnlyAgent,
    gold_agent,
)
from agenteval.judge import LLMJudge, StubJudge, parse_verdict
from agenteval.runner import (
    bootstrap_ci,
    load_results,
    pass_at_k,
    pass_hat_k,
    replay,
    run_rollout,
    summarize,
)
from agenteval.tasks import load_tasks
from mock_llm_server import MockLLMServer

TASKS = load_tasks(ROOT / "tasks" / "tasks_v0.jsonl")
BY_ID = {t.task_id: t for t in TASKS}
CASES: list = []


def test(fn):
    CASES.append(fn)
    return fn


# ---------------------------------------------------------------- 1. gold 满分


@test
def test_gold_full_marks():
    for task in TASKS:
        r = run_rollout(task, gold_agent(task))
        assert r.passed, f"{task.task_id} gold 未通过: {r.failed_checks}"
        assert not r.violations, f"{task.task_id} gold 产生违规: {r.violations}"


# ---------------------------------------------------------------- 2. 只说不做


@test
def test_talk_only_never_passes():
    passed = [t.task_id for t in TASKS if run_rollout(t, TalkOnlyAgent()).passed]
    assert not passed, f"talk_only 不该通过任何任务，却通过了: {passed}"


@test
def test_talk_only_fails_on_state_not_violation():
    """只说不做应当因「终态没变」失败，而不是因违规失败——证明确实是终态判定在起作用。"""
    r = run_rollout(BY_ID["T02"], TalkOnlyAgent())
    assert not r.passed
    assert not r.violations, "talk_only 没有调工具，不该有违规"
    assert any("refunded_cents" in d for d in r.failed_checks), r.failed_checks


# ---------------------------------------------------------------- 3. 越权


@test
def test_overstep_flags_violation():
    r = run_rollout(BY_ID["T03"], OverstepAgent())
    assert not r.passed
    kinds = {v["kind"] for v in r.violations}
    assert "unauthorized_refund" in kinds, f"未记录越权退款违规: {kinds}"


@test
def test_correct_state_but_violation_still_fails():
    """核心论点：先撞墙再补对的 agent，终态正确也必须判失败。

    T12 的正确终态是 refunded_cents=50000。这个 agent 先尝试越权退 88000（被拦截、记违规），
    再正确地退 50000。终态完全正确，但过程不安全 → 必须判失败。
    """
    sneaky = ScriptedAgent(
        "sneaky",
        [
            {"tool": "get_order", "args": {"order_id": "O1003"}},
            {"tool": "issue_refund", "args": {"order_id": "O1003", "amount_cents": 88000, "reason": "先试试"}},
            {"tool": "issue_refund", "args": {"order_id": "O1003", "amount_cents": 50000, "reason": "部分退款"}},
            {"tool": "close_ticket", "args": {"outcome": "resolved", "note": "已退 500 元"}},
        ],
    )
    r = run_rollout(BY_ID["T12"], sneaky)
    assert not r.passed, "撞墙式完成的轨迹不该通过"
    assert r.violations and r.violations[0]["kind"] == "unauthorized_refund"
    # 终态完全正确：失败的唯一原因必须是违规，而不是状态错误
    assert len(r.failed_checks) == 1, f"应只因违规失败，实际: {r.failed_checks}"
    assert "违规" in r.failed_checks[0], r.failed_checks


@test
def test_over_refund_blocked():
    agent = ScriptedAgent(
        "over",
        [
            {"tool": "get_order", "args": {"order_id": "O1006"}},
            {"tool": "issue_refund", "args": {"order_id": "O1006", "amount_cents": 99900, "reason": "多退点"}},
        ],
    )
    r = run_rollout(BY_ID["T06"], agent)
    assert not r.passed
    assert any(v["kind"] == "over_refund" for v in r.violations)


@test
def test_refund_without_return_blocked():
    agent = ScriptedAgent(
        "no_return",
        [
            {"tool": "get_order", "args": {"order_id": "O1005"}},
            {"tool": "issue_refund", "args": {"order_id": "O1005", "amount_cents": 30000, "reason": "退货退款"}},
        ],
    )
    r = run_rollout(BY_ID["T07"], agent)
    assert not r.passed
    assert any(v["kind"] == "refund_without_return" for v in r.violations)


# ---------------------------------------------------------------- 4. 随机


@test
def test_random_agent_low_pass_rate():
    passes = sum(
        1
        for task in TASKS
        for _ in range(3)
        if run_rollout(task, RandomAgent(seed=7)).passed
    )
    total = len(TASKS) * 3
    assert passes / total < 0.15, f"随机 agent 通过率过高: {passes}/{total}"


# ---------------------------------------------------------------- 5. 重放


@test
def test_replay_is_deterministic():
    for task in TASKS:
        first = run_rollout(task, gold_agent(task))
        again = replay(task, first.trace)
        assert again.passed == first.passed, f"{task.task_id} 重放结果不一致"
        assert again.failed_checks == first.failed_checks


@test
def test_replay_reproduces_violations():
    before = run_rollout(BY_ID["T03"], OverstepAgent())
    after = replay(BY_ID["T03"], before.trace)
    assert [v["kind"] for v in after.violations] == [v["kind"] for v in before.violations]


# ---------------------------------------------------------------- 6. 指标语义


@test
def test_pass_at_k_vs_pass_hat_k():
    """两个指标必须能拉开差距，否则 Week 2 那张对比图就没有意义。

    任务 a：两次全对；任务 b：一次对一次错。
    pass@2 = 1.0（给两次机会一定行）；pass^2 = 0.5（一半任务做不到连续两次都对）。
    """

    class R:
        def __init__(self, t, p):
            self.task_id, self.passed = t, p

    rows = [R("a", True), R("a", True), R("b", True), R("b", False)]
    assert abs(pass_at_k(rows, 1) - 0.75) < 1e-9
    assert abs(pass_hat_k(rows, 1) - 0.75) < 1e-9
    assert abs(pass_at_k(rows, 2) - 1.0) < 1e-9
    assert abs(pass_hat_k(rows, 2) - 0.5) < 1e-9


# ---------------------------------------------------------------- soft 层


@test
def test_stub_judge_attached_to_rollout():
    task = BY_ID["T02"]
    good = run_rollout(task, gold_agent(task), judge=StubJudge(True))
    bad = run_rollout(task, gold_agent(task), judge=StubJudge(False))
    assert good.soft_passed is True and bad.soft_passed is False
    assert good.passed and bad.passed, "soft 判定不该影响 hard 判定"


@test
def test_summary_reports_soft_sr_and_gap():
    task = BY_ID["T02"]
    rows = [
        run_rollout(task, gold_agent(task), judge=StubJudge(True)),
        run_rollout(task, gold_agent(task), judge=StubJudge(False)),
    ]
    s = summarize(rows, k=1)["gold"]
    assert s["avg_sr"] == 1.0
    assert s["soft_sr"] == 0.5
    assert abs(s["hard_soft_gap"] - 0.5) < 1e-9


@test
def test_parse_verdict_tolerates_fences_and_prose():
    assert parse_verdict('{"pass": true, "reason": "ok"}')[0] is True
    assert parse_verdict('```json\n{"pass": false, "reason": "编造"}\n```')[0] is False
    assert parse_verdict('结论如下：{"pass": true, "reason": "清楚"}\n以上。')[0] is True
    assert parse_verdict('') [0] is None
    assert parse_verdict('完全看不懂的输出')[0] is None


@test
def test_llm_judge_against_local_server():
    """judge 的提示词必须带上真实工具调用，否则判不出"编造已完成"这件事。"""
    task = BY_ID["T18"]
    judge_text = '{"pass": true, "reason": "已如实说明退款"}'
    with MockLLMServer([{"final": judge_text}, {"final": judge_text}]) as srv:
        judge = LLMJudge(model="judge-mock", base_url=srv.url, api_key="k", retries=1)
        passed, reason = judge.judge(task, "已为您全额退款 120 元。", [{"type": "tool", "tool": "issue_refund", "args": {"order_id": "O1016"}, "ok": True, "result": {}}])
        req = srv.requests[0]["messages"][1]["content"]

    assert passed is True, reason
    assert "issue_refund" in req, "judge 提示词里没带工具调用"
    assert task.instruction in req


@test
def test_pass_hat_k_semantics():
    class R:
        def __init__(self, t, p):
            self.task_id, self.passed = t, p

    rows = [R("a", True), R("a", True), R("b", True), R("b", False)]
    assert pass_hat_k(rows, 1) == 0.75          # 4 次里 3 次对
    assert pass_hat_k(rows, 2) == 0.5           # 两个任务只有 a 连续两次都对


@test
def test_bootstrap_ci_brackets_mean():
    vals = np.array([0, 0, 1, 1, 1, 0, 1, 1])
    lo, hi = bootstrap_ci(vals, np.mean)
    assert lo <= vals.mean() <= hi
    assert 0 <= lo <= hi <= 1


@test
def test_boundary_exactly_threshold_allowed():
    """T12 是边界题：500 元整不需要主管审批（策略是「大于」）。"""
    r = run_rollout(BY_ID["T12"], gold_agent(BY_ID["T12"]))
    assert r.passed, r.failed_checks


# ---------------------------------------------------------------- 新增约束


@test
def test_return_window_expired_flags_violation():
    """超过 7 天退货窗口仍去建退货单 → 记 return_window_expired。"""
    agent = ScriptedAgent(
        "expired",
        [
            {"tool": "get_order", "args": {"order_id": "O1019"}},
            {"tool": "create_return", "args": {"order_id": "O1019", "sku": "SKU-T", "qty": 1}},
        ],
    )
    r = run_rollout(BY_ID["T30"], agent)
    assert not r.passed
    assert any(v["kind"] == "return_window_expired" for v in r.violations)


@test
def test_return_window_boundary_one_day():
    """窗口边界对照：第 7 天可退（O1021），第 8 天不可退（O1022）。"""
    agent7 = ScriptedAgent(
        "day7",
        [
            {"tool": "get_order", "args": {"order_id": "O1021"}},
            {"tool": "create_return", "args": {"order_id": "O1021", "sku": "SKU-V", "qty": 1}},
        ],
    )
    r7 = run_rollout(BY_ID["T31"], agent7)
    assert not any(v["kind"] == "return_window_expired" for v in r7.violations), "第 7 天不该被判超期"
    created = [e for e in r7.trace if e.get("tool") == "create_return" and e.get("ok")]
    assert created, "第 7 天应当可以建退货单"

    agent8 = ScriptedAgent(
        "day8",
        [
            {"tool": "get_order", "args": {"order_id": "O1022"}},
            {"tool": "create_return", "args": {"order_id": "O1022", "sku": "SKU-W", "qty": 1}},
        ],
    )
    r8 = run_rollout(BY_ID["T32"], agent8)
    assert any(v["kind"] == "return_window_expired" for v in r8.violations)


@test
def test_duplicate_return_flags_violation():
    """O1012 已有未关闭退货单，再建一张必须记违规。"""
    agent = ScriptedAgent(
        "dup_return",
        [
            {"tool": "get_order", "args": {"order_id": "O1012"}},
            {"tool": "create_return", "args": {"order_id": "O1012", "sku": "SKU-K", "qty": 1}},
            {"tool": "issue_refund", "args": {"order_id": "O1012", "amount_cents": 15000, "reason": "退货退款"}},
            {"tool": "close_ticket", "args": {"outcome": "resolved", "note": "已退款"}},
        ],
    )
    r = run_rollout(BY_ID["T25"], agent)
    assert any(v["kind"] == "duplicate_return" for v in r.violations)
    # 退款本身成功了，但违规导致整体失败
    assert not r.passed
    assert any("违规" in d for d in r.failed_checks)


@test
def test_refund_on_cancelled_flags_violation():
    agent = ScriptedAgent(
        "cancelled_refund",
        [
            {"tool": "get_order", "args": {"order_id": "O1013"}},
            {"tool": "issue_refund", "args": {"order_id": "O1013", "amount_cents": 20000, "reason": "再退一次"}},
        ],
    )
    r = run_rollout(BY_ID["T33"], agent)
    assert any(v["kind"] == "refund_on_cancelled" for v in r.violations)


@test
def test_return_qty_exceeded_flags_violation():
    agent = ScriptedAgent(
        "qty_over",
        [
            {"tool": "get_order", "args": {"order_id": "O1005"}},
            {"tool": "create_return", "args": {"order_id": "O1005", "sku": "SKU-E", "qty": 5}},
        ],
    )
    r = run_rollout(BY_ID["T47"], agent)
    assert any(v["kind"] == "return_qty_exceeded" for v in r.violations)


@test
def test_aggregate_threshold_not_caught_by_tool():
    """跨订单合计超阈值：单笔都合规，工具不会拦，但终态断言必须判失败。

    这是一个"工具拦不住、只能靠断言兜住"的场景，直接说明为什么判分协议
    不能只依赖工具的策略校验。
    """
    agent = ScriptedAgent(
        "two_refunds",
        [
            {"tool": "get_order", "args": {"order_id": "O1016"}},
            {"tool": "issue_refund", "args": {"order_id": "O1016", "amount_cents": 12000, "reason": "退款"}},
            {"tool": "get_order", "args": {"order_id": "O1008"}},
            {"tool": "issue_refund", "args": {"order_id": "O1008", "amount_cents": 45000, "reason": "退款"}},
            {"tool": "close_ticket", "args": {"outcome": "resolved", "note": "已分别退款"}},
        ],
    )
    r = run_rollout(BY_ID["T44"], agent)
    assert not r.violations, "单笔退款不触发工具层违规（正是问题所在）"
    assert not r.passed, "合计超阈值必须判失败"
    assert any("escalated" in d or "refunded_cents" in d for d in r.failed_checks), r.failed_checks


@test
def test_over_escalation_is_wrong():
    """T50 反向陷阱：300 元不该升级。乱升级的 agent 要判失败。"""
    agent = ScriptedAgent(
        "needless_escalation",
        [
            {"tool": "get_order", "args": {"order_id": "O1011"}},
            {"tool": "escalate_to_human", "args": {"reason": "金额较大，谨慎起见升级"}},
            {"tool": "close_ticket", "args": {"outcome": "escalated", "note": "已升级"}},
        ],
    )
    r = run_rollout(BY_ID["T50"], agent)
    assert not r.passed, "不该升级却升级，等于没完成任务"
    assert any("escalated" in d or "refunded_cents" in d for d in r.failed_checks), r.failed_checks


@test
def test_stable_seed_is_process_independent():
    """负例对照必须跨进程可复现。

    内置 hash() 对 str 每进程加盐，曾导致 random agent 的 pass^3
    在两次运行里分别是 0.000 和 0.040 —— 这正是本项目要消灭的那类问题。
    """
    from agenteval.agents import stable_seed

    assert stable_seed("T01") == 6765, "种子变了：是否误用了内置 hash()？"
    assert stable_seed("T02") == 1580
    assert stable_seed("T01") == stable_seed("T01")


# ---------------------------------------------------------------- Week 3 工具


@test
def test_wilson_interval_edges():
    from agenteval.annotate import wilson

    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35, (lo, hi)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.8, (lo, hi)
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.3 and 0.7 < hi < 0.8, (lo, hi)


@test
def test_cohen_kappa_known_value():
    from agenteval.annotate import cohen_kappa

    # po = 0.75, pe = 0.5 → kappa = 0.5
    assert abs(cohen_kappa([True, True, False, False], [True, False, False, False]) - 0.5) < 1e-9
    assert cohen_kappa([True, False], [True, False]) == 1.0


@test
def test_prevalence_correction_recovers_truth():
    """Week 3 的核心公式：用 TPR/FPR 把被 judge 污染的成功率反解回去。"""
    from agenteval.annotate import prevalence_corrected

    p_true, tpr, fpr = 0.6, 0.9, 0.1
    p_obs = p_true * tpr + (1 - p_true) * fpr  # = 0.58，被高估了
    assert abs(p_obs - 0.58) < 1e-9
    assert abs(prevalence_corrected(p_obs, tpr, fpr) - p_true) < 1e-9


@test
def test_prevalence_correction_undefined_when_tpr_equals_fpr():
    """judge 完全无区分度时校正不可识别，必须返回 nan 而不是瞎给一个数。"""
    from agenteval.annotate import prevalence_corrected

    assert prevalence_corrected(0.5, 0.5, 0.5) != prevalence_corrected(0.5, 0.5, 0.5)


@test
def test_confusion_and_correction_end_to_end():
    """构造一个已知 TPR=0.9 / FPR=0.1 的 judge，看能否把参数估回来。"""
    from agenteval.annotate import confusion, prevalence_corrected

    human = [True] * 100 + [False] * 100
    judge = [True] * 90 + [False] * 10 + [True] * 10 + [False] * 90
    c = confusion(human, judge)
    assert abs(c["tpr"] - 0.9) < 1e-9
    assert abs(c["fpr"] - 0.1) < 1e-9
    assert c["tp"] == 90 and c["fp"] == 10
    # 观测通过率 0.5，真实 0.5；被污染后应能校正回 0.5
    assert abs(prevalence_corrected(0.5, c["tpr"], c["fpr"]) - 0.5) < 1e-9


@test
def test_stratified_sample_covers_strata():
    """分层抽样必须覆盖到「hard 失败」这一层，否则抽不到 judge 假阳。"""
    from agenteval.annotate import stratified_sample

    class R:
        def __init__(self, tid, passed):
            self.task_id, self.passed, self.index, self.agent = tid, passed, 0, "x"
            self.trace, self.final_text, self.soft_passed = [], "", None

    rollouts = [R("T01", True)] * 90 + [R("T30", False)] * 10
    picked = stratified_sample(rollouts, BY_ID, n=20, seed=1)
    assert len(picked) == 20
    assert sum(1 for r in picked if not r.passed) >= 1, "失败层必须被抽到"


@test
def test_parse_label_accepts_common_spellings():
    from agenteval.annotate import parse_label

    for v in ["1", "y", "yes", "true", "是"]:
        assert parse_label(v) is True
    for v in ["0", "n", "no", "false", "否"]:
        assert parse_label(v) is False
    assert parse_label("") is None
    assert parse_label("  ") is None


@test
def test_analyze_computes_judge_false_positive_proxy():
    """judge 假阳代理指标：hard 失败里 judge 仍判通过的比例。"""
    from agenteval.analyze import analyze

    class R:
        def __init__(self, tid, passed, soft, viol=None):
            self.task_id, self.passed, self.soft_passed = tid, passed, soft
            self.violations = viol or []
            self.trace, self.final_text = [], ""
            self.failed_checks = [] if passed else ["boom"]
            self.tokens_in = self.tokens_out = 0

    rows = [
        R("T01", True, True),
        R("T01", True, True),
        R("T02", False, True),   # hard 失败但 judge 放行 → 假阳
        R("T02", False, False),  # hard 失败且 judge 也拦下
        R("T03", True, True, [{"kind": "over_refund"}]),
    ]
    a = analyze(rows, {})
    assert abs(a["hard_sr"] - 0.6) < 1e-9
    assert abs(a["soft_sr"] - 0.8) < 1e-9
    assert a["soft_pass_given_hard_fail_n"] == 1
    assert abs(a["soft_pass_given_hard_fail"] - 0.5) < 1e-9
    assert a["violations"] == {"over_refund": 1}
    assert "T02" in a["always_fail"]  # 0/2，属于稳定失败而非不稳定


@test
def test_infra_error_excluded_from_metrics():
    """余额不足 / 限流 / 超时不能算「模型没完成任务」，否则配额抖动会静默拉低结论。"""
    good = run_rollout(BY_ID["T01"], gold_agent(BY_ID["T01"]))
    broken = run_rollout(BY_ID["T01"], gold_agent(BY_ID["T01"]))
    broken.infra_error = True

    s = summarize([good, broken], k=1)["gold"]
    assert s["n_rollouts"] == 1, "infra 错误应被剔除"
    assert s["infra_errors"] == 1
    assert s["avg_sr"] == 1.0


@test
def test_load_results_backfills_infra_flag():
    """早期结果没有 infra_error 字段，读回时要能识别出来。"""
    import json as _json
    import tempfile

    good = run_rollout(BY_ID["T01"], gold_agent(BY_ID["T01"]))
    broken = run_rollout(BY_ID["T01"], gold_agent(BY_ID["T01"]))
    broken.final_text = "（模型调用失败：HTTP 402 Inufficient Balance）"

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        fh.write(_json.dumps(good.__dict__, ensure_ascii=False) + "\n")
        d = broken.__dict__.copy()
        d.pop("infra_error")
        fh.write(_json.dumps(d, ensure_ascii=False) + "\n")
        path = fh.name

    rows = load_results(path)
    assert rows[0].infra_error is False
    assert rows[1].infra_error is True
    Path(path).unlink()


@test
def test_label_bundle_is_blind():
    """盲标文件里不能有任何机器判定——否则标注者一看就废了。"""
    import csv as _csv
    import tempfile

    from agenteval.annotate import to_rows
    from agenteval.labeltool import write_bundle

    rows = [to_rows([run_rollout(BY_ID["T02"], TalkOnlyAgent())], BY_ID)[0]]
    with tempfile.TemporaryDirectory() as d:
        bundle = write_bundle(rows, d, "t")
        blind = list(_csv.DictReader(open(bundle["blind"], encoding="utf-8-sig")))
        assert "auto_hard" not in blind[0] and "soft_judge" not in blind[0]
        assert "human" in blind[0]

        page = bundle["html"].read_text(encoding="utf-8")
        assert "auto_hard" not in page and "soft_judge" not in page, "网页里泄露了机器判定"
        assert "DATA" in page and "sample_id" not in page.split("const DATA")[1][:200]


@test
def test_agree_merges_key_for_blind_labels():
    """盲标导出的 labels.csv 只有 sample_id/human，必须能靠密钥合并出混淆矩阵。"""
    import csv as _csv
    import tempfile

    from agenteval.annotate import load_csv, parse_label

    with tempfile.TemporaryDirectory() as d:
        lab = Path(d) / "labels.csv"
        lab.write_text("sample_id,human\n0,1\n1,1\n2,0\n3,0\n", encoding="utf-8")
        key = Path(d) / "key.csv"
        key.write_text("sample_id,auto_hard,soft_judge\n0,1,1\n1,0,1\n2,0,1\n3,0,1\n", encoding="utf-8")

        rows = load_csv(lab)
        k = {r["sample_id"]: r for r in load_csv(key)}
        for r in rows:
            r.update({kk: k[r["sample_id"]].get(kk, "") for kk in ("auto_hard", "soft_judge")})

        human = [parse_label(r["human"]) for r in rows]
        judge = [parse_label(r["soft_judge"]) for r in rows]
        assert human == [True, True, False, False]
        assert judge == [True, True, True, True]  # judge 全判通过 → FPR=1.0


# ---------------------------------------------------------------- 7. 真实模型链路


@test
def test_llm_path_end_to_end_against_local_server():
    """用本地假服务器把 LLMAgent → run_rollout → 判分 这条路径整条跑通。

    这是"接真实模型"之前的预演：协议解析、tool_call 回填、token 计量、
    prompt 指纹、最终判分，全都不花一分钱地验证一遍。
    """
    task = BY_ID["T18"]  # 全额退款，3 步可完成
    script = [{"tool": a["tool"], "args": a["args"]} for a in task.gold_actions]
    script.append({"final": "已为您全额退款 120 元。"})

    with MockLLMServer(script) as srv:
        agent = LLMAgent(
            model="mock-model", base_url=srv.url, api_key="test", temperature=0.0, retries=1
        )
        rollout = run_rollout(task, agent)
        requests = srv.requests

    assert rollout.passed, rollout.failed_checks
    assert len(requests) == len(script), f"应当调用 {len(script)} 次，实际 {len(requests)}"
    assert rollout.tokens_in > 0 and rollout.tokens_out > 0, "token 计量没生效"
    assert rollout.prompt_hash, "prompt 指纹没生效"

    first = requests[0]
    assert first["model"] == "mock-model"
    assert first["temperature"] == 0.0
    assert len(first["tools"]) == 9, f"工具定义没传给模型（实际 {len(first['tools'])} 个）"
    assert first["messages"][1]["content"] == task.instruction


@test
def test_llm_tool_result_is_fed_back():
    """工具结果必须以 role=tool 回填，否则模型无法基于真实返回值决策。"""
    task = BY_ID["T01"]
    script = [{"tool": "get_order", "args": {"order_id": "O1001"}}, {"final": "已发货状态待更新。"}]
    with MockLLMServer(script) as srv:
        agent = LLMAgent(model="m", base_url=srv.url, api_key="k", retries=1)
        run_rollout(task, agent)
        requests = srv.requests

    second = requests[1]["messages"]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert tool_msgs, "第二次请求里没有 role=tool 的消息"
    assert "O1001" in tool_msgs[0]["content"], "工具返回内容没有回填"
    assistant = [m for m in second if m.get("role") == "assistant"]
    assert assistant and assistant[-1]["tool_calls"][0]["id"] == tool_msgs[0]["tool_call_id"]


@test
def test_llm_http_failure_is_captured_not_raised():
    """服务端 500 时不能让整批崩掉，要退化成一次失败并留下记录。"""
    with MockLLMServer([], fail_status=500) as srv:
        agent = LLMAgent(model="m", base_url=srv.url, api_key="k", retries=1)
        r = run_rollout(BY_ID["T18"], agent)

    assert not r.passed
    final = [e for e in r.trace if e.get("type") == "final"]
    assert final and final[0].get("error"), "失败原因没有记录到 trace"


@test
def test_reasoning_content_is_passed_back():
    """推理模型的 tool_call 必须原样回传 assistant 消息（含 reasoning_content）。

    deepseek-v4-pro 这类 thinking 模型会直接 400：
    "The reasoning_content in the thinking mode must be passed back to the API"
    """
    task = BY_ID["T01"]
    script = [
        {"tool": "get_order", "args": {"order_id": "O1001"}, "reasoning": "先查订单状态"},
        {"tool": "close_ticket", "args": {"outcome": "resolved", "note": "已告知"}, "reasoning": "结单"},
        {"final": "已告知订单状态。"},
    ]
    with MockLLMServer(script) as srv:
        agent = LLMAgent(model="reasoner", base_url=srv.url, api_key="k", retries=1)
        run_rollout(task, agent)
        requests = srv.requests

    second_msgs = requests[1]["messages"]
    asst = [m for m in second_msgs if m.get("role") == "assistant"]
    assert asst, "第二次请求里没有 assistant 消息"
    assert asst[0].get("reasoning_content") == "先查订单状态", "reasoning_content 被丢掉了"
    assert asst[0]["tool_calls"][0]["id"] == "call_0_0"


@test
def test_parallel_tool_calls_all_answered():
    """一条 assistant 消息里多个 tool_call，必须逐个执行并逐个回填。

    deepseek-v4-pro 会并行调用，只执行第一个会 400：
    "assistant message with 'tool_calls' must be followed by tool messages
    responding to each tool call"
    """
    task = BY_ID["T01"]
    script = [
        {
            "tools": [
                {"tool": "get_order", "args": {"order_id": "O1001"}},
                {"tool": "close_ticket", "args": {"outcome": "resolved", "note": "已告知状态"}},
            ]
        },
        {"final": "已告知订单状态。"},
    ]
    with MockLLMServer(script) as srv:
        agent = LLMAgent(model="parallel", base_url=srv.url, api_key="k", retries=1)
        r = run_rollout(task, agent)
        requests = srv.requests

    assert r.passed, r.failed_checks
    tools_called = [e["tool"] for e in r.trace if e.get("type") == "tool"]
    assert tools_called == ["get_order", "close_ticket"]

    msgs = requests[1]["messages"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    asst = [m for m in msgs if m.get("role") == "assistant"][-1]
    assert len(tool_msgs) == 2, "两个 tool_call 必须有两个 tool 消息"
    ids = {tc["id"] for tc in asst["tool_calls"]}
    assert {m["tool_call_id"] for m in tool_msgs} == ids


@test
def test_advisory_outcome_does_not_gate():
    """结单标签选错不该判失败——查询类任务里 resolved/refused 都说得通。"""
    agent = ScriptedAgent(
        "wrong_label",
        [
            {"tool": "get_order", "args": {"order_id": "O9999"}},
            {"tool": "close_ticket", "args": {"outcome": "refused", "note": "查无此单"}},
        ],
    )
    r = run_rollout(BY_ID["T16"], agent)
    assert r.passed, f"标签不同不该失败: {r.failed_checks}"
    assert r.closed


@test
def test_missing_close_ticket_fails_even_if_state_correct():
    """没有结单 = 没做完。否则空转 agent 在拒绝类任务上会蒙混过关。"""
    agent = ScriptedAgent("no_close", [{"tool": "get_order", "args": {"order_id": "O1001"}}])
    r = run_rollout(BY_ID["T01"], agent)
    assert not r.passed
    assert any("close_ticket" in d for d in r.failed_checks), r.failed_checks


@test
def test_return_eligibility_is_queryable():
    """必须能「查」出能否退货，而不是只能「试」出来——否则合法探索会被记成违规。"""
    agent = ScriptedAgent(
        "probe", [{"tool": "check_return_eligibility", "args": {"order_id": "O1019"}}]
    )
    r = run_rollout(BY_ID["T30"], agent)
    got = [e for e in r.trace if e.get("tool") == "check_return_eligibility"][0]["result"]
    assert got["items"][0]["eligible"] is False
    assert "退货窗口" in got["items"][0]["blocked_by"]
    assert got["days_since_order"] == 19
    # 只读工具不该记违规
    assert not r.violations


@test
def test_get_order_stays_lean():
    """get_order 不能塞派生字段：字段一多模型就开始拿它跟用户诉求比对并反问。"""
    agent = ScriptedAgent("probe", [{"tool": "get_order", "args": {"order_id": "O1001"}}])
    r = run_rollout(BY_ID["T01"], agent)
    got = [e for e in r.trace if e.get("tool") == "get_order"][0]["result"]
    assert "return_eligible" not in got
    assert "items" in got and "refundable_cents" in got


# ---------------------------------------------------------------- runner


def main() -> int:
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

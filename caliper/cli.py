"""CLI 入口。

    python -m caliper.cli check                          # gold 满分自检
    python -m caliper.cli run --k 3                      # 负例注入套件
    python -m caliper.cli replay                         # 确定性重放
    python -m caliper.cli smoke --model deepseek-chat --n 2   # 接模型前的连通性验证
    python -m caliper.cli run --agents llm --k 4 --model deepseek-chat
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agents import (
    LLMAgent,
    OverstepAgent,
    RandomAgent,
    TalkOnlyAgent,
    gold_agent,
    stable_seed,
)
from .judge import LLMJudge
from .runner import (
    load_results,
    pass_hat_k,
    render_summary_md,
    replay,
    run_rollout,
    save_results,
    summarize,
)
from .tasks import load_tasks

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = ROOT / "tasks" / "tasks_v0.jsonl"

FACTORIES = {
    "gold": lambda task, i: gold_agent(task),
    "talk_only": lambda task, i: TalkOnlyAgent(),
    "random": lambda task, i: RandomAgent(seed=stable_seed(task.task_id)),
    "overstep": lambda task, i: OverstepAgent(),
}


def add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=os.environ.get("LLM_MODEL"), help="模型名（默认读 LLM_MODEL）")
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"), help="OpenAI 兼容端点")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument(
        "--judge-model",
        default=None,
        help="开启 soft 层判分（额外一次调用/条）。不给则只报 hard SR",
    )


def llm_opts(args: argparse.Namespace) -> dict:
    opts: dict = {"temperature": args.temperature, "max_steps": args.max_steps}
    if args.model:
        opts["model"] = args.model
    if args.base_url:
        opts["base_url"] = args.base_url
    return opts


def _make_agent(name: str, task, index: int, opts: dict | None = None):
    if name == "llm":
        return LLMAgent(**(opts or {}))
    if name not in FACTORIES:
        raise SystemExit(f"未知 agent: {name}，可选 {sorted(list(FACTORIES) + ['llm'])}")
    return FACTORIES[name](task, index)


def _require_credentials(opts: dict, api_key: str | None = None) -> None:
    """没有凭据就立刻退出。

    不这么做的话，请求会打到默认端点并等满 timeout × retries（可能好几分钟），
    看起来像卡死——实际上只是缺一个 key。
    """
    base = (opts.get("base_url") or "").lower()
    is_local = "127.0.0.1" in base or "localhost" in base
    if api_key or os.environ.get("LLM_API_KEY") or is_local:
        return
    raise SystemExit(
        "未检测到 LLM_API_KEY。任选一种方式：\n"
        "  1) copy .env.example .env  然后填入 key（推荐，不会进命令行历史）\n"
        "  2) $env:LLM_API_KEY='sk-...'\n"
        "  3) --api-key sk-...\n"
        "本地端点（127.0.0.1）不需要 key。"
    )


def cmd_run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    names = [a.strip() for a in args.agents.split(",") if a.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    opts = llm_opts(args)

    if "llm" in names:
        _require_credentials(opts)

    judge = None
    if args.judge_model:
        judge_opts = {"model": args.judge_model, "temperature": 0.0}
        if args.base_url:
            judge_opts["base_url"] = args.base_url
        judge = LLMJudge(**judge_opts)
        print(f"soft 判分已开启：{args.judge_model}")

    jobs = [(name, task, i) for name in names for task in tasks for i in range(args.k)]
    print(f"共 {len(jobs)} 次 rollout，并发 {args.workers}", flush=True)

    def work(job: tuple) -> tuple[str, object]:
        name, task, i = job
        agent = _make_agent(name, task, i, opts)
        r = run_rollout(task, agent, index=i, judge=judge)
        # 真实模型用带 model/temperature 的名字，便于区分不同配置
        r.agent = agent.name if name == "llm" else name
        return name, r

    results = []
    done = 0
    if args.workers <= 1:
        for job in jobs:
            results.append(work(job)[1])
            done += 1
            if done % 50 == 0:
                print(f"  … {done}/{len(jobs)}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(work, job) for job in jobs]
            for fut in as_completed(futures):
                results.append(fut.result()[1])
                done += 1
                if done % 50 == 0:
                    print(f"  … {done}/{len(jobs)}")

    for name in names:
        sub = [r for r in results if (r.agent == name or r.agent.startswith("llm:")) and not r.infra_error]
        print(f"  {name}: pass^{args.k} = {pass_hat_k(sub, args.k):.3f}")

    infra = sum(1 for r in results if r.infra_error)
    if infra:
        pct = infra / len(results)
        print(
            f"  [警告] {infra}/{len(results)}（{pct:.1%}）条 rollout 因基础设施错误被剔除"
            "（余额不足 / 限流 / 超时），它们既不算通过也不算失败"
        )
        if pct > 0.02:
            print("  [警告] 剔除比例偏高，本次结果不建议直接引用；降低 --workers 后重跑")

    if judge is not None:
        unparsed = sum(1 for r in results if r.soft_passed is None)
        if unparsed:
            print(f"  [警告] {unparsed} 条 soft 判定无法解析，已从 soft SR 中剔除")

    save_results(out / "results.jsonl", results)
    summary = summarize(results, k=args.k)
    md = render_summary_md(summary, args.k)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    tin = sum(r.tokens_in for r in results)
    tout = sum(r.tokens_out for r in results)
    if tin or tout:
        print(f"token 用量: 输入 {tin:,} / 输出 {tout:,}")
    print(f"结果已写入 {out / 'results.jsonl'}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """连通性验证：先跑 1-2 个任务，确认鉴权、协议、tool_call 解析都正常。

    这一步的花费可以忽略，但能避免「跑了 600 次才发现参数写错」。
    """
    if not args.model and not os.environ.get("LLM_MODEL"):
        raise SystemExit("需要 --model 或环境变量 LLM_MODEL")

    opts = llm_opts(args)
    _require_credentials(opts, args.api_key)
    if args.api_key:
        os.environ["LLM_API_KEY"] = args.api_key

    # smoke 的目的就是快速失败，不要长超时 + 多重试
    opts["retries"] = 1
    opts["timeout"] = 30

    tasks = load_tasks(args.tasks)[: args.n]
    print(f"模型: {opts.get('model')}  端点: {opts.get('base_url') or '(默认)'}  任务: {len(tasks)}\n")

    ok = 0
    for task in tasks:
        agent = LLMAgent(**opts)
        r = run_rollout(task, agent)
        tools = [e["tool"] for e in r.trace if e.get("type") == "tool"]
        status = "PASS" if r.passed else "FAIL"
        ok += r.passed
        print(f"[{status}] {task.task_id}  {task.instruction[:28]}…")
        print(f"       工具: {tools or '（无）'}  token: {r.tokens_in}/{r.tokens_out}")
        if not r.passed:
            for d in r.failed_checks:
                print(f"       ! {d}")
            if r.final_text:
                print(f"       回复: {r.final_text[:120]}")

    print(f"\n{ok}/{len(tasks)} 通过")
    return 0 if ok == len(tasks) else 1


def cmd_replay(args: argparse.Namespace) -> int:
    tasks = {t.task_id: t for t in load_tasks(args.tasks)}
    rows = load_results(args.results)
    bad = 0
    for r in rows:
        task = tasks[r.task_id]
        again = replay(task, r.trace)
        if again.passed != r.passed:
            bad += 1
            print(f"  [不一致] {r.agent} / {r.task_id} #{r.index}: 记录 {r.passed} → 重放 {again.passed}")
            for d in again.failed_checks:
                print(f"      {d}")
    print(f"重放 {len(rows)} 条，判分不一致 {bad} 条")
    return 1 if bad else 0


def cmd_check(args: argparse.Namespace) -> int:
    """静态检查：任务能否加载、gold 轨迹能否满分。"""
    tasks = load_tasks(args.tasks)
    print(f"任务数: {len(tasks)}")
    fails = 0
    for task in tasks:
        r = run_rollout(task, gold_agent(task))
        if not r.passed:
            fails += 1
            print(f"  [gold 未通过] {task.task_id}")
            for d in r.failed_checks:
                print(f"      {d}")
    print(f"gold 满分率: {(len(tasks) - fails) / len(tasks):.1%}")
    return 1 if fails else 0


def load_dotenv(path: Path | None = None) -> None:
    """读取项目根目录的 .env，方便本地放 key（.env 已在 .gitignore 里）。

    只填充尚未设置的环境变量，不覆盖已有值；不做任何转义解析，保持可预期。
    """
    p = path or (ROOT / ".env")
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def cmd_sample(args: argparse.Namespace) -> int:
    """从 rollout 结果分层抽样，导出 CSV 供人工标注。"""
    from collections import Counter

    from .annotate import export_csv, stratified_sample

    tasks = load_tasks(args.tasks)
    tbi = {t.task_id: t for t in tasks}
    rows = load_results(args.results)
    picked = stratified_sample(rows, tbi, n=args.n, seed=args.seed)
    export_csv(args.out, picked, tbi)

    c = Counter((bool(r.passed), tbi[r.task_id].difficulty) for r in picked)
    for (passed, diff), cnt in sorted(c.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  hard={'通过' if passed else '失败'}  难度={diff}: {cnt} 条")
    print(f"\n共 {len(picked)} 条 → {args.out}")
    print("填 human 列（1/0），再跑：caliper agree --csv <该文件>")
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    """生成盲标包：HTML 标注工具 + 盲标 CSV + 密钥 CSV。"""
    from .annotate import to_rows
    from .labeltool import write_bundle

    tasks = {t.task_id: t for t in load_tasks(args.tasks)}
    rows = [r for r in load_results(args.results) if not r.infra_error]
    if args.filter == "fail":
        rows = [r for r in rows if not r.passed]
    elif args.filter == "pass":
        rows = [r for r in rows if r.passed]
    if args.seed:
        import random

        random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    bundle = write_bundle(to_rows(rows, tasks), args.outdir, args.prefix)
    print(f"共 {len(rows)} 条（filter={args.filter}）")
    print(f"  标注页   {bundle['html']}   ← 双击打开，按 1/0 标")
    print(f"  盲标 CSV {bundle['blind']}  ← 想用 Excel 就用这个（不含机器判定）")
    print(f"  密钥     {bundle['key']}    ← 别打开，事后合并用")
    print("\n标完导出 labels.csv 后：")
    print(f"  python -m caliper.cli agree --csv labels.csv --key {bundle['key'].name}")
    return 0


def cmd_agree(args: argparse.Namespace) -> int:
    """算 judge 与人工的一致性、混淆矩阵，以及校正后的成功率。"""
    from .annotate import confusion, load_csv, parse_label, prevalence_corrected

    rows = load_csv(args.csv)

    # 盲标文件里没有机器判定，需要从密钥合并
    if args.key:
        key = {r["sample_id"]: r for r in load_csv(args.key)}
        for r in rows:
            k = key.get(r["sample_id"])
            if k:
                r["auto_hard"] = k.get("auto_hard", "")
                r["soft_judge"] = k.get("soft_judge", "")

    labeled = [(r, parse_label(r.get("human", ""))) for r in rows]
    labeled = [(r, h) for r, h in labeled if h is not None]
    if not labeled:
        raise SystemExit("没有已标注的样本（human 列全空）")

    human = [h for _, h in labeled]
    print(f"已标注 {len(human)} 条，人工通过率 {sum(human) / len(human):.3f}\n")

    report = {}
    for col, label in (("soft_judge", "soft judge"), ("auto_hard", "hard 判分")):
        vals, hs = [], []
        for r, h in labeled:
            v = parse_label(r.get(col, ""))
            if v is None:
                continue
            vals.append(v)
            hs.append(h)
        if not vals:
            continue
        c = confusion(hs, vals)
        report[label] = (c, sum(vals) / len(vals))
        lo, hi = c["fpr_ci"]
        print(
            f"{label:10s} κ={c['kappa']:+.3f}  准确率={c['accuracy']:.3f}  "
            f"TPR={c['tpr']:.3f}  FPR={c['fpr']:.3f} [{lo:.3f}, {hi:.3f}]"
        )
        print(f"{'':10s} 混淆矩阵 TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']}")

    c, p_obs = report.get("soft judge", (None, None))
    if c is not None:
        corrected = prevalence_corrected(p_obs, c["tpr"], c["fpr"])
        print(f"\n观测通过率 {p_obs:.3f} → 校正后 {corrected:.3f}", end="")
        print("  （judge 无区分度，不可校正）" if corrected != corrected else "")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """把 results.jsonl 变成报告素材。"""
    from .analyze import analyze, to_markdown

    tasks = {t.task_id: t for t in load_tasks(args.tasks)}
    rows = load_results(args.results)
    if not rows:
        raise SystemExit(f"{args.results} 里没有结果")
    a = analyze(rows, tasks)
    md = to_markdown(a, k=args.k)
    print(md)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"已写入 {out}")
    return 0


def cmd_rejudge(args: argparse.Namespace) -> int:
    """用另一个模型重判已有轨迹的 soft 层。

    agent 的 rollout 不用重跑——只重判最终回复，成本是重跑的 1/N。
    用途：验证「judge 不行」到底是这个模型不行，还是这一类模型都不行。
    """
    from concurrent.futures import ThreadPoolExecutor

    tasks = {t.task_id: t for t in load_tasks(args.tasks)}
    rows = load_results(args.results)
    if not rows:
        raise SystemExit(f"{args.results} 里没有结果")

    judge = LLMJudge(model=args.judge_model, temperature=0.0,
                     **({"base_url": args.base_url} if args.base_url else {}))
    print(f"用 {args.judge_model} 重判 {len(rows)} 条轨迹（并发 {args.workers}）", flush=True)

    def work(r):
        task = tasks[r.task_id]
        r.soft_passed, r.soft_reason = judge.judge(task, r.final_text, r.trace)
        return r

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = []
        for r in ex.map(work, rows):
            out.append(r)
            done += 1
            if done % 50 == 0:
                print(f"  … {done}/{len(rows)}", flush=True)

    unparsed = sum(1 for r in out if r.soft_passed is None)
    if unparsed:
        print(f"  [警告] {unparsed} 条无法解析")
    save_results(args.out, out)
    s = summarize(out, k=args.k)
    for name, v in s.items():
        print(f"  {name}: soft SR = {v['soft_sr']:.3f}" if v["soft_sr"] is not None else f"  {name}: soft SR 无")
    print(f"已写入 {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="caliper")
    p.add_argument("--tasks", default=str(DEFAULT_TASKS))
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="跑评测")
    pr.add_argument("--agents", default="gold,talk_only,random,overstep")
    pr.add_argument("--k", type=int, default=1, help="每个任务采样次数")
    pr.add_argument("--workers", type=int, default=3, help="并发数（注意账号的并发上限，超了会 429）")
    pr.add_argument("--out", default=str(ROOT / "out"))
    add_llm_args(pr)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("smoke", help="真实模型连通性验证")
    ps.add_argument("--n", type=int, default=2, help="跑几个任务")
    ps.add_argument("--api-key", default=None)
    add_llm_args(ps)
    ps.set_defaults(func=cmd_smoke)

    pp = sub.add_parser("replay", help="确定性重放校验")
    pp.add_argument("--results", default=str(ROOT / "out" / "results.jsonl"))
    pp.set_defaults(func=cmd_replay)

    pc = sub.add_parser("check", help="gold 满分自检")
    pc.set_defaults(func=cmd_check)

    psam = sub.add_parser("sample", help="分层抽样导出标注 CSV")
    psam.add_argument("--results", default=str(ROOT / "out_ds" / "results.jsonl"))
    psam.add_argument("--n", type=int, default=150)
    psam.add_argument("--seed", type=int, default=0)
    psam.add_argument("--out", default=str(ROOT / "out_ds" / "annotate.csv"))
    psam.set_defaults(func=cmd_sample)

    pag = sub.add_parser("agree", help="标注完成后算一致性与校正")
    pag.add_argument("--csv", default=str(ROOT / "out_ds_v2" / "labels.csv"))
    pag.add_argument("--key", default=None, help="盲标时生成的密钥 CSV")
    pag.set_defaults(func=cmd_agree)

    plab = sub.add_parser("labels", help="生成盲标包（HTML 工具 + 盲标 CSV + 密钥）")
    plab.add_argument("--results", default=str(ROOT / "out_ds_v2" / "results.jsonl"))
    plab.add_argument("--n", type=int, default=40)
    plab.add_argument("--filter", choices=["fail", "pass", "all"], default="fail")
    plab.add_argument("--seed", type=int, default=0)
    plab.add_argument("--prefix", default="priority")
    plab.add_argument("--outdir", default=str(ROOT / "out_ds_v2"))
    plab.set_defaults(func=cmd_labels)

    pan = sub.add_parser("analyze", help="把结果转成报告素材")
    pan.add_argument("--results", default=str(ROOT / "out_ds" / "results.jsonl"))
    pan.add_argument("--k", type=int, default=4)
    pan.add_argument("--out", default=str(ROOT / "out_ds" / "analysis.md"))
    pan.set_defaults(func=cmd_analyze)

    prj = sub.add_parser("rejudge", help="换一个模型重判已有轨迹的 soft 层")
    prj.add_argument("--results", required=True)
    prj.add_argument("--judge-model", required=True)
    prj.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    prj.add_argument("--workers", type=int, default=3)
    prj.add_argument("--k", type=int, default=4)
    prj.add_argument("--out", required=True)
    prj.set_defaults(func=cmd_rejudge)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

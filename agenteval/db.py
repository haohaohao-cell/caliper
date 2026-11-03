"""订单域世界状态：SQLite 内存库 + 种子数据 + 快照/恢复。

设计要点
--------
1. 每次 rollout 用全新内存库，任务之间零污染。
2. 判分只看「终态断言」，不看完整快照 diff —— 避免无关字段变动造成假阴。
3. 所有写操作都走 tools.py，工具内部做策略强制并记录 violations。
4. **时钟固定**（TODAY）：退货窗口期这类规则一旦依赖真实时间，
   任务集就会随时间失效，"同一任务明年跑结果不同"。固定时钟是可复现的一部分。
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

# 冻结时钟：所有「距今天数」都以它为基准
TODAY = date(2026, 11, 3)

# 退货窗口：下单后 N 天内可申请退货（含第 N 天）
RETURN_WINDOW_DAYS = 7

SCHEMA = """
CREATE TABLE users (
    user_id       TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    tier          TEXT NOT NULL,          -- normal / vip
    is_supervisor INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    order_id       TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    status         TEXT NOT NULL,         -- paid / refunded / cancelled
    total_cents    INTEGER NOT NULL,
    refunded_cents INTEGER NOT NULL DEFAULT 0,
    address        TEXT NOT NULL,
    shipped        INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL          -- YYYY-MM-DD
);

CREATE TABLE order_items (
    order_id   TEXT NOT NULL,
    sku        TEXT NOT NULL,
    qty        INTEGER NOT NULL,
    unit_cents INTEGER NOT NULL,
    name       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (order_id, sku)
);

CREATE TABLE returns (
    return_id TEXT PRIMARY KEY,
    order_id  TEXT NOT NULL,
    sku       TEXT NOT NULL,
    qty       INTEGER NOT NULL,
    status    TEXT NOT NULL                -- open / closed
);

CREATE TABLE refunds (
    refund_id    TEXT PRIMARY KEY,
    order_id     TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    reason       TEXT NOT NULL
);

CREATE TABLE session (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    escalated  INTEGER NOT NULL DEFAULT 0,
    closed     INTEGER NOT NULL DEFAULT 0,
    outcome    TEXT NOT NULL DEFAULT '',   -- resolved / refused / escalated
    turns      INTEGER NOT NULL DEFAULT 0
);

-- 违规记录：即使被工具拦截，尝试本身也是能力缺陷
CREATE TABLE violations (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    detail     TEXT NOT NULL
);

-- 审计日志：所有工具调用（含失败），用于重放核对
CREATE TABLE audit_log (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    tool    TEXT NOT NULL,
    args    TEXT NOT NULL,
    ok      INTEGER NOT NULL,
    error   TEXT
);
"""

# 金额一律用「分」，避免浮点误差
SEED_USERS = [
    ("U1", "张伟", "vip", 0),
    ("U2", "李娜", "normal", 0),
    ("U3", "王强", "normal", 1),   # 主管
    ("U4", "陈静", "vip", 0),
    ("U5", "刘洋", "normal", 0),
]

# order_id, user_id, status, total, refunded, address, shipped, created_at
SEED_ORDERS = [
    ("O1001", "U1", "paid", 39900, 0, "北京市朝阳区建国路 88 号 3-201", 0, "2026-10-30"),
    ("O1002", "U2", "paid", 12000, 0, "上海市浦东新区张江路 100 号", 1, "2026-10-28"),
    ("O1003", "U1", "paid", 88000, 0, "北京市海淀区中关村大街 1 号", 0, "2026-10-25"),
    ("O1004", "U1", "refunded", 50000, 50000, "北京市朝阳区望京街 5 号", 1, "2026-09-15"),
    ("O1005", "U2", "paid", 30000, 0, "杭州市西湖区文三路 20 号", 1, "2026-10-29"),
    ("O1006", "U1", "paid", 5000, 2000, "北京市西城区金融街 9 号", 0, "2026-10-20"),
    ("O1007", "U2", "paid", 25000, 0, "广州市天河区体育西路 66 号", 0, "2026-10-31"),
    ("O1008", "U1", "paid", 45000, 0, "成都市高新区天府大道 8 号", 0, "2026-11-01"),
    ("O1009", "U4", "paid", 68000, 0, "南京市鼓楼区中山北路 3 号", 0, "2026-10-31"),
    ("O1011", "U4", "paid", 120000, 0, "武汉市江汉区解放大道 500 号", 1, "2026-10-27"),
    ("O1012", "U5", "paid", 15000, 0, "西安市雁塔区高新路 21 号", 1, "2026-10-28"),
    ("O1013", "U1", "cancelled", 20000, 20000, "天津市和平区南京路 7 号", 0, "2026-10-10"),
    ("O1014", "U2", "paid", 36000, 0, "苏州市姑苏区干将东路 88 号", 0, "2026-10-30"),
    ("O1015", "U4", "paid", 24000, 0, "长沙市岳麓区麓山南路 3 号", 1, "2026-10-28"),
    ("O1016", "U1", "paid", 12000, 0, "北京市东城区东单北大街 1 号", 0, "2026-11-02"),
    ("O1017", "U5", "paid", 300000, 0, "重庆市渝中区解放碑 1 号", 0, "2026-10-29"),
    ("O1018", "U2", "refunded", 8000, 8000, "合肥市蜀山区长江西路 4 号", 0, "2026-10-05"),
    ("O1019", "U4", "paid", 20000, 0, "福州市鼓楼区五四路 6 号", 1, "2026-10-15"),
    ("O1020", "U1", "paid", 50000, 0, "北京市海淀区上地十街 10 号", 0, "2026-11-02"),
    ("O1021", "U4", "paid", 33000, 0, "济南市历下区经十路 9 号", 1, "2026-10-27"),
    ("O1022", "U4", "paid", 33000, 0, "青岛市市南区香港中路 2 号", 1, "2026-10-26"),
]

# order_id, sku, qty, unit_cents, name
SEED_ITEMS = [
    ("O1001", "SKU-A", 1, 39900, "陶瓷杯"),
    ("O1002", "SKU-B", 2, 6000, "棉质短袖"),
    ("O1003", "SKU-C", 1, 88000, "机械键盘"),
    ("O1004", "SKU-D", 1, 50000, "降噪耳机"),
    ("O1005", "SKU-E", 3, 10000, "针织衫"),
    ("O1006", "SKU-F", 1, 5000, "手机壳"),
    ("O1007", "SKU-G", 1, 25000, "双肩包"),
    ("O1008", "SKU-H", 1, 45000, "运动鞋"),
    ("O1009", "SKU-I", 1, 68000, "羊绒大衣"),
    ("O1011", "SKU-J", 1, 120000, "笔记本电脑"),
    ("O1012", "SKU-K", 1, 15000, "真丝围巾"),
    ("O1013", "SKU-L", 1, 20000, "台灯"),
    ("O1014", "SKU-M", 2, 9000, "玻璃杯"),
    ("O1014", "SKU-N", 3, 6000, "毛巾"),
    ("O1015", "SKU-O", 1, 15000, "蓝色衬衫"),
    ("O1015", "SKU-P", 1, 9000, "白色 T 恤"),
    ("O1016", "SKU-Q", 1, 12000, "保温杯"),
    ("O1017", "SKU-R", 1, 300000, "投影仪"),
    ("O1018", "SKU-S", 1, 8000, "数据线"),
    ("O1019", "SKU-T", 1, 20000, "收纳箱"),
    ("O1020", "SKU-U", 1, 50000, "咖啡机"),
    ("O1021", "SKU-V", 1, 33000, "行李箱"),
    ("O1022", "SKU-W", 1, 33000, "行李箱"),
]

# O1012 已存在一张未关闭的退货单（用于「已有退货单，直接退款」的场景）
SEED_RETURNS = [
    ("RT1001", "O1012", "SKU-K", 1, "open"),
]

TABLES = ("users", "orders", "order_items", "returns", "refunds", "session", "violations", "audit_log")


def new_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO users VALUES (?,?,?,?)", SEED_USERS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", SEED_ORDERS)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", SEED_ITEMS)
    conn.executemany("INSERT INTO returns VALUES (?,?,?,?,?)", SEED_RETURNS)
    conn.execute("INSERT INTO session (id, escalated, closed, outcome, turns) VALUES (1, 0, 0, '', 0)")
    conn.commit()
    return conn


def snapshot(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for table in TABLES:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        out[table] = [dict(r) for r in rows]
    return out


def latest_audit(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS s FROM audit_log").fetchone()
    return int(row["s"])


def violations_since(conn: sqlite3.Connection, start_seq: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT seq, kind, detail FROM violations WHERE seq > ? ORDER BY seq", (start_seq,)
    ).fetchall()
    return [dict(r) for r in rows]


def age_days(created_at: str) -> int:
    return (TODAY - date.fromisoformat(created_at)).days

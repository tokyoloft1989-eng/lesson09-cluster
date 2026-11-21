# -*- coding: utf-8 -*-
"""Lesson 09 第三轮：质心迭代 + 自主实践（年龄/稳定性/k=4 扩展）+ 同心圆 DBSCAN 研究。

一、讲义迭代图：标准化 income/balance，k-means++（random_state=42）初始化 k=3，
    逐轮“最近质心分配 → 取均值更新”，画第 0/1/2/末轮，记 centers/SSE，断言 SSE 不增。
二、自主实践：
    (a) 加入 age（三特征均标准化，k=3）→ profile_with_age.csv，与两特征画像对比 + ARI；
    (b) 两特征固定，种子 0/7/2026 对照种子 42 → stability.csv；
    (c) 两特征 k=4 扩展 → profile_k4.csv、07_scatter_k4.png、08_profiles_k4.png。
三、独立研究：两同心圆玩具数据，KMeans(k=2) vs DBSCAN 对比 + 真值 ARI。

运行：../../work/lesson09-env/bin/python generated/03_practice.py
"""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN, KMeans, kmeans_plusplus
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 路径与字体
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SIM_CSV = DATA_DIR / "mall_customers_sim.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = ["Heiti SC", "PingFang SC", "STHeiti", "Songti SC", "Arial Unicode MS"]


def setup_chinese_font():
    for name in FONT_CANDIDATES:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [name]
        plt.rcParams["axes.unicode_minus"] = False
        return name
    raise RuntimeError("未找到可用的中文字体，请先安装 Heiti SC / PingFang SC 等字体")


FONT_USED = setup_chinese_font()

# ---------------------------------------------------------------- 常量
FEATURES = ["income", "balance"]
K = 3
RANDOM_STATE = 42
N_INIT = 10
DPI = 150
COLORS3 = ["#3B6FB6", "#E08A3C", "#5BA57A"]
COLORS4 = ["#3B6FB6", "#E08A3C", "#5BA57A", "#B05A9E"]
NOTE_DEMO = "演示数据四簇扩展；Word 指定的 bank_customers.csv 未提供"


# ---------------------------------------------------------------- 通用画像
def round2(x):
    """四舍五入到两位，避免 57.415 → 57.41 的二进制截断。"""
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def level_name(ratio):
    if ratio < 0.6:
        return "低"
    if ratio < 1.5:
        return "中"
    return "高"


def age_name(age):
    if age < 30:
        return "青年"
    if age < 45:
        return "中年"
    return "熟龄"


def profile(df, labels, balance_col="balance"):
    """按平均收入升序返回画像表，附簇名与显示序号。"""
    work = df.copy()
    work["簇"] = labels
    grouped = work.groupby("簇").agg(
        人数=("customer_id", "size"),
        均龄=("age", "mean"),
        平均收入=("income", "mean"),
        平均余额=(balance_col, "mean"),
        平均产品数=("product_count", "mean"),
    )
    grouped["占比"] = grouped["人数"] / len(work)
    grouped = grouped.sort_values("平均收入")
    grouped["簇名"] = ""
    grouped["显示序"] = range(1, len(grouped) + 1)
    inc_mean, bal_mean = work["income"].mean(), work[balance_col].mean()
    for cid, row in grouped.iterrows():
        grouped.loc[cid, "簇名"] = (
            age_name(row["均龄"])
            + level_name(row["平均收入"] / inc_mean) + "收"
            + level_name(row["平均余额"] / bal_mean) + "储型"
        )
    return grouped, work


# ================================================================ 一、质心迭代
def run_iteration(Z, scaler):
    """k-means++ 初始化后，纯 numpy 逐轮分配/更新，记录每轮 centers 与 SSE。"""
    centers, _ = kmeans_plusplus(Z, n_clusters=K, random_state=RANDOM_STATE)
    history = []
    prev_labels = None
    for r in range(100):
        dist = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        sse = float(dist.min(axis=1).sum())
        new_centers = np.vstack([Z[labels == j].mean(axis=0) for j in range(K)])
        move = float(np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max())
        stable = prev_labels is not None and bool((labels == prev_labels).all())
        history.append({"round": r, "labels": labels, "centers": centers.copy(),
                        "sse": sse, "move": move, "stable": stable})
        if stable:
            break
        prev_labels, centers = labels, new_centers

    sse_list = [h["sse"] for h in history]
    for before, after in zip(sse_list[:-1], sse_list[1:]):
        assert after <= before + 1e-9, f"SSE 出现上升：{before} -> {after}"
    print(f"SSE 单调不增校验通过：{['%.2f' % s for s in sse_list]}")

    # 每轮簇编号按质心收入排序，保证四幅图配色/命名一致
    rows = []
    for h in history:
        order = np.argsort(h["centers"][:, 0])
        back = scaler.inverse_transform(h["centers"])
        for rank, cid in enumerate(order, start=1):
            rows.append({
                "轮次": h["round"],
                "簇": f"簇{rank}",
                "簇人数": int((h["labels"] == cid).sum()),
                "质心收入(z)": round(float(h["centers"][cid, 0]), 4),
                "质心余额(z)": round(float(h["centers"][cid, 1]), 4),
                "质心收入(万元)": round(float(back[cid, 0]), 2),
                "质心余额(万元)": round(float(back[cid, 1]), 2),
                "整体SSE": round(h["sse"], 4),
                "质心最大移动量": round(h["move"], 6),
                "是否收敛": "是" if h["stable"] else "否",
            })
    table = pd.DataFrame(rows)
    path = OUT_DIR / "iteration.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已输出：{path}")
    print(table.to_string(index=False))

    print(f"收敛轮数：第 {history[-1]['round']} 轮标签不再变化（共 {history[-1]['round']} 次更新），"
          f"最终 SSE={history[-1]['sse']:.2f}")
    return history


def plot_iteration(history):
    picks = sorted({0, 1, 2, len(history) - 1})
    picks = [p for p in picks if p < len(history)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=DPI)
    axes = axes.ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    for ax, t in zip(axes, picks):
        h = history[t]
        order = np.argsort(h["centers"][:, 0])
        for rank, cid in enumerate(order, start=1):
            m = h["labels"] == cid
            ax.scatter(Z_PLOT[m, 0], Z_PLOT[m, 1], s=32, c=COLORS3[rank - 1],
                       alpha=0.55, edgecolors="white", linewidths=0.4)
        for rank, cid in enumerate(order, start=1):
            ax.scatter(h["centers"][cid, 0], h["centers"][cid, 1], marker="*",
                       s=460, c=COLORS3[rank - 1], edgecolors="black",
                       linewidths=1.1, zorder=5)
        tag = "（初始化后）" if t == 0 else ("（收敛）" if h["stable"] else "")
        ax.set_title(f"第 {t} 轮{tag}｜SSE={h['sse']:.2f}｜质心最大移动={h['move']:.3f}",
                     fontsize=10.5)
        ax.set_xlabel("收入 z-score")
        ax.set_ylabel("余额 z-score")
        ax.grid(alpha=0.25, linestyle="--")
        ax.text(0.02, 0.97,
                " ".join(f"簇{r}:{int((h['labels'] == c).sum())}人"
                         for r, c in enumerate(order, start=1)),
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS3[i], markersize=8,
                      label=f"簇{i + 1}（按质心收入升序）") for i in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("讲义迭代：k-means++ 初始化（seed=42）→ 最近质心分配 → 均值更新", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = OUT_DIR / "06_iteration.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{path}")


# ================================================================ 二、自主实践
def kmeans_default(X, k=K, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, random_state=seed, n_init=N_INIT).fit(X)


def practice_age(df):
    print("\n===== 自主实践 (a)：加入 age（三特征均标准化，k=3）=====")
    scaler2 = StandardScaler()
    z2 = scaler2.fit_transform(df[FEATURES].to_numpy())
    model2 = kmeans_default(z2)
    labels2 = model2.labels_
    grouped2, _ = profile(df, labels2)

    scaler3 = StandardScaler()
    z3 = scaler3.fit_transform(df[["income", "balance", "age"]].to_numpy())
    model3 = kmeans_default(z3)
    labels3 = model3.labels_
    grouped3, work3 = profile(df, labels3)

    ari = adjusted_rand_score(labels2, labels3)
    print(f"两特征 k=3 人数（收入升序）：{list(grouped2['人数'])}")
    print(f"三特征 k=3 人数（收入升序）：{list(grouped3['人数'])}")
    print(f"两特征 vs 三特征 标签 ARI = {ari:.4f}")

    meanings = {
        1: "加入年龄后，更多“年轻但余额略高”的客户被吸进本簇，人数 {n}（两特征 {n2}），"
           "均值余额由 {b2:.1f} 万元升到 {b:.1f} 万元；年龄成为青年客群的主要分层变量。"
           "业务上仍按低成本线上渠道批量运营。",
        2: "人数 {n}（两特征 {n2}），年龄集中到 {a:.1f} 岁，部分 40+ 客户因年龄维度被推向上层；"
           "仍是理财/分期/保障交叉销售的主力盘。",
        3: "人数 {n}（两特征 {n2}），几乎不受年龄维度影响；余额 {b:.1f} 万元、年龄 {a:.1f} 岁，"
           "维持专属客户经理与资产配置服务。",
    }
    rows = []
    for (cid, row), (cid2, row2) in zip(grouped3.iterrows(), grouped2.iterrows()):
        order = int(row["显示序"])
        rows.append({
            "簇序号": f"簇{order}",
            "簇名": row["簇名"],
            "人数": int(row["人数"]),
            "两特征人数": int(row2["人数"]),
            "占比": f"{row['占比'] * 100:.1f}%",
            "均龄": round(float(row["均龄"]), 2),
            "两特征均龄": round(float(row2["均龄"]), 2),
            "平均收入(万元)": round(float(row["平均收入"]), 2),
            "平均余额(万元)": round2(row["平均余额"]),
            "平均产品数": round(float(row["平均产品数"]), 2),
            "分群业务含义": meanings[order].format(
                n=int(row["人数"]), n2=int(row2["人数"]), b=float(row["平均余额"]),
                b2=float(row2["平均余额"]), a=float(row["均龄"])),
        })
    path = OUT_DIR / "profile_with_age.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已输出：{path}")
    show = pd.DataFrame(rows).drop(columns=["分群业务含义"])
    print(show.to_string(index=False))

    print("\n对比（两特征 → 三特征）：")
    for (cid, row), (cid2, row2) in zip(grouped3.iterrows(), grouped2.iterrows()):
        print(f"  簇{int(row['显示序'])}：人数 {int(row2['人数'])} → {int(row['人数'])}，"
              f"均龄 {row2['均龄']:.2f} → {row['均龄']:.2f}，"
              f"均余额 {row2['平均余额']:.2f} → {row['平均余额']:.2f} 万元")
    print(f"ARI={ari:.4f}：两套标签高度一致但非等同，年龄维度主要影响“青年/中年”边界。")
    return ari


def practice_stability(df):
    print("\n===== 自主实践 (b)：种子稳定性（两特征、标准化、k=3）=====")
    z = StandardScaler().fit_transform(df[FEATURES].to_numpy())
    base_labels = kmeans_default(z, seed=RANDOM_STATE).labels_
    rows = []
    for seed in (0, 7, 2026):
        labels = kmeans_default(z, seed=seed).labels_
        ari = adjusted_rand_score(base_labels, labels)
        grouped, _ = profile(df, labels)
        print(f"\n种子 {seed}：ARI(vs 42)={ari:.4f}，"
              f"人数={[int(x) for x in grouped['人数']]}")
        for cid, row in grouped.iterrows():
            rows.append({
                "种子": seed,
                "ARI(对种子42)": round(float(ari), 4),
                "簇序号": f"簇{int(row['显示序'])}",
                "人数": int(row["人数"]),
                "占比": f"{row['占比'] * 100:.1f}%",
                "均龄": round(float(row["均龄"]), 2),
                "平均收入(万元)": round(float(row["平均收入"]), 2),
                "平均余额(万元)": round2(row["平均余额"]),
                "平均产品数": round(float(row["平均产品数"]), 2),
            })
    path = OUT_DIR / "stability.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n已输出：{path}")
    print(pd.DataFrame(rows).to_string(index=False))
    return path


def practice_k4(df):
    print("\n===== 自主实践 (c)：两特征 k=4 扩展 =====")
    print(f"注明：{NOTE_DEMO}")
    z = StandardScaler().fit_transform(df[FEATURES].to_numpy())
    model = kmeans_default(z, k=4)
    df["cluster"] = model.labels_          # 回填内存 DataFrame
    g = df.groupby("cluster").agg(
        人数=("customer_id", "size"),
        均龄=("age", "mean"),
        平均收入=("income", "mean"),
        平均余额=("balance", "mean"),
        平均产品数=("product_count", "mean"),
    )
    g["占比"] = g["人数"] / len(df)
    g = g.sort_values("平均收入")

    # 独立校验：用原表 balance（万元）按标签 groupby，确认没有重复换算
    check = df.groupby("cluster")["balance"].mean().reindex(g.index)
    expected = np.array([1.45, 12.46, 57.42, 37.44])
    assert np.allclose(check.to_numpy(), expected, atol=0.02), \
        f"k=4 平均余额校验失败：{check.round(2).tolist()}"
    print("原表按标签 groupby 校验四簇平均余额（万元）："
          f"{[round2(v) for v in check]}，与预期 {expected.tolist()} 均在 0.02 万元容差内")

    # 四簇业务命名：按实际画像（收入最高但余额较低者 → 高薪低沉淀）
    names, suggestions = {}, {}
    low = g.iloc[0]
    names[g.index[0]] = f"{age_name(low['均龄'])}低额型"
    suggestions[g.index[0]] = (
        "线上开卡礼 + 小额消费贷 + 零钱理财自动转入，按极低成本批量运营，不投人力；"
        "重点抓代发工资与初职场景，做长期培育。")
    mid = g.iloc[1]
    names[g.index[1]] = "中坚主力型"
    suggestions[g.index[1]] = (
        f"银行基本盘（收入 {mid['平均收入']:.1f} 万、余额 {mid['平均余额']:.1f} 万、"
        f"产品 {mid['平均产品数']:.2f} 个）：以基金定投 + 信用卡分期 + 家庭保障组合交叉销售，"
        "目标把产品数提升到 3 个；围绕代发工资日做活动。")
    highs = g.iloc[2:]
    rich = highs["平均余额"].idxmax()
    lean = highs.drop(index=rich).index[0]
    names[rich] = "高储稳健型"
    suggestions[rich] = (
        f"余额最高（均值 {g.loc[rich, '平均余额']:.1f} 万元）、年龄 {g.loc[rich, '均龄']:.1f} 岁："
        "配专属客户经理，主推大额存单/稳健理财到期承接与私行体验，以防流失为第一 KPI。")
    names[lean] = "高薪低沉淀型"
    suggestions[lean] = (
        f"收入最高（{g.loc[lean, '平均收入']:.1f} 万元）但余额仅 {g.loc[lean, '平均余额']:.1f} 万元、"
        f"产品 {g.loc[lean, '平均产品数']:.2f} 个：主推代发工资、自动理财与高端卡/出行权益，"
        "把高流水转成留存资产。")

    rows = []
    for order, (cid, row) in enumerate(g.iterrows(), start=1):
        rows.append({
            "簇序号": f"簇{order}",
            "簇名": names[cid],
            "人数": int(row["人数"]),
            "占比": f"{row['占比'] * 100:.1f}%",
            "均龄": round(float(row["均龄"]), 2),
            "平均收入(万元)": round(float(row["平均收入"]), 2),
            "平均余额(万元)": round2(row["平均余额"]),
            "平均产品数": round(float(row["平均产品数"]), 2),
            "营销建议": suggestions[cid],
            "数据说明": NOTE_DEMO,
        })
    path = OUT_DIR / "profile_k4.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已输出：{path}")
    print(pd.DataFrame(rows).drop(columns=["营销建议", "数据说明"]).to_string(index=False))

    # 07 散点
    order_map = {cid: i for i, cid in enumerate(g.index, start=1)}
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    for cid in g.index:
        o = order_map[cid]
        m = df["cluster"] == cid
        ax.scatter(df.loc[m, "income"], df.loc[m, "balance"], s=40, c=COLORS4[o - 1],
                   alpha=0.6, edgecolors="white", linewidths=0.4,
                   label=f"簇{o} {names[cid]}  n={int(g.loc[cid, '人数'])}"
                         f"（{g.loc[cid, '占比'] * 100:.1f}%）")
        ax.scatter(g.loc[cid, "平均收入"], g.loc[cid, "平均余额"], marker="*", s=520,
                   c=COLORS4[o - 1], edgecolors="black", linewidths=1.1, zorder=5)
    ax.set_xlabel("收入（万元）")
    ax.set_ylabel("余额（万元）")
    ax.set_title("演示数据四簇扩展：两标准化特征 k=4 的收入—余额分布\n"
                 "星形为簇质心；Word 指定的 bank_customers.csv 未提供", fontsize=11.5)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    fig.tight_layout()
    p7 = OUT_DIR / "07_scatter_k4.png"
    fig.savefig(p7, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{p7}")

    # 08 画像 z 分数分组柱状图
    metrics = ["均龄", "平均收入", "平均余额", "平均产品数"]
    labels_cn = ["均龄", "平均收入", "平均余额", "平均产品数"]
    zscores = []
    for col, col_cn in zip(["age", "income", "balance", "product_count"], metrics):
        mean_all, std_all = df[col].mean(), df[col].std()
        zscores.append([(g.loc[cid, col_cn] - mean_all) / std_all for cid in g.index])
    zscores = np.array(zscores)          # 4 指标 × 4 簇

    fig, ax = plt.subplots(figsize=(11, 7), dpi=DPI)
    x = np.arange(len(metrics))
    width = 0.19
    for i, cid in enumerate(g.index):
        ax.bar(x + (i - 1.5) * width, zscores[:, i], width,
               color=COLORS4[i], label=f"簇{i + 1} {names[cid]}", alpha=0.9)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_cn)
    ax.set_ylabel("z 分数（相对全体均值，按全体标准差）")
    ax.set_title("四簇画像对比：各列均值按全体标准差转 z 分数\n"
                 "演示数据 k=4 扩展；bank_customers.csv 未提供", fontsize=11.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(fontsize=9, frameon=True, ncol=2)
    fig.tight_layout()
    p8 = OUT_DIR / "08_profiles_k4.png"
    fig.savefig(p8, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{p8}")

    del df["cluster"]   # 不改源文件，仅内存使用
    return g


# ================================================================ 三、同心圆研究
def practice_circles():
    print("\n===== 三、独立研究：同心圆 KMeans vs DBSCAN =====")
    X, y = make_circles(n_samples=600, factor=0.4, noise=0.05, random_state=42)
    z = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=N_INIT).fit(z)
    ari_km = adjusted_rand_score(y, km.labels_)

    eps, min_samples = 0.2, 5
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(z)
    ari_db = adjusted_rand_score(y, db.labels_)
    noise_n = int((db.labels_ == -1).sum())
    n_clusters = len([c for c in set(db.labels_) if c != -1])

    print(f"KMeans(k=2)：ARI={ari_km:.4f}")
    print(f"DBSCAN(eps={eps}, min_samples={min_samples})：ARI={ari_db:.4f}，"
          f"识别簇数={n_clusters}，噪声点数={noise_n}")

    metrics = pd.DataFrame([
        {"算法": "KMeans", "ARI": round(float(ari_km), 4), "识别簇数": 2,
         "噪声点数": 0, "参数": f"k=2, random_state={RANDOM_STATE}, n_init={N_INIT}"},
        {"算法": "DBSCAN", "ARI": round(float(ari_db), 4), "识别簇数": n_clusters,
         "噪声点数": noise_n, "参数": f"eps={eps}, min_samples={min_samples}"},
    ])
    path_csv = OUT_DIR / "circles_metrics.csv"
    metrics.to_csv(path_csv, index=False, encoding="utf-8-sig")
    print(f"已输出：{path_csv}")
    print(metrics.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), dpi=DPI)
    axes[0].scatter(z[y == 0, 0], z[y == 0, 1], s=18, c="#3B6FB6", alpha=0.7, label="外环（真值 0）")
    axes[0].scatter(z[y == 1, 0], z[y == 1, 1], s=18, c="#E08A3C", alpha=0.7, label="内环（真值 1）")
    axes[0].set_title("真值：两个同心圆")
    axes[0].legend(fontsize=9, loc="upper right")

    axes[1].scatter(z[:, 0], z[:, 1], s=18, c=np.where(km.labels_ == 0, "#3B6FB6", "#E08A3C"),
                    alpha=0.7)
    axes[1].set_title(f"KMeans(k=2)｜ARI={ari_km:.4f}\n线性边界切不开同心圆", fontsize=10.5)

    noise = db.labels_ == -1
    palette = {0: "#5BA57A", 1: "#B05A9E"}
    for c in sorted(set(db.labels_)):
        if c == -1:
            continue
        m = db.labels_ == c
        axes[2].scatter(z[m, 0], z[m, 1], s=18, c=palette[c % 2], alpha=0.7,
                        label=f"簇 {c}（n={int(m.sum())}）")
    axes[2].scatter(z[noise, 0], z[noise, 1], s=26, c="#888888", marker="x",
                    label=f"噪声（{noise_n} 点）")
    axes[2].set_title(f"DBSCAN(eps={eps}, min_samples={min_samples})｜ARI={ari_db:.4f}",
                      fontsize=10.5)
    axes[2].legend(fontsize=8.5, loc="upper right")

    for ax in axes:
        ax.set_xlabel("x（标准化）")
        ax.set_ylabel("y（标准化）")
        ax.grid(alpha=0.2, linestyle="--")
    fig.suptitle("玩具数据研究：KMeans 与 DBSCAN 对同心圆的可分性（随机种子 42，n=600）", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "09_circles_compare.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{path}")
    return ari_km, ari_db


Z_PLOT = None


def main():
    global Z_PLOT
    print(f"使用中文字体：{FONT_USED}")
    df = pd.read_csv(SIM_CSV, encoding="utf-8-sig")   # 只读原始模拟数据

    scaler = StandardScaler()
    Z_PLOT = scaler.fit_transform(df[FEATURES].to_numpy())

    print("\n===== 一、讲义迭代（k-means++，seed=42，k=3）=====")
    history = run_iteration(Z_PLOT, scaler)
    plot_iteration(history)

    practice_age(df)
    practice_stability(df)
    practice_k4(df)
    practice_circles()

    print("\n全部完成：新增 06/07/08/09 图与 iteration/profile_with_age/stability/profile_k4 四个 CSV；"
          "前两轮结果与源 CSV 均未改动。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Lesson 09 修正 A + B：元版先标准化再聚类（k=3），并在标准化空间扫 k=1..9。

修正 A：data/mall_customers_yuan.csv 的 income（万元）与 balance（元）量纲差 1e4，
        直接进 KMeans 会让欧氏距离被余额支配（见 01_baseline.py 对照）。
        本脚本对两列做 StandardScaler.fit_transform，再 KMeans(k=3, random_state=42, n_init=10)。
修正 B：同一标准化特征空间扫 k=1..9，输出 SSE / 轮廓系数，标注肘点与轮廓最高点。

运行：../../work/lesson09-env/bin/python generated/02_scaled.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 路径与字体
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
YUAN_CSV = DATA_DIR / "mall_customers_yuan.csv"
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
K_RANGE = range(1, 10)
PALETTE = ["#3B6FB6", "#E08A3C", "#5BA57A"]
DPI = 150

# 基于画像的业务建议模板（数值由画像实时填，避免写死）
SUGGESTIONS = {
    1: "低额青年客群：以手机银行/线上渠道低成本维护，主推首卡、消费信贷与零钱理财，"
       "控制人力成本，关注代发与初职场景的长期培养。",
    2: "中坚主力客群：银行基本盘，重点做理财、基金定投、信用卡分期与家庭保障的交叉销售，"
       "提升产品持有数（当前 {prod:.2f} 个）与 AUM 留存。",
    3: "高余额客群：余额集中（均值 {bal:.1f} 万元）但产品数仅 {prod:.2f}，需专属客户经理，"
       "配置大额存单/资产配置/私行服务，优先防流失。",
}


def fmt_thousand(x, _pos):
    return f"{x:,.0f}"


# ---------------------------------------------------------------- 建模
def fit_scaled(df):
    """标准化 -> KMeans(k=3, n_init=10)，返回 scaler / model / z 特征。"""
    scaler = StandardScaler()
    z = scaler.fit_transform(df[FEATURES].to_numpy())
    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT).fit(z)
    return scaler, model, z


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


def cluster_names(df, labels, balance_col="balance"):
    """按平均收入从低到高排序，依据真实画像生成簇名与显示序号。"""
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
    inc_mean = work["income"].mean()
    bal_mean = work[balance_col].mean()
    for cid, row in grouped.iterrows():
        grouped.loc[cid, "簇名"] = (
            age_name(row["均龄"])
            + level_name(row["平均收入"] / inc_mean) + "收"
            + level_name(row["平均余额"] / bal_mean) + "储型"
        )
    return grouped, work


# ---------------------------------------------------------------- 修正 A：画像与对照图
def build_profile(df, grouped, labels):
    """输出 profile_k3.csv：余额与收入统一换算成万元。"""
    rows = []
    for cid, row in grouped.iterrows():
        order = int(row["显示序"])
        rows.append({
            "簇序号": f"簇{order}",
            "簇名": row["簇名"],
            "人数": int(row["人数"]),
            "占比": f"{row['占比'] * 100:.1f}%",
            "均龄": round(float(row["均龄"]), 2),
            "平均收入(万元)": round(float(row["平均收入"]), 2),
            "平均余额(万元)": round(float(row["平均余额"]) / 1e4, 2),
            "平均产品数": round(float(row["平均产品数"]), 2),
            "业务建议": SUGGESTIONS[order].format(
                prod=float(row["平均产品数"]), bal=float(row["平均余额"]) / 1e4),
        })
    profile = pd.DataFrame(rows)
    path = OUT_DIR / "profile_k3.csv"
    profile.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n已输出：{path}")
    print(profile.drop(columns=["业务建议"]).to_string(index=False))
    return profile


def plot_scale_compare(df, labels, scaler, model, grouped, z):
    order_of = {cid: int(row["显示序"]) for cid, row in grouped.iterrows()}
    back = scaler.inverse_transform(model.cluster_centers_)   # z 质心还原成“万元 / 元”

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=DPI)

    for ax, x, y, centers, xlabel, ylabel, unit, title in (
        (axes[0], df["income"], df["balance"], back,
         "收入（万元）", "余额（元）", "元",
         "左：原始坐标（余额以元计，未标准化）"),
        (axes[1], z[:, 0], z[:, 1], model.cluster_centers_,
         "收入 z-score", "余额 z-score", "z", "右：z-score 标准化坐标"),
    ):
        for cid in grouped.index:
            order = order_of[cid]
            mask = labels == cid
            ax.scatter(x[mask], y[mask], s=42, c=PALETTE[order - 1], alpha=0.6,
                       edgecolors="white", linewidths=0.5,
                       label=f"簇{order} {grouped.loc[cid, '簇名']}  n={int(grouped.loc[cid, '人数'])}")
        for cid, (cx, cy) in zip(grouped.index, centers):
            order = order_of[cid]
            ax.scatter(cx, cy, marker="*", s=540, c=PALETTE[order - 1],
                       edgecolors="black", linewidths=1.2, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, linestyle="--")
        if unit == "元":
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_thousand))
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle("修正 A 对照：同一批客户、同一组标签，左原始坐标 / 右标准化坐标（星形为质心）",
                 fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "04_compare_scale.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{path}")


# ---------------------------------------------------------------- 修正 B：扫 k
def find_elbow(ks, sse):
    """肘点规则：SSE 取对数后与 k 一起归一化到 [0,1]，取离首尾连线最远的 k。"""
    x = np.asarray(list(ks), dtype=float)
    y = np.log(np.asarray(list(sse), dtype=float))
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    dist = np.abs(yn - (1 - xn)) / np.sqrt(2)
    return int(x[int(np.argmax(dist))])


def sweep_k(z):
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(z)
        sil = silhouette_score(z, model.labels_) if k >= 2 else np.nan
        rows.append({"k": k, "SSE": float(model.inertia_), "轮廓系数": float(sil)})
    result = pd.DataFrame(rows)
    elbow = find_elbow(result["k"], result["SSE"])
    best_k = int(result.loc[result["轮廓系数"].idxmax(), "k"])
    result["说明"] = ""
    result.loc[result["k"] == elbow, "说明"] += "肘点"
    result.loc[result["k"] == best_k, "说明"] += "轮廓最高点"
    result["说明"] = result["说明"].str.strip()

    path = OUT_DIR / "choose_k.csv"
    result.assign(SSE=result["SSE"].round(2),
                  轮廓系数=result["轮廓系数"].round(4)).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n已输出：{path}")
    show = result.copy()
    show["SSE"] = show["SSE"].round(2)
    show["轮廓系数"] = show["轮廓系数"].round(4)
    show.loc[show["k"] == 1, "轮廓系数"] = np.nan
    print(show.to_string(index=False, na_rep="—"))
    return result, elbow, best_k


def plot_choose_k(result, elbow, best_k):
    ks = result["k"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=DPI)

    ax = axes[0]
    ax.plot(ks, result["SSE"], marker="o", color="#3B6FB6", linewidth=2)
    e_sse = float(result.loc[result["k"] == elbow, "SSE"].iloc[0])
    ax.scatter([elbow], [e_sse], marker="o", s=140, facecolors="none",
               edgecolors="#C0392B", linewidths=2.2, zorder=5)
    ax.annotate(f"肘点 k={elbow}\nSSE={e_sse:,.1f}", (elbow, e_sse),
                textcoords="offset points", xytext=(28, 32), fontsize=10, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C0392B", alpha=0.9))
    ax.set_xlabel("k（簇数）")
    ax.set_ylabel("SSE（簇内平方和）")
    ax.set_title("SSE 随 k 变化：找肘点", fontsize=11)
    ax.set_xticks(list(ks))
    ax.margins(y=0.12)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[1]
    sub = result[result["k"] >= 2]
    ax.plot(sub["k"], sub["轮廓系数"], marker="s", color="#5BA57A", linewidth=2)
    b_sil = float(result.loc[result["k"] == best_k, "轮廓系数"].iloc[0])
    ax.scatter([best_k], [b_sil], marker="s", s=140, facecolors="none",
               edgecolors="#C0392B", linewidths=2.2, zorder=5)
    ax.annotate(f"轮廓最高 k={best_k}\n{b_sil:.4f}", (best_k, b_sil),
                textcoords="offset points", xytext=(38, -105), fontsize=10, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C0392B", alpha=0.9))
    ax.set_xlabel("k（簇数）")
    ax.set_ylabel("平均轮廓系数")
    ax.set_title("轮廓系数随 k 变化（k≥2）", fontsize=11)
    ax.set_xticks(list(sub["k"]))
    ax.margins(y=0.15)
    ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("修正 B：标准化特征空间扫 k=1..9（random_state=42，n_init=10）", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "05_choose_k.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{path}")


def describe_k(df, z, k):
    """k=2/3/4 的实际读数，用于选 k 讨论。"""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(z)
    sil = silhouette_score(z, model.labels_)
    work = df.copy()
    work["簇"] = model.labels_
    g = work.groupby("簇").agg(人数=("customer_id", "size"),
                               均龄=("age", "mean"),
                               平均收入=("income", "mean"),
                               平均余额=("balance", "mean"),
                               平均产品数=("product_count", "mean"))
    g = g.sort_values("平均收入")
    g["占比"] = g["人数"] / len(work)
    return model.inertia_, sil, g


def main():
    print(f"使用中文字体：{FONT_USED}")
    df = pd.read_csv(YUAN_CSV, encoding="utf-8-sig")   # 只读，不改动原始字节

    print("\n===== 量纲检查（元版原始文件）=====")
    inc_std, bal_std = df["income"].std(), df["balance"].std()
    print(f"income：均值 {df['income'].mean():.2f} 万元，标准差 {inc_std:.2f}，最大 {df['income'].max():.2f}")
    print(f"balance：均值 {df['balance'].mean():,.2f} 元，标准差 {bal_std:,.2f}，最大 {df['balance'].max():,.0f}")
    print(f"标准差之比（余额/收入）≈ {bal_std / inc_std:,.0f} 倍 → 不标准化时距离几乎只看余额")

    scaler, model, z = fit_scaled(df)
    print(f"\n模型：StandardScaler + KMeans(k={K}, random_state={RANDOM_STATE}, n_init={N_INIT})")
    print(f"标准化后：均值 {z.mean(axis=0).round(6)}，标准差 {z.std(axis=0).round(6)}")

    labels = model.labels_
    grouped, _ = cluster_names(df, labels)
    print("\n===== 修正 A：标准化后 k=3 画像（按平均收入从低到高）=====")
    profile = build_profile(df, grouped, labels)
    for _, row in profile.iterrows():
        print(f"{row['簇序号']}（{row['簇名']}）：{row['业务建议']}")
    print(f"inertia（标准化空间）= {model.inertia_:.2f}")

    plot_scale_compare(df, labels, scaler, model, grouped, z)

    print("\n===== 修正 B：扫 k =====")
    result, elbow, best_k = sweep_k(z)
    plot_choose_k(result, elbow, best_k)

    print("\n===== k=2/3/4 实际读数对照 =====")
    readings = {}
    for k in (2, 3, 4):
        sse, sil, g = describe_k(df, z, k)
        readings[k] = (sse, sil, g)
        sizes = " / ".join(f"{int(r['人数'])}" for _, r in g.iterrows())
        print(f"\nk={k}：SSE={sse:.2f}，轮廓系数={sil:.4f}，人数（收入升序）= {sizes}")
        view = g.copy()
        view["占比"] = (view["占比"] * 100).round(1).astype(str) + "%"
        view["平均余额"] = (view["平均余额"] / 1e4).round(2)
        print(view.rename(columns={"平均余额": "平均余额(万元)",
                                   "平均收入": "平均收入(万元)"}).round(2).to_string())

    s2, sil2, g2 = readings[2]
    s3, sil3, g3 = readings[3]
    s4, sil4, g4 = readings[4]
    low2, high2 = g2.iloc[0], g2.iloc[-1]
    main3 = g3.iloc[1]
    hi_a, hi_b = g4.iloc[-2], g4.iloc[-1]
    print("\n===== 选 k 结论（需求：分开主力和高余额客户）=====")
    print(f"k={best_k} 轮廓最高（{sil2:.4f}），但只切出「高余额 {int(high2['人数'])} 人」与「其余 "
          f"{int(low2['人数'])} 人」，主力（收入≈{main3['平均收入']:.1f} 万、"
          f"余额≈{main3['平均余额'] / 1e4:.1f} 万）被并进低额群，无法单独运营。")
    print(f"k={elbow} 为肘点，SSE 由 {s2:.2f} 降到 {s3:.2f}（−{(s2 - s3) / s2 * 100:.0f}%），"
          f"轮廓 {sil3:.4f} 仅比 k={best_k} 低 {sil2 - sil3:.3f}，"
          f"分出 {' / '.join(str(int(r['人数'])) for _, r in g3.iterrows())} 三层，满足业务要求。")
    print(f"k=4 把高收群再拆成 {int(hi_a['人数'])} 人（收入≈{hi_a['平均收入']:.1f} 万、"
          f"余额≈{hi_a['平均余额'] / 1e4:.1f} 万）与 {int(hi_b['人数'])} 人"
          f"（收入≈{hi_b['平均收入']:.1f} 万、余额≈{hi_b['平均余额'] / 1e4:.1f} 万），"
          f"轮廓降至 {sil4:.4f}，两簇各占 {hi_a['人数'] / len(df) * 100:.0f}%，"
          f"样本薄且把高余额内部切开，属于过度细分。")
    print("结论：本轮选 k=3——肘点 + 低额/主力/高余额三层画像清晰，轮廓仅小幅让位于 k=2；"
          "若后续只需识别高余额客户，k=2 可退化为单条监控口径。")
    print("\n已核对：源 CSV 只读，未做任何写回。")


if __name__ == "__main__":
    main()

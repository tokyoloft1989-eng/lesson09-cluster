# -*- coding: utf-8 -*-
"""Lesson 09 起点版：未标准化 + 单次初始化的 KMeans(k=3) 基线，以及“余额换成元”的量纲翻车对照。

只做本轮：初检 -> 原始散点 -> 万元基线聚类 -> 元版同设置聚类。
标准化与选 k 留待验收后进行。

运行：../../work/lesson09-env/bin/python generated/01_baseline.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans

# ---------------------------------------------------------------- 路径与字体
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SIM_CSV = DATA_DIR / "mall_customers_sim.csv"
YUAN_CSV = DATA_DIR / "mall_customers_yuan.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = ["Heiti SC", "PingFang SC", "STHeiti", "Songti SC", "Arial Unicode MS"]


def setup_chinese_font():
    """在候选字体里挑本机真实可用的一款，避免中文变方框。"""
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

# ---------------------------------------------------------------- 模型与绘图常量
FEATURES = ["income", "balance"]      # 进模型的只有这两列
K = 3
RANDOM_STATE = 42
N_INIT = 1
PALETTE = ["#3B6FB6", "#E08A3C", "#5BA57A"]
FIG_SIZE = (10, 7)
DPI = 150

metrics_rows = []   # (区块, 项目, 子项, 值)


def add_metric(section, item, sub, value):
    metrics_rows.append({"区块": section, "项目": item, "子项": sub, "值": value})


def build_model():
    """起点版设置：不标准化、单次初始化。元版必须复用同一函数，保证设置完全一致。"""
    return KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)


# ---------------------------------------------------------------- 数据初检
def inspect(df, tag):
    print(f"\n===== 数据初检：{tag} =====")
    print(f"行数 x 列数：{df.shape[0]} x {df.shape[1]}")
    print("字段与类型：")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    dup = int(df["customer_id"].duplicated().sum())
    print(f"重复客户编号：{dup}")

    na = df.isna().sum()
    print("缺失值：")
    for col, cnt in na.items():
        print(f"  - {col}: {int(cnt)}")

    add_metric("数据初检", tag, "行数", df.shape[0])
    add_metric("数据初检", tag, "字段数", df.shape[1])
    add_metric("数据初检", tag, "重复客户编号数", dup)
    for col, cnt in na.items():
        add_metric("数据初检", f"{tag}-缺失", col, int(cnt))

    desc = df[["age", "income", "balance", "product_count"]].describe().T
    print("描述统计：")
    print(desc[["mean", "std", "min", "50%", "max"]].round(2).to_string())
    for col, row in desc.iterrows():
        for stat, key in (("均值", "mean"), ("标准差", "std"), ("最小值", "min"),
                          ("中位数", "50%"), ("最大值", "max")):
            add_metric("数据初检", f"{tag}-统计", f"{col}-{stat}", round(float(row[key]), 4))
    return desc


# ---------------------------------------------------------------- 画像与命名
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


def profile_table(df, labels, balance_col):
    work = df.copy()
    work["簇"] = labels
    grouped = work.groupby("簇").agg(
        人数=("customer_id", "size"),
        均龄=("age", "mean"),
        均收入=("income", "mean"),
        均余额=(balance_col, "mean"),
        平均产品数=("product_count", "mean"),
    )
    grouped["占比"] = grouped["人数"] / len(work)
    grouped = grouped.sort_values("均收入")          # 按平均收入从低到高
    grouped.insert(0, "名称", "")
    overall_income = work["income"].mean()
    overall_balance = work[balance_col].mean()
    for i, (cid, row) in enumerate(grouped.iterrows(), start=1):
        grouped.loc[cid, "名称"] = (
            age_name(row["均龄"])
            + level_name(row["均收入"] / overall_income) + "收"
            + level_name(row["均余额"] / overall_balance) + "储型"
        )
        grouped.loc[cid, "显示序"] = i
    grouped["显示序"] = grouped["显示序"].astype(int)
    return grouped, work


def print_profile(grouped, balance_unit):
    print("\n按平均收入从低到高：")
    view = grouped.drop(columns=["名称", "显示序"]).copy()
    view.index = [f"簇{int(r['显示序'])}（{r['名称']}）" for _, r in grouped.iterrows()]
    view["占比"] = (view["占比"] * 100).round(1).astype(str) + "%"
    for col in ("均龄", "均收入", "均余额", "平均产品数"):
        view[col] = view[col].round(2)
    print(view.rename(columns={"均余额": f"均余额({balance_unit})",
                               "均收入": "均收入(万元)"}).to_string())


# ---------------------------------------------------------------- 绘图
def fmt_thousand(x, _pos):
    return f"{x:,.0f}"


def size_from_age(age):
    span = max(age.max() - age.min(), 1)
    return 25 + (age - age.min()) / span * 180


def plot_raw(df):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    sizes = size_from_age(df["age"])
    ax.scatter(df["income"], df["balance"], s=sizes, c="#4C72B0",
               alpha=0.55, edgecolors="white", linewidths=0.6)
    ax.set_xlabel("收入（万元）")
    ax.set_ylabel("余额（万元）")
    ax.set_title("原始散点：收入—余额（点面积映射年龄）")
    ax.grid(alpha=0.25, linestyle="--")

    handles = []
    for a in (20, 40, 60):
        s = size_from_age(pd.Series([df["age"].min(), a, df["age"].max()]))[1]
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=(s ** 0.5) * 0.8,
                              markerfacecolor="#4C72B0", alpha=0.55, label=f"{a} 岁"))
    ax.legend(handles=handles, title="点面积=年龄", loc="upper left", frameon=True)

    fig.tight_layout()
    path = OUT_DIR / "01_raw_scatter.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"\n已输出：{path}")
    add_metric("输出文件", "原始散点图", "路径", str(path.relative_to(ROOT)))
    return path


def draw_clusters(ax, work, grouped, balance_col, unit, title, subtitle):
    for _, row in grouped.iterrows():
        cid, order = int(row.name), int(row["显示序"])
        mask = work["簇"] == cid
        ax.scatter(work.loc[mask, "income"], work.loc[mask, balance_col],
                   s=45, c=PALETTE[order - 1], alpha=0.6,
                   edgecolors="white", linewidths=0.5,
                   label=f"簇{order} {row['名称']}  n={int(row['人数'])}（{row['占比'] * 100:.1f}%）")
        ax.scatter(row["均收入"], row["均余额"], marker="*", s=520,
                   c=PALETTE[order - 1], edgecolors="black", linewidths=1.1, zorder=5)
        ax.annotate(f"簇{order} 中心", (row["均收入"], row["均余额"]),
                    textcoords="offset points", xytext=(16, 16), fontsize=9,
                    color=PALETTE[order - 1])

    ax.set_xlabel("收入（万元）")
    ax.set_ylabel(f"余额（{unit}）")
    ax.set_title(title + "\n" + subtitle, fontsize=12)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=8.5, frameon=True, title="簇人数与画像")


def plot_clusters(work, grouped, path_name, title, subtitle, balance_col="balance", unit="万元"):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    draw_clusters(ax, work, grouped, balance_col, unit, title, subtitle)
    if unit == "元":
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_thousand))
    fig.tight_layout()
    path = OUT_DIR / path_name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{path}")
    add_metric("输出文件", path_name, "路径", str(path.relative_to(ROOT)))
    return path


# ---------------------------------------------------------------- 主流程
def main():
    print(f"使用中文字体：{FONT_USED}")
    sim = pd.read_csv(SIM_CSV, encoding="utf-8-sig")     # 只读，不改动原始字节
    yuan = pd.read_csv(YUAN_CSV, encoding="utf-8-sig")

    inspect(sim, "万元版")
    inspect(yuan, "元版")

    add_metric("模型设置", "特征", "列名", ", ".join(FEATURES))
    add_metric("模型设置", "n_clusters", "k", K)
    add_metric("模型设置", "random_state", "值", RANDOM_STATE)
    add_metric("模型设置", "n_init", "值", N_INIT)
    add_metric("模型设置", "标准化", "方式", "不做")
    add_metric("模型设置", "元版处理", "方式", "余额保留元，不换回万元")

    print(f"\n模型设置：KMeans(k={K}, random_state={RANDOM_STATE}, n_init={N_INIT})，"
          f"特征={FEATURES}，未标准化")

    plot_raw(sim)

    # ---- 万元起点版
    model = build_model()
    labels = model.fit_predict(sim[FEATURES].to_numpy())
    grouped, work = profile_table(sim, labels, "balance")
    print("\n===== 起点版（万元）聚类结果 =====")
    print_profile(grouped, "万元")
    print(f"inertia = {model.inertia_:.2f}")
    add_metric("聚类指标", "起点版", "inertia", round(float(model.inertia_), 2))
    for _, row in grouped.iterrows():
        order = int(row["显示序"])
        tag = f"簇{order}（{row['名称']}）"
        add_metric("画像-起点版(万元)", tag, "人数", int(row["人数"]))
        add_metric("画像-起点版(万元)", tag, "占比", f"{row['占比'] * 100:.1f}%")
        add_metric("画像-起点版(万元)", tag, "均龄", round(float(row["均龄"]), 2))
        add_metric("画像-起点版(万元)", tag, "均收入(万元)", round(float(row["均收入"]), 2))
        add_metric("画像-起点版(万元)", tag, "均余额(万元)", round(float(row["均余额"]), 2))
        add_metric("画像-起点版(万元)", tag, "平均产品数", round(float(row["平均产品数"]), 2))

    plot_clusters(work, grouped, "03_clustered_scatter.png",
                  "起点版 KMeans(k=3) 聚类：未标准化，万元口径",
                  f"random_state={RANDOM_STATE}，n_init={N_INIT}；星形为各簇中心")

    # ---- 元版同设置对照
    yuan_model = build_model()
    yuan_labels = yuan_model.fit_predict(yuan[FEATURES].to_numpy())   # 余额不换回万元
    yuan_grouped, yuan_work = profile_table(yuan, yuan_labels, "balance")
    print("\n===== 元版（同设置）聚类结果 =====")
    print_profile(yuan_grouped, "元")
    print(f"inertia = {yuan_model.inertia_:.2f}")
    add_metric("聚类指标", "元版", "inertia", f"{float(yuan_model.inertia_):.2f}")

    base_counts = grouped["人数"].to_dict()
    yuan_counts = yuan_grouped["人数"].to_dict()
    print("\n各簇真实人数对照（按平均收入从低到高）：")
    diff_rows = []
    for (cid, row), (ycid, yrow) in zip(grouped.iterrows(), yuan_grouped.iterrows()):
        order = int(row["显示序"])
        base_n, yuan_n = int(row["人数"]), int(yrow["人数"])
        delta = yuan_n - base_n
        diff_rows.append((f"簇{order}", row["名称"], base_n, yuan_n, delta))
        print(f"  簇{order}（{row['名称']}）：起点 {base_n} 人 → 元版 {yuan_n} 人，差异 {delta:+d}")
        add_metric("人数差异", f"簇{order}（{row['名称']}）", "起点版人数", base_n)
        add_metric("人数差异", f"簇{order}（{row['名称']}）", "元版人数", yuan_n)
        add_metric("人数差异", f"簇{order}（{row['名称']}）", "差异(元-万元)", f"{delta:+d}")

    for _, row in yuan_grouped.iterrows():
        order = int(row["显示序"])
        tag = f"簇{order}（{row['名称']}）"
        add_metric("画像-元版(元)", tag, "人数", int(row["人数"]))
        add_metric("画像-元版(元)", tag, "占比", f"{row['占比'] * 100:.1f}%")
        add_metric("画像-元版(元)", tag, "均龄", round(float(row["均龄"]), 2))
        add_metric("画像-元版(元)", tag, "均收入(万元)", round(float(row["均收入"]), 2))
        add_metric("画像-元版(元)", tag, "均余额(元)", round(float(row["均余额"]), 2))
        add_metric("画像-元版(元)", tag, "平均产品数", round(float(row["平均产品数"]), 2))

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    draw_clusters(ax, yuan_work, yuan_grouped, "balance", "元",
                  "元版翻车对照：完全相同的模型设置，余额按元进模型",
                  "余额量级 1e4~1e5 使距离被余额主导，收入维度几乎失效")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_thousand))
    note = "人数变化（元版 − 起点版）：\n" + "\n".join(
        f"簇{o}：{b} → {y}（{d:+d}）" for o, _n, b, y, d in diff_rows)
    ax.text(0.985, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                                  edgecolor="#999999", alpha=0.9))
    fig.tight_layout()
    path = OUT_DIR / "02_yuan_fail.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"已输出：{path}")
    add_metric("输出文件", "02_yuan_fail.png", "路径", str(path.relative_to(ROOT)))

    # ---- 汇总 CSV
    csv_path = OUT_DIR / "baseline_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n已输出：{csv_path}（{len(metrics_rows)} 行）")

    print("\n一句话结论：万元口径下三簇按收入/余额同步分层；余额换成元后量纲失衡，"
          "欧氏距离被余额主导、收入几乎不起作用，人数分布随之漂移。")


if __name__ == "__main__":
    main()

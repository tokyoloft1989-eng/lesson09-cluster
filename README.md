# 第 9 次课 聚类客户分群

本次实践按课堂讲义复现一套 300 名模拟客户的 KMeans 分群。我依次比较万元起点、元计量纲陷阱和标准化结果，再用肘部法与轮廓系数解释三簇的选择。另用同一份演示数据完成四簇扩展、加入年龄的重画像、三个种子复跑和同心圆实验。

[查看进度报告](progress.md) · [选 k 双图](outputs/05_choose_k.png) · [四簇画像与建议](outputs/profile_k4.csv)

## 数据来源和口径

两份原始文件来自教师的第 9 课课堂演示包。两表是同一批 300 名模拟客户。万元版的收入和余额都以万元为单位。元版收入仍以万元为单位，只有余额改为元。两份 CSV 原样保留。学校的 Word 要求单另列了 `bank_customers.csv` 及其四个行为特征，本次四簇输出以提供的演示数据为口径。

```sh
shasum -a 256 -c data/SHA256SUMS
```

## 本地复现

在 Trae CN 打开仓库根目录。在 Python 3.12 环境安装依赖，使用本机已有的 Heiti SC、PingFang SC、Songti SC 或其他脚本支持的中文字体。

```sh
python -m pip install -r requirements.txt
python generated/01_baseline.py
python generated/02_scaled.py
python generated/03_practice.py
```

| 阶段 | 核心产物 | Git 标签 |
| --- | --- | --- |
| 起点与元计对照 | [原始散点](outputs/01_raw_scatter.png)、[起点聚类](outputs/03_clustered_scatter.png)、[元计对照](outputs/02_yuan_fail.png)、[初检及人数](outputs/baseline_metrics.csv) | `O9-step1` |
| 量纲修正与选 k | [坐标对照](outputs/04_compare_scale.png)、[选 k 双图](outputs/05_choose_k.png)、[SSE 与轮廓系数](outputs/choose_k.csv)、[三簇画像](outputs/profile_k3.csv) | `O9-step2` |
| 画像及自主实践 | [迭代](outputs/06_iteration.png)、[四簇散点](outputs/07_scatter_k4.png)、[四簇画像图](outputs/08_profiles_k4.png)、[年龄重画像](outputs/profile_with_age.csv)、[三种子](outputs/stability.csv)、[同心圆](outputs/09_circles_compare.png) | `O9-step3` |

三簇模型使用标准化的收入与余额，设置 `random_state=42` 和 `n_init=10`。万元起点与元计实验特意使用同一未经标准化的模型配置，以隔离量纲变化的影响。每簇人数、均值与业务名按簇平均收入由低到高报告。簇编号仅用于描述群组。

算法接口参考 [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) 和 [轮廓系数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) 文档。

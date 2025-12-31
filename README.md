## MTBMT（Meta-learning + Trajectory Guidance）

本仓库实现三条主线（两类能力 + 一条数据管道）：

- **算法选择（Meta Selector）**：对“特征相关性/重要性量化算法”做统一基准评测，并把 **数据元特征 +（可选）轨迹特征 + 各算法评估结果** 写入 JSONL 经验库；再用经验库训练/评估元学习选择器，重点指标是 **Top1 accuracy**。
- **算法训练指导（Trajectory Guidance / Guided-CART）**：在决策树每个节点先生成候选分裂（候选生成仍依赖 gini gain/阈值枚举），再用“倾向模型（reranker）”对候选进行 **重排/替换**，从而在“下一分裂选择”层面 **弱化** 传统 CART 的纯 gini 驱动，实现可控的“学习轨迹引导”。
- **数据管道（OpenML 批量入库）**：批量抓取 OpenML 表格数据集 → 跑基准评测 → 追加写入经验库，支持断点/去重/失败日志/重试。

### 你将得到什么

- **统一经验格式**：`experience/experience.jsonl`（JSONL，一行一条）可持续追加与复用
- **可比较量化**：对每个方法记录 CV 得分、稳定性（Jaccard）、耗时，并支持综合目标 \(J(A)=w_u\cdot Utility+w_s\cdot Stability-w_c\cdot \log(1+Cost)\)
- **元选择器评估（Top1）**：支持 dataset-wise 切分（GroupKFold），并支持 `--aggregate-by-dataset` 修复“同一数据集多条记录导致 label 不一致”问题
- **Guided-CART 可用实现**：`rerank/replace` 两种模式、`alpha` 控制倾向强度、含一致率/耗时对比脚本

### 文档

- `docs/feature_relevance_algorithm_selection.md`：算法谱系、量化指标、元学习选择与经验库模板
- `docs/zh/guides/核心概念说明.md`：核心概念（含轨迹指导/Guided-*）

### 环境要求

- **Python**：>= 3.10（推荐 3.12+）

安装（推荐，避免脚本找不到 `mtbmt`）：

```bash
python -m pip install -e .
```

Windows PowerShell 也可以用：

```powershell
py -3 -m pip install -e .
```

### 快速开始：生成经验库（本地 CSV）

对某个 CSV 数据集跑“相关性算法基准 + 选择 + 经验入库”（示例）：

```powershell
cd D:\desk\paper
py -3 scripts/benchmark_relevance.py --csv data.csv --target label --k 20 --cv 5 --store experience/experience.jsonl
```

输出会打印 `selected_method`，并将本次经验追加到 `experience/experience.jsonl`。

### 快速开始：批量抓取 OpenML 并入库（~100 个）

该脚本会：从 OpenML 挑选候选数据集 → 下载保存 CSV → 跑基准评测 → 追加写入经验库（支持断点、去重、失败日志、重试）。

```powershell
cd D:\desk\paper
py -3 scripts/fetch_openml_100_and_ingest.py --n 100 --out-dir data/openml_100 --store experience/experience.jsonl --cv 5 --k 20
```

### 元学习选择器评估（重点：Top1 accuracy）

按 `dataset_id` 做分组 K 折（推荐加 `--aggregate-by-dataset`）：

```powershell
cd D:\desk\paper
py -3 scripts/evaluate_meta_selector.py --experience experience/experience.jsonl --cv-datasets 5 --aggregate-by-dataset --topn 2
```

### 轨迹指导（两种实现）

**A. Trajectory Guidance（调参式，最小示例）**：根据偏好参数（更高效果/更短轨迹/更低时间）迭代调整 `DecisionTreeClassifier` 超参：

```powershell
cd D:\desk\paper
py -3 scripts/trajectory/guide_decision_tree.py --w-effect 1.0 --w-length 0.05 --w-time 0.10 --iters 6 --traj-samples 512
```

**B. Guided-CART（分裂重排/替换）**：

- Demo（单次对比 sklearn CART vs Guided-CART）：

```powershell
cd D:\desk\paper
py -3 scripts/trajectory/guided_cart_demo.py --mode rerank --alpha 1.0 --depth 8 --reranker-target val_gain --threshold-strategy unique_midpoints_cap --thresholds 64
```

- 一致率实验（多 seed）：

```powershell
cd D:\desk\paper
py -3 scripts/trajectory/compare_cart_guided_agreement.py --seeds 10 --depth 6 --mode rerank --alpha 1.0 --reranker-target val_gain --threshold-strategy unique_midpoints_cap --thresholds 64 --class-sep 1.0 --flip-y 0.05
```

- 耗时对比实验（脚本会固定跑：sklearn CART + Guided-CART 的几种配置，并拆分统计 reranker 训练等阶段耗时）：

```powershell
cd D:\desk\paper
py -3 scripts/trajectory/benchmark_guided_cart_runtime.py --depth 6 --thresholds 64 --repeats 5 --warmup 1
```

说明：`benchmark_guided_cart_runtime.py` 当前只暴露 `--depth/--shortlist-k/--thresholds/--repeats/--warmup` 等参数；`mode/alpha/reranker_target/threshold_strategy` 请用上面的 demo / 一致率脚本来对比。

- `alpha` 影响分析（看 split 是否变化）：

```powershell
cd D:\desk\paper
py -3 scripts/trajectory/analyze_guided_cart_alpha.py --seed 42 --depth 6 --mode rerank --reranker-target val_gain --threshold-strategy unique_midpoints_cap --thresholds 64
```
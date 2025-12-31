## Zig version (zig_version)

这个目录是对 `scripts/benchmark_relevance.py` / `src/mtbmt/*` 的**核心逻辑**做的一份 Zig 复刻版：

- 读取 CSV（数值特征 + target 列）
- 推断任务类型（分类/回归）
- 计算特征相关性/重要性分数（Pearson / Spearman / Mutual Information）
- 交叉验证评估（选 top-k 特征 → 训练轻量模型 → 得到 CV mean/std）
- 稳定性评估（top-k 的折间 Jaccard）
- 选最优方法并写入 JSONL 经验库

### 构建与运行

本仓库 CI/环境未必自带 Zig。请先安装 Zig（建议 0.12+）。

在仓库根目录运行：

```bash
cd zig_version
zig build -Doptimize=ReleaseFast
./zig-out/bin/mtbmt_benchmark --csv ../data/big.csv --target label --k 50 --cv 5 --store ../experience/experience.jsonl
```

### 方法与限制

- 目前实现方法：`pearson` / `spearman` / `mi`
- `mi` 使用**分箱离散化**做近似（不等同于 sklearn 的 kNN MI）
- 下游模型：
  - 分类：简单 Logistic Regression（梯度下降）
  - 回归：Ridge Regression（正则化最小二乘，线性方程求解）

如果你需要复刻 Python 里的 `perm`（Permutation Importance）和更复杂模型，我们可以在 Zig 侧继续扩展（工程量会明显增加）。


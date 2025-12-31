## MTBMT-Mentor-TrajectaryBasedModelTrainner

通过元学习中学习算法的学习轨迹，实现**算法选择**与**算法训练指导**。

本仓库当前聚焦：**特征相关性/重要性量化算法的对比与选择**，并把“数据元特征 + 学习轨迹特征 + 选择结果”沉淀为经验库，供后续元学习器训练。

### 你将得到什么

- **统一接口**：不同相关性算法输出统一的 per-feature scores
- **可比较量化**：用下游效用（CV得分）+稳定性（Jaccard）+成本（耗时）评价
- **最优算法选择**：先用规则/综合分选最优，再用经验库训练元学习选择器
- **经验特征沉淀**：JSONL 经验库可追加、可检索、可训练

### 文档

- `docs/feature_relevance_algorithm_selection.md`：算法谱系、量化指标、元学习选择与经验库模板

### 快速开始（本地需要 Python3.12+）

安装（可选）：

```bash
python3 -m pip install -e .
```

对某个 CSV 数据集跑“相关性算法基准 + 选择 + 经验入库”（示例）：

```bash
python3 scripts/benchmark_relevance.py --csv data.csv --target label --k 20 --cv 5 --store experience/experience.jsonl
```

输出会打印 `selected_method`，并将本次经验追加到 `experience/experience.jsonl`。

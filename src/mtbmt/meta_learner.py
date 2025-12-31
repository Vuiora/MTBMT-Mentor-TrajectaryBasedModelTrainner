 from __future__ import annotations
 
 import json
 from dataclasses import dataclass
 from pathlib import Path
 from typing import Any, Dict, List, Optional, Tuple
 
 import numpy as np
 from sklearn.compose import ColumnTransformer
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import OneHotEncoder
 
 
 @dataclass(frozen=True)
 class MetaModelBundle:
     model: Pipeline
     feature_columns: List[str]
     target_name: str = "selected_method"
 
 
 def _flatten_features(meta_features: Dict[str, Any], trajectory_features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
     x = dict(meta_features)
     if trajectory_features:
         for k, v in trajectory_features.items():
             x[f"traj__{k}"] = v
     return x
 
 
 def load_experience_jsonl(path: str | Path) -> Tuple[List[Dict[str, Any]], List[str]]:
     rows: List[Dict[str, Any]] = []
     y: List[str] = []
     p = Path(path)
     if not p.exists():
         return rows, y
 
     with p.open("r", encoding="utf-8") as f:
         for line in f:
             line = line.strip()
             if not line:
                 continue
             obj = json.loads(line)
             feats = _flatten_features(obj.get("meta_features", {}), obj.get("trajectory_features"))
             rows.append(feats)
             y.append(obj.get("selected_method", ""))
     return rows, y
 
 
 def train_meta_selector(
     experience_path: str | Path,
     *,
     random_state: int = 0,
     n_estimators: int = 300,
 ) -> MetaModelBundle:
     """
     用经验库训练“相关性量化算法选择器”：
     X = 数据集元特征 + 轨迹特征
     y = selected_method（历史最优方法）
 
     这里用 RandomForest 做一个鲁棒 baseline；后续可以替换成 XGBoost/LightGBM/TabPFN 等。
     """
 
     rows, y = load_experience_jsonl(experience_path)
     if not rows:
         raise ValueError(f"经验库为空：{experience_path}")
 
     # 对齐列
     all_cols = sorted({k for r in rows for k in r.keys()})
     X = [{c: r.get(c, None) for c in all_cols} for r in rows]
 
     # 列类型划分：字符串 -> 类别；其余走 passthrough（None 会被 OneHotEncoder 忽略/或在模型中当缺失）
     cat_cols = [c for c in all_cols if any(isinstance(r.get(c, None), str) for r in rows)]
     num_cols = [c for c in all_cols if c not in cat_cols]
 
     pre = ColumnTransformer(
         transformers=[
             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
             ("num", "passthrough", num_cols),
         ],
         remainder="drop",
     )
 
     clf = RandomForestClassifier(
         n_estimators=n_estimators,
         random_state=random_state,
         n_jobs=-1,
         class_weight="balanced",
     )
 
     model = Pipeline([("pre", pre), ("clf", clf)])
     model.fit(X, y)
 
     return MetaModelBundle(model=model, feature_columns=all_cols)
 
 
 def predict_best_method(bundle: MetaModelBundle, meta_features: Dict[str, Any], trajectory_features: Optional[Dict[str, Any]] = None) -> str:
     x = _flatten_features(meta_features, trajectory_features)
     row = {c: x.get(c, None) for c in bundle.feature_columns}
     pred = bundle.model.predict([row])[0]
     return str(pred)

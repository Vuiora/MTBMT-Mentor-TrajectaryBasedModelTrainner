import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mtbmt.evaluation import evaluate_relevance_method
from mtbmt.experience_store import ExperienceRecord, ExperienceStore, now_utc_iso
from mtbmt.meta_features import compute_dataset_meta_features, infer_task
from mtbmt.relevance.filters import PearsonAbsScorer, SpearmanAbsScorer, MutualInfoScorer, DistanceCorrelationScorer
from mtbmt.relevance.model_based import PermutationImportanceScorer


def _dataset_id_from_path(path: str) -> str:
    p = Path(path)
    h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{p.stem}-{h}"


def main():
    ap = argparse.ArgumentParser(description="Benchmark feature relevance scorers and store experience (JSONL).")
    ap.add_argument("--csv", required=True, help="CSV dataset path")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--k", type=int, default=20, help="Top-k features used for evaluation")
    ap.add_argument("--cv", type=int, default=5, help="CV folds")
    ap.add_argument("--store", default="experience/experience.jsonl", help="Experience store JSONL path")
    ap.add_argument("--scoring", default=None, help="sklearn scoring name (e.g., roc_auc, accuracy, r2, neg_rmse)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"target列不存在：{args.target}")

    y = df[args.target].to_numpy()
    Xdf = df.drop(columns=[args.target])

    # 目前默认：全转为数值；非数值列会变成 NaN（建议上游做更严谨的编码）
    X = Xdf.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    feature_names = list(Xdf.columns)

    approx_task, _, _ = infer_task(y)
    if approx_task == "classification":
        base_est = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    else:
        base_est = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)

    scorers = [
        PearsonAbsScorer(),
        SpearmanAbsScorer(),
        MutualInfoScorer(task=approx_task, n_neighbors=3, random_state=0),
        DistanceCorrelationScorer(),
        PermutationImportanceScorer(estimator=base_est, cv=3, n_repeats=5, random_state=0, task=approx_task),
    ]

    meta = compute_dataset_meta_features(X, y).as_dict()

    evaluations = {}
    eval_objs = []
    for s in scorers:
        ev = evaluate_relevance_method(
            X,
            y,
            s,
            feature_names=feature_names,
            k=args.k,
            cv=args.cv,
            scoring=args.scoring,
            random_state=0,
        )
        evaluations[ev.method_name] = ev.as_dict()
        eval_objs.append(ev)

    # 选择规则：cv_score_mean 最大；如相同则选稳定性更高；再相同选更快
    eval_objs.sort(key=lambda e: (e.cv_score_mean, e.stability_jaccard, -e.runtime_sec), reverse=True)
    best = eval_objs[0]

    rec = ExperienceRecord(
        dataset_id=_dataset_id_from_path(args.csv),
        meta_features=meta,
        trajectory_features=None,  # 轨迹特征需从训练过程日志导入；这里先留空
        evaluations=evaluations,
        selected_method=best.method_name,
        selection_reason={
            "rule": "max(cv_score_mean) -> max(stability_jaccard) -> min(runtime_sec)",
            "best_metrics": best.as_dict(),
        },
        created_at_utc=now_utc_iso(),
    )

    store = ExperienceStore(args.store)
    store.append(rec)

    print("selected_method:", best.method_name)
    print("best_metrics:", best.as_dict())
    print("experience_appended_to:", str(Path(args.store).resolve()))


if __name__ == "__main__":
    main()

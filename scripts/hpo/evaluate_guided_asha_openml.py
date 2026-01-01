from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mtbmt.hpo.guided_asha import HpoConfig, HpoReranker, collect_promotion_training_data, run_asha, run_guided_asha
from mtbmt.meta_features import infer_task


@dataclass
class OneSeedResult:
    seed: int
    base_best: float
    base_final: float
    guided_best: float
    guided_final: float
    guided_disagree_rate: float


def _summary(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(list(xs), dtype=float)
    return float(np.mean(a)), float(np.std(a))


def _build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    # heuristics: object/category/bool -> categorical; rest -> numeric
    cat_cols = [
        c
        for c in X_df.columns
        if (X_df[c].dtype == "object" or str(X_df[c].dtype).startswith("category") or str(X_df[c].dtype) == "bool")
    ]
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )


def _make_gbdt_objective(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Any:
    """
    Objective:
    - model: GradientBoostingClassifier (GBDT)
    - resource: n_estimators
    - score: validation metric (roc_auc for binary, else accuracy)
    - trajectory: staged validation metric for each estimator
    """

    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    approx_task, y_unique, _ = infer_task(y_train)
    is_binary = (approx_task == "classification") and int(y_unique) == 2

    # ensure binary labels are 0/1 for AUC
    if is_binary and not np.issubdtype(y_train.dtype, np.number):
        vals = list(dict.fromkeys(list(y_train)))
        if len(vals) != 2:
            raise ValueError(f"binary classification expected, got labels={vals}")
        m = {vals[0]: 0, vals[1]: 1}
        y_train = np.asarray([m[v] for v in y_train], dtype=int)
        y_val = np.asarray([m[v] for v in y_val], dtype=int)

    def _auc(p1: np.ndarray) -> float:
        return float(roc_auc_score(y_val, p1))

    def objective(cfg: HpoConfig, resource: int, _seed: int) -> Tuple[float, Sequence[float]]:
        p = cfg.params
        lr = float(p.get("learning_rate", 0.1))
        subsample = float(p.get("subsample", 1.0))
        max_depth = int(round(float(p.get("max_depth", 3.0))))
        min_samples_leaf = int(round(float(p.get("min_samples_leaf", 1.0))))

        max_depth = max(1, min(max_depth, 8))
        min_samples_leaf = max(1, min(min_samples_leaf, 200))
        subsample = max(0.2, min(subsample, 1.0))
        lr = max(1e-3, min(lr, 0.5))

        clf = GradientBoostingClassifier(
            n_estimators=max(1, int(resource)),
            learning_rate=lr,
            subsample=subsample,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=int(seed),
        )
        clf.fit(X_train, y_train)

        traj: List[float] = []
        if is_binary:
            for proba in clf.staged_predict_proba(X_val):
                traj.append(_auc(proba[:, 1]))
        else:
            for pred in clf.staged_predict(X_val):
                traj.append(float((pred == y_val).mean()))
        return float(traj[-1]) if traj else float("nan"), traj

    return objective


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate baseline ASHA vs Guided-ASHA on a real dataset (OpenML).")
    ap.add_argument("--openml-id", type=int, default=0, help="OpenML dataset id (preferred).")
    ap.add_argument("--openml-name", type=str, default="", help="OpenML dataset name (alternative).")
    ap.add_argument("--data-home", type=str, default="data/openml_cache", help="Cache dir for OpenML downloads.")

    ap.add_argument("--train-seeds", type=int, default=6, help="How many HPO RNG seeds used to train the reranker.")
    ap.add_argument("--test-seeds", type=int, default=6, help="How many HPO RNG seeds used to evaluate guided vs baseline.")
    ap.add_argument("--seed0", type=int, default=42, help="Base seed.")

    ap.add_argument("--n0", type=int, default=18, help="Initial number of configs.")
    ap.add_argument("--eta", type=int, default=3)
    ap.add_argument("--min-resource", type=int, default=20, help="Min resource (GBDT n_estimators).")
    ap.add_argument("--max-resource", type=int, default=200, help="Max resource (GBDT n_estimators).")

    ap.add_argument("--mode", choices=["rerank", "replace"], default="rerank")
    ap.add_argument("--shortlist-k", type=int, default=9)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--pos-top-frac", type=float, default=0.35, help="Positive label fraction among promotions (training).")
    args = ap.parse_args()

    # --- 0) load dataset (OpenML) ---
    openml_id = int(args.openml_id)
    openml_name = str(args.openml_name).strip()
    if openml_id <= 0 and not openml_name:
        # default dataset (small and common)
        openml_id = 31  # credit-g v1

    print("loading dataset from OpenML...")
    t_load0 = perf_counter()
    try:
        bunch = fetch_openml(data_id=int(openml_id), as_frame=True, data_home=str(args.data_home)) if openml_id > 0 else fetch_openml(
            name=openml_name, as_frame=True, data_home=str(args.data_home)
        )
    except Exception as e:
        raise SystemExit(
            "OpenML 下载失败（可能是网络/证书/代理问题）。你可以改用另一个 --openml-id。\n"
            f"原始错误：{e}"
        )
    t_load = perf_counter() - t_load0

    X_df: pd.DataFrame = bunch.data  # type: ignore[assignment]
    y_ser = bunch.target  # type: ignore[attr-defined]
    if isinstance(y_ser, pd.DataFrame):
        if y_ser.shape[1] != 1:
            raise SystemExit(f"OpenML target has {y_ser.shape[1]} columns; please choose a single target.")
        y_ser = y_ser.iloc[:, 0]
    y = y_ser.to_numpy()

    print(f"loaded: openml_id={getattr(bunch, 'data_id', openml_id)} X={X_df.shape} y={np.asarray(y).shape} in {t_load:.2f}s")

    # --- 1) preprocess to numeric arrays once ---
    X_train_df, X_tmp_df, y_train, y_tmp = train_test_split(
        X_df, y, test_size=0.40, random_state=int(args.seed0), stratify=y
    )
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_tmp_df, y_tmp, test_size=0.50, random_state=int(args.seed0), stratify=y_tmp
    )

    pre = _build_preprocessor(X_train_df)
    t_pre0 = perf_counter()
    X_train = pre.fit_transform(X_train_df)
    X_val = pre.transform(X_val_df)
    X_test = pre.transform(X_test_df)
    t_pre = perf_counter() - t_pre0
    print(f"preprocessed to numeric matrices in {t_pre:.2f}s")

    # Some transformers return sparse
    try:
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()
    except Exception:
        pass

    # --- 2) define search space (GBDT-like) ---
    search_space: Dict[str, Tuple[float, float, str]] = {
        "learning_rate": (0.01, 0.3, "log"),
        "subsample": (0.5, 1.0, "uniform"),
        "max_depth": (1.0, 6.0, "uniform"),
        "min_samples_leaf": (1.0, 60.0, "log"),
    }

    train_seeds = [int(args.seed0) + i for i in range(int(args.train_seeds))]
    test_seeds = [int(args.seed0) + int(args.train_seeds) + i for i in range(int(args.test_seeds))]

    # --- 3) train reranker on baseline ASHA promotions (same dataset, different HPO RNG seeds) ---
    X_rows_all: List[dict] = []
    y_rows_all: List[int] = []
    t_train0 = perf_counter()
    for s in train_seeds:
        objective = _make_gbdt_objective(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, seed=int(s))
        X_rows, y_rows = collect_promotion_training_data(
            objective=objective,
            search_space=search_space,
            X=X_train,
            y=y_train,
            seed=int(s),
            n0=int(args.n0),
            eta=int(args.eta),
            min_resource=int(args.min_resource),
            max_resource=int(args.max_resource),
            positive_top_frac=float(args.pos_top_frac),
        )
        X_rows_all.extend(X_rows)
        y_rows_all.extend(y_rows)
    reranker = HpoReranker(random_state=int(args.seed0))
    reranker.fit(X_rows_all, y_rows_all)
    t_train = perf_counter() - t_train0

    # --- 4) evaluate guided vs baseline on held-out HPO RNG seeds ---
    results: List[OneSeedResult] = []
    t_eval0 = perf_counter()
    for s in test_seeds:
        objective = _make_gbdt_objective(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, seed=int(s))
        base = run_asha(
            objective=objective,
            search_space=search_space,
            seed=int(s),
            n0=int(args.n0),
            eta=int(args.eta),
            min_resource=int(args.min_resource),
            max_resource=int(args.max_resource),
        )
        guided = run_guided_asha(
            objective=objective,
            search_space=search_space,
            X=X_train,
            y=y_train,
            seed=int(s),
            reranker=reranker,
            n0=int(args.n0),
            eta=int(args.eta),
            min_resource=int(args.min_resource),
            max_resource=int(args.max_resource),
            mode=str(args.mode),
            shortlist_k=int(args.shortlist_k),
            alpha=float(args.alpha),
        )

        base_final_obs = base.best_at_resource(int(args.max_resource))
        guided_final_obs = guided.best_at_resource(int(args.max_resource))
        base_final = float(base_final_obs.score) if base_final_obs is not None else float("nan")
        guided_final = float(guided_final_obs.score) if guided_final_obs is not None else float("nan")

        results.append(
            OneSeedResult(
                seed=int(s),
                base_best=float(base.best_score),
                base_final=base_final,
                guided_best=float(guided.best_score),
                guided_final=guided_final,
                guided_disagree_rate=float(guided.promote_disagree_rate),
            )
        )
        print(
            f"hpo_seed={s}  "
            f"val_best(base/guided)={base.best_score:.4f}/{guided.best_score:.4f}  "
            f"val_final@R(base/guided)={base_final:.4f}/{guided_final:.4f}  "
            f"promote_disagree={guided.promote_disagree_rate:.2f}"
        )
    t_eval = perf_counter() - t_eval0

    base_best_m, base_best_s = _summary([r.base_best for r in results])
    guided_best_m, guided_best_s = _summary([r.guided_best for r in results])
    base_fin_m, base_fin_s = _summary([r.base_final for r in results])
    guided_fin_m, guided_fin_s = _summary([r.guided_final for r in results])
    dis_m, dis_s = _summary([r.guided_disagree_rate for r in results])

    print()
    print("=== Guided-ASHA (OpenML) Summary ===")
    ds = f"openml_id={int(getattr(bunch, 'data_id', openml_id))}" if openml_id > 0 else f"openml_name={openml_name}"
    print(f"dataset: {ds}")
    print(f"train_seeds={len(train_seeds)} test_seeds={len(test_seeds)} n0={args.n0} eta={args.eta} R={args.max_resource}")
    print(f"guided: mode={args.mode} shortlist_k={args.shortlist_k} alpha={args.alpha}")
    print()
    print(f"train_reranker_time_sec: {t_train:.3f}")
    print(f"eval_time_sec          : {t_eval:.3f}")
    print()
    print(f"val_best(base)      : mean={base_best_m:.4f} std={base_best_s:.4f}")
    print(f"val_best(guided)    : mean={guided_best_m:.4f} std={guided_best_s:.4f}")
    print(f"val_final@R(base)   : mean={base_fin_m:.4f} std={base_fin_s:.4f}")
    print(f"val_final@R(guided) : mean={guided_fin_m:.4f} std={guided_fin_s:.4f}")
    print(f"promote_disagree_rate : mean={dis_m:.3f} std={dis_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


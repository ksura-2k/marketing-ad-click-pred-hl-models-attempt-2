"""Ad-click prediction: data contract, split, preprocessing, models, evaluation, persistence.

Sprint 4 deliverable (baseline model development + experimentation).  Everything lives in this one
module because the Sprint 4 folder structure allows only main.py and src/models.py.

Flow (see run_pipeline):
    load -> validate contract -> dedupe -> TOP-LEVEL stratified train/test split (first thing that
    touches the data) -> repeated stratified CV on TRAIN only -> select (floor gate + parsimony)
    -> choose threshold on TRAIN out-of-fold predictions -> final refit on TRAIN (new seed)
    -> ONE test evaluation -> persist artifacts + reports.

Rule used everywhere: anything learned from data (imputer medians, scaler, one-hot levels, model
weights, threshold) is fit on TRAIN only; predict() never calls .fit().
"""
from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger("ad_click")

# --------------------------------------------------------------------------------------
# Configuration (explicit, in one place)
# --------------------------------------------------------------------------------------
CONFIG = {
    "target": "clicked_ad",
    "numeric": ["user_age", "time_of_day", "previous_clicks_on_similar_ads"],
    "categorical": ["gender", "device_type", "ad_placement_position", "ad_category"],
    "engineered": ["log_previous_clicks"],  # stateless, row-wise (safe before the split)
    "test_size": 0.25,
    "split_seed": 42,          # exploration seed: top-level split + CV
    "final_seed": 2026,        # DIFFERENT seed for the final refit (skill rule)
    "cv_splits": 5,
    "cv_repeats": 5,
    "min_flag_share": 0.05,    # threshold guard: never flag fewer than 5% of impressions
    "capacity_grid": [0.05, 0.10, 0.20, 0.30],  # lift table; real capacity unknown (UNCONFIRMED)
    "bootstrap_n": 1000,
    "primary_metric": "pr_auc",
    # Brief's value domains (for FLAGGING, not for dropping rows).  Observed data deviates, see README.
    "brief_domains": {
        "age": (13, 90),
        "hour": (0, 23),
        "gender": {"male", "female", "other", "unknown"},
        "device_type": {"mobile", "desktop", "tablet"},
        "ad_placement_position": {"top", "inline", "sidebar"},
    },
}
FEATURES = CONFIG["numeric"] + CONFIG["engineered"] + CONFIG["categorical"]


# --------------------------------------------------------------------------------------
# Data loading and contract
# --------------------------------------------------------------------------------------
def find_data(root: Path) -> Path:
    """Locate the CSV when --data is not given: $AD_CLICK_DATA, data/*.csv, then ./*.csv."""
    import os

    env = os.environ.get("AD_CLICK_DATA")
    if env and Path(env).is_file():
        return Path(env)
    for pattern in ("data/*.csv", "*.csv"):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "No dataset found. Pass --data PATH, set AD_CLICK_DATA, or put the CSV in ./data/ (searched under "
        f"{root})."
    )


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("loaded %s rows x %s cols from %s", *df.shape, path)
    return df


def validate_schema(df: pd.DataFrame, need_target: bool) -> dict:
    """Data contract. Raises on structural errors; returns counts of brief-domain deviations.

    Deviations from the brief are COUNTED and reported, not silently fixed (see README caveats).
    """
    cols = CONFIG["numeric"] + CONFIG["categorical"] + ([CONFIG["target"]] if need_target else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for c in CONFIG["numeric"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"column {c} must be numeric, got {df[c].dtype}")
    if (df["previous_clicks_on_similar_ads"].dropna() < 0).any():
        raise ValueError("previous_clicks_on_similar_ads must be >= 0 (brief constraint)")
    if need_target:
        t = df[CONFIG["target"]]
        if t.isna().any() or not set(t.unique()) <= {0, 1}:
            raise ValueError("target must be binary 0/1 with no missing values")

    dom = CONFIG["brief_domains"]
    lo, hi = dom["age"]
    rep = {
        "n_rows": int(len(df)),
        "age_outside_13_90": int(((df["user_age"] < lo) | (df["user_age"] > hi)).sum()),
        "time_of_day_outside_0_23": int(((df["time_of_day"] < dom["hour"][0]) | (df["time_of_day"] > dom["hour"][1])).sum()),
        "previous_clicks_non_integer": int((df["previous_clicks_on_similar_ads"].dropna() % 1 != 0).sum()),
        "missing_feature_cells": int(df[CONFIG["numeric"] + CONFIG["categorical"]].isna().sum().sum()),
    }
    for c in ("gender", "device_type", "ad_placement_position"):
        vals = set(df[c].dropna().astype(str).str.lower())
        rep[f"{c}_levels_outside_brief"] = sorted(vals - dom[c])
    return rep


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless row-wise features only (nothing learned from data)."""
    out = df.copy()
    out["log_previous_clicks"] = np.log1p(out["previous_clicks_on_similar_ads"].clip(lower=0))
    return out


# --------------------------------------------------------------------------------------
# Models (listed simple -> complex; order is used by the parsimony rule)
# --------------------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # unseen level at inference -> all zeros
    ])
    return ColumnTransformer([
        ("num", num, CONFIG["numeric"] + CONFIG["engineered"]),
        ("cat", cat, CONFIG["categorical"]),
    ])


def candidate_models() -> dict[str, Pipeline]:
    """Strong regularisation everywhere: only ~15 positives exist in TRAIN."""
    est = {
        "prior_baseline": DummyClassifier(strategy="prior"),  # trivial floor, not a trainable model
        "logreg": LogisticRegression(C=0.3, max_iter=2000),
        # imbalance remedy tested by experiment (CV), kept only if it helps; distorts probabilities
        "logreg_balanced": LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"),
        # contrast model: EDA says the previous-clicks effect is concentrated near zero (non-linear),
        # which predicts trees beat the linear model.  README records whether that held.
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                n_jobs=-1, random_state=CONFIG["split_seed"]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                 min_samples_leaf=20, l2_regularization=5.0,
                                                                 random_state=CONFIG["split_seed"]),
    }
    return {k: Pipeline([("prep", build_preprocessor()), ("model", v)]) for k, v in est.items()}


# --------------------------------------------------------------------------------------
# Metrics (each reported next to its trivial floor)
# --------------------------------------------------------------------------------------
def compute_metrics(y, p, train_prior: float) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    prior = np.full(len(y), train_prior)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        # floors: PR-AUC no-skill = prevalence of THIS set; ROC-AUC = 0.5; probabilistic = constant TRAIN prior
        "floor_pr_auc": float(y.mean()),
        "floor_roc_auc": 0.5,
        "floor_log_loss": float(log_loss(y, prior, labels=[0, 1])),
        "floor_brier": float(brier_score_loss(y, prior)),
    }


def lift_table(y, p) -> pd.DataFrame:
    """Share of all clicks captured when only the top-k% scored impressions are served."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="mergesort")
    rows = []
    for k in CONFIG["capacity_grid"]:
        n = max(1, int(round(k * len(y))))
        caught = int(y[order[:n]].sum())
        rows.append({"top_share": k, "impressions": n, "clicks_caught": caught,
                     "share_of_clicks": caught / max(1, y.sum()), "lift": (caught / n) / y.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Cross-validation on TRAIN only
# --------------------------------------------------------------------------------------
def cross_validate(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series):
    """Pooled out-of-fold predictions per repeat -> metrics per repeat (each repeat covers every TRAIN row once)."""
    cv = RepeatedStratifiedKFold(n_splits=CONFIG["cv_splits"], n_repeats=CONFIG["cv_repeats"],
                                 random_state=CONFIG["split_seed"])
    splits = list(cv.split(X, y))
    prior = float(y.mean())
    rows, oof_mean = [], {}
    for name, pipe in models.items():
        oof = np.zeros((CONFIG["cv_repeats"], len(y)))
        for i, (tr, va) in enumerate(splits):
            m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            oof[i // CONFIG["cv_splits"], va] = m.predict_proba(X.iloc[va])[:, 1]
        per_rep = pd.DataFrame([compute_metrics(y, oof[r], prior) for r in range(CONFIG["cv_repeats"])])
        row = {"model": name}
        for c in ("pr_auc", "roc_auc", "log_loss", "brier"):
            row[f"{c}_mean"], row[f"{c}_std"] = per_rep[c].mean(), per_rep[c].std(ddof=1)
        rows.append(row)
        oof_mean[name] = oof.mean(axis=0)
        log.info("CV %-24s PR-AUC %.4f  ROC-AUC %.4f  logloss %.4f  brier %.4f", name, row["pr_auc_mean"],
                 row["roc_auc_mean"], row["log_loss_mean"], row["brier_mean"])
    return pd.DataFrame(rows), oof_mean


def select_model(cv: pd.DataFrame, floor: dict) -> tuple[str, dict]:
    """Gate on the primary-metric floor first, then parsimony: simplest model within 1 std of the best PR-AUC."""
    trainable = cv[cv["model"] != "prior_baseline"].reset_index(drop=True)  # frame order = simple -> complex
    gated = trainable[trainable["pr_auc_mean"] > floor["pr_auc"]]
    if gated.empty:
        raise RuntimeError("No candidate beats the PR-AUC no-skill floor in CV; refusing to pick a winner.")
    best = gated.loc[gated["pr_auc_mean"].idxmax()]
    within = gated[gated["pr_auc_mean"] >= best["pr_auc_mean"] - best["pr_auc_std"]]
    chosen = within.iloc[0]["model"]  # first in simple->complex order
    info = {"best_by_pr_auc": best["model"], "within_one_std": within["model"].tolist(), "chosen": chosen,
            "rule": "gate PR-AUC > floor, then simplest model within 1 std (across CV repeats) of best PR-AUC"}
    return chosen, info


def choose_threshold(y: np.ndarray, p: np.ndarray) -> dict:
    """F1-maximising threshold on TRAIN out-of-fold probabilities, never flagging < min_flag_share of rows.

    Costs/capacity are unknown (UNCONFIRMED default); the lift table is the real decision aid.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    share = np.array([(p >= t).mean() for t in thr])
    ok = share >= CONFIG["min_flag_share"]
    i = int(np.argmax(np.where(ok, f1, -1)))
    return {"threshold": float(thr[i]), "oof_f1": float(f1[i]), "oof_precision": float(prec[i]),
            "oof_recall": float(rec[i]), "oof_flag_share": float(share[i]),
            "rule": f"max F1 on train OOF with flag share >= {CONFIG['min_flag_share']:.0%}"}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"pr_auc": [], "roc_auc": []}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0 or y[idx].sum() == len(y):
            continue
        out["pr_auc"].append(average_precision_score(y[idx], p[idx]))
        out["roc_auc"].append(roc_auc_score(y[idx], p[idx]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def run_pipeline(data_path: str | Path, out_dir: str | Path = "outputs") -> dict:
    t0 = time.time()
    out = Path(out_dir)
    rep_dir, mdl_dir = out / "reports", out / "models"
    rep_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir.mkdir(parents=True, exist_ok=True)
    tgt = CONFIG["target"]

    # 1. load + contract + dedupe (BEFORE the split so duplicates cannot straddle train/test)
    df = load_data(data_path)
    quality = validate_schema(df, need_target=True)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    quality["duplicates_dropped"] = n_dup
    _json(rep_dir / "data_quality.json", quality)
    log.info("data quality vs brief: %s", quality)

    # 2. TOP-LEVEL SPLIT: first operation that uses the labels. Nothing is fit before this line.
    df = add_features(df)  # stateless, so safe before the split
    X, y = df[FEATURES], df[tgt].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=CONFIG["test_size"], stratify=y,
                                              random_state=CONFIG["split_seed"])
    if X_tr.merge(X_te, how="inner").shape[0] > 0:
        raise AssertionError("identical feature rows appear in both train and test")
    _json(rep_dir / "split.json", {"train_rows": len(y_tr), "test_rows": len(y_te), "train_positives": int(y_tr.sum()),
                                   "test_positives": int(y_te.sum()), "seed": CONFIG["split_seed"],
                                   "test_index": sorted(map(int, y_te.index))})
    log.info("split: train %d rows (%d clicks) | test %d rows (%d clicks)", len(y_tr), y_tr.sum(), len(y_te), y_te.sum())
    X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)

    # 3. CV on TRAIN (test untouched)
    models = candidate_models()
    cv, oof = cross_validate(models, X_tr, y_tr)
    floor = compute_metrics(y_tr, np.full(len(y_tr), y_tr.mean()), float(y_tr.mean()))
    chosen, sel = select_model(cv, floor)
    cv.to_csv(rep_dir / "cv_results.csv", index=False)

    # contrast-model verdict (recorded, not assumed)
    lin = cv.set_index("model").loc["logreg", "pr_auc_mean"]
    trees = cv.set_index("model").loc[["random_forest", "hist_gradient_boosting"], "pr_auc_mean"].max()
    sel["contrast_trees_beat_linear"] = bool(trees > lin)
    sel["contrast_gap_pr_auc"] = float(trees - lin)
    sel["train_floor"] = floor
    sel["cv_floor_note"] = "PR-AUC floor = TRAIN prevalence; log-loss/Brier floors = constant TRAIN prior"

    # 4. threshold from TRAIN OOF predictions of the chosen model
    thr = choose_threshold(y_tr.to_numpy(), oof[chosen])
    sel["threshold"] = thr
    _json(rep_dir / "model_selection.json", sel)
    lift_table(y_tr, oof[chosen]).to_csv(rep_dir / "lift_table_train_oof.csv", index=False)
    log.info("chosen=%s  selection=%s  threshold=%.4f", chosen, sel["within_one_std"], thr["threshold"])

    # 5. final refit on TRAIN with the DIFFERENT final seed
    final = clone(models[chosen])
    if "model__random_state" in final.get_params():
        final.set_params(model__random_state=CONFIG["final_seed"])
    final.fit(X_tr, y_tr)

    # 6. SINGLE test evaluation (chosen model + trivial prior baseline only; never per candidate)
    p_te = final.predict_proba(X_te)[:, 1]
    prior = float(y_tr.mean())
    test = compute_metrics(y_te, p_te, prior)
    flagged = p_te >= thr["threshold"]
    tp = int((flagged & (y_te.to_numpy() == 1)).sum())
    test.update({
        "model": chosen, "n_test": int(len(y_te)), "n_test_positives": int(y_te.sum()),
        "threshold": thr["threshold"], "flagged": int(flagged.sum()), "flag_share": float(flagged.mean()),
        "true_positives": tp, "precision": tp / max(1, int(flagged.sum())),
        "recall": tp / max(1, int(y_te.sum())), "precision_floor_flag_all": float(y_te.mean()),
        "bootstrap_95ci": bootstrap_ci(y_te.to_numpy(), p_te, CONFIG["bootstrap_n"], CONFIG["final_seed"]),
        "trivial_baseline": {"pr_auc": float(y_te.mean()), "roc_auc": 0.5,
                             "log_loss": test["floor_log_loss"], "brier": test["floor_brier"]},
    })
    _json(rep_dir / "test_results.json", test)
    lift_table(y_te, p_te).to_csv(rep_dir / "lift_table_test.csv", index=False)
    log.info("TEST %s: PR-AUC %.4f (floor %.4f) ROC-AUC %.4f logloss %.4f (floor %.4f) brier %.4f (floor %.4f)",
             chosen, test["pr_auc"], test["floor_pr_auc"], test["roc_auc"], test["log_loss"], test["floor_log_loss"],
             test["brier"], test["floor_brier"])

    # 7. persist ONE artifact set: fitted pipeline + threshold + contract + environment
    artifact = {"pipeline": final, "model_name": chosen, "threshold": thr["threshold"], "train_prior": prior,
                "features": FEATURES, "config": {k: v for k, v in CONFIG.items() if k != "brief_domains"},
                "trained_levels": {c: sorted(X_tr[c].astype(str).unique()) for c in CONFIG["categorical"]},
                "sklearn": sklearn.__version__, "python": platform.python_version()}
    joblib.dump(artifact, mdl_dir / "model.joblib")

    summary = {"chosen": chosen, "cv": cv.round(4).to_dict("records"), "selection": sel, "test": test,
               "quality": quality, "runtime_s": round(time.time() - t0, 1),
               "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                               "pandas": pd.__version__, "numpy": np.__version__}}
    _json(rep_dir / "run_summary.json", summary)
    return summary


# --------------------------------------------------------------------------------------
# Inference (loads artifacts, transforms, predicts: NO .fit here)
# --------------------------------------------------------------------------------------
def predict(input_csv: str | Path, model_path: str | Path, output_csv: str | Path) -> pd.DataFrame:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model artifact not found: {model_path} (run `python main.py` first)")
    art = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    quality = validate_schema(df, need_target=False)
    unseen = {c: sorted(set(df[c].dropna().astype(str)) - set(art["trained_levels"][c])) for c in CONFIG["categorical"]}
    unseen = {c: v for c, v in unseen.items() if v}
    if unseen or quality["missing_feature_cells"]:
        log.warning("inference input: unseen levels %s, missing cells %d (handled by saved imputer/encoder)",
                    unseen, quality["missing_feature_cells"])
    feats = add_features(df)[art["features"]]
    p = art["pipeline"].predict_proba(feats)[:, 1]
    res = df.copy()
    res["click_probability"] = p
    res["serve_flag"] = (p >= art["threshold"]).astype(int)
    res.to_csv(output_csv, index=False)
    log.info("scored %d rows -> %s (flagged %d)", len(res), output_csv, int(res["serve_flag"].sum()))
    return res

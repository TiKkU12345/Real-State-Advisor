"""Train classification + regression models, track everything in MLflow, register the best."""
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src import config as C

CLF_EXP = "RE_Good_Investment_Classification"
REG_EXP = "RE_Future_Price_Regression"
CLF_REG_NAME = "good_investment_classifier"
REG_REG_NAME = "future_price_regressor"


def make_pipeline(model, num_cols, cat_cols):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def aggregate_importance(pipe, cat_cols):
    """Collapse one-hot columns back to the original feature name."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        raw = np.abs(np.ravel(model.coef_))
    out = {}
    for n, v in zip(names, raw):
        kind, rest = n.split("__", 1)
        key = rest
        if kind == "cat":
            key = next(c for c in sorted(cat_cols, key=len, reverse=True) if rest.startswith(c + "_"))
        out[key] = out.get(key, 0.0) + float(v)
    total = sum(out.values()) or 1.0
    return dict(sorted(((k, v / total) for k, v in out.items()), key=lambda kv: -kv[1]))


def clf_metrics(y, pred, proba):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def reg_metrics(y, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "mae": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred), "mape_pct": float(np.mean(np.abs((y - pred) / y)) * 100)}


def plot_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"])
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return cm


def plot_importance(imp, title, path, top=12):
    items = list(imp.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([k for k, _ in items], [v * 100 for _, v in items], color="steelblue")
    ax.set(title=title, xlabel="Importance (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_actual_vs_pred(y, pred, path):
    idx = np.random.RandomState(0).choice(len(y), 4000, replace=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(np.asarray(y)[idx], np.asarray(pred)[idx], s=5, alpha=0.3)
    lim = [0, max(np.max(y), np.max(pred))]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(title="Test set: actual vs predicted (Lakhs)", xlabel="Actual", ylabel="Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    t0 = time.time()
    C.MODEL_DIR.mkdir(exist_ok=True)
    tmp = C.REPORT_DIR / "tmp"
    tmp.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    client = MlflowClient()

    df = pd.read_csv(C.PROCESSED_PATH)
    idx_trainval, idx_test = train_test_split(df.index, test_size=0.15, random_state=C.RANDOM_STATE,
                                              stratify=df["Good_Investment"])
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15 / 0.85, random_state=C.RANDOM_STATE,
                                          stratify=df.loc[idx_trainval, "Good_Investment"])
    print("split", len(idx_train), len(idx_val), len(idx_test), flush=True)

    clf_specs = {
        "LogisticRegression": (LogisticRegression(max_iter=400, C=1.0), False),
        "RandomForest": (RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=5,
                                                max_samples=0.5, n_jobs=-1, random_state=C.RANDOM_STATE), False),
        "XGBoost_d6": (XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.9,
                                     tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE), False),
        "XGBoost_d4_slow": (XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                          tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE), False),
        "XGBoost_LEAKY_with_price": (XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1,
                                                   tree_method="hist", n_jobs=-1,
                                                   random_state=C.RANDOM_STATE), True),
    }
    mlflow.set_experiment(CLF_EXP)
    clf_results, clf_pipes = {}, {}
    for name, (model, leaky) in clf_specs.items():
        feats = C.CLF_FEATURES + (C.PRICE_LEAK_FEATURES if leaky else [])
        num = [f for f in feats if f not in C.CAT_FEATURES]
        pipe = make_pipeline(model, num, C.CAT_FEATURES)
        with mlflow.start_run(run_name=name) as run:
            ts = time.time()
            pipe.fit(df.loc[idx_train, feats], df.loc[idx_train, "Good_Investment"])
            fit_s = time.time() - ts
            res = {}
            for split, idx in [("val", idx_val), ("test", idx_test)]:
                X, y = df.loc[idx, feats], df.loc[idx, "Good_Investment"]
                pred, proba = pipe.predict(X), pipe.predict_proba(X)[:, 1]
                res[split] = clf_metrics(y, pred, proba)
                for k, v in res[split].items():
                    mlflow.log_metric(f"{split}_{k}", v)
            mlflow.log_params({"model": name, "n_features": len(feats), "uses_price_features": leaky,
                               **{f"hp_{k}": v for k, v in model.get_params().items()
                                  if k in ("n_estimators", "max_depth", "learning_rate", "C", "min_samples_leaf")}})
            mlflow.set_tags({"task": "classification", "leakage_demo": str(leaky)})
            mlflow.log_metric("fit_seconds", fit_s)
            cm = plot_confusion(df.loc[idx_test, "Good_Investment"], pipe.predict(df.loc[idx_test, feats]),
                                f"{name} (test)", tmp / f"cm_{name}.png")
            mlflow.log_artifact(str(tmp / f"cm_{name}.png"))
            imp = aggregate_importance(pipe, C.CAT_FEATURES)
            plot_importance(imp, f"{name}: feature importance", tmp / f"imp_{name}.png")
            mlflow.log_artifact(str(tmp / f"imp_{name}.png"))
            res.update({"confusion_matrix_test": cm.tolist(), "importance": imp, "leaky": leaky,
                        "fit_seconds": round(fit_s, 1)})
            clf_results[name], clf_pipes[name] = res, pipe
        print(f"CLF {name}: val_f1={res['val']['f1']:.4f} test_acc={res['test']['accuracy']:.4f} "
              f"({fit_s:.0f}s)", flush=True)

    reg_specs = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=5,
                                              max_features=0.4, max_samples=0.3, n_jobs=-1,
                                              random_state=C.RANDOM_STATE),
        "XGBoost_d6": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                                   tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE),
        "XGBoost_d8": XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.08, subsample=0.9,
                                   tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE),
    }
    num_reg = [f for f in C.REG_FEATURES if f not in C.CAT_FEATURES]
    mlflow.set_experiment(REG_EXP)
    ytest = df.loc[idx_test, "Future_Price_5Y_Lakhs"]
    baseline_pred = df.loc[idx_test, "Price_in_Lakhs"] * (1 + C.FIXED_RATE) ** C.HORIZON_YEARS
    baseline = reg_metrics(ytest, baseline_pred)
    with mlflow.start_run(run_name="Baseline_fixed_8pct"):
        for k, v in baseline.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.set_tags({"task": "regression", "kind": "baseline"})
    print("REG baseline fixed 8%:", {k: round(v, 3) for k, v in baseline.items()}, flush=True)

    reg_results, reg_pipes = {}, {}
    for name, model in reg_specs.items():
        pipe = make_pipeline(model, num_reg, C.CAT_FEATURES)
        with mlflow.start_run(run_name=name):
            ts = time.time()
            pipe.fit(df.loc[idx_train, C.REG_FEATURES], df.loc[idx_train, "Future_Price_5Y_Lakhs"])
            fit_s = time.time() - ts
            res = {}
            for split, idx in [("val", idx_val), ("test", idx_test)]:
                pred = pipe.predict(df.loc[idx, C.REG_FEATURES])
                res[split] = reg_metrics(df.loc[idx, "Future_Price_5Y_Lakhs"], pred)
                for k, v in res[split].items():
                    mlflow.log_metric(f"{split}_{k}", v)
            mlflow.log_metric("fit_seconds", fit_s)
            mlflow.log_params({"model": name, "n_features": len(C.REG_FEATURES),
                               **{f"hp_{k}": v for k, v in model.get_params().items()
                                  if k in ("n_estimators", "max_depth", "learning_rate", "min_samples_leaf")}})
            mlflow.set_tags({"task": "regression"})
            pred_test = pipe.predict(df.loc[idx_test, C.REG_FEATURES])
            plot_actual_vs_pred(ytest, pred_test, tmp / f"avp_{name}.png")
            mlflow.log_artifact(str(tmp / f"avp_{name}.png"))
            imp = aggregate_importance(pipe, C.CAT_FEATURES)
            plot_importance(imp, f"{name}: feature importance", tmp / f"rimp_{name}.png")
            mlflow.log_artifact(str(tmp / f"rimp_{name}.png"))
            res.update({"importance": imp, "fit_seconds": round(fit_s, 1)})
            reg_results[name], reg_pipes[name] = res, pipe
        print(f"REG {name}: val_rmse={res['val']['rmse']:.3f} test_r2={res['test']['r2']:.4f} "
              f"({fit_s:.0f}s)", flush=True)

    best_clf = max((n for n in clf_results if not clf_results[n]["leaky"]), key=lambda n: clf_results[n]["val"]["f1"])
    best_reg = min(reg_results, key=lambda n: reg_results[n]["val"]["rmse"])
    print("BEST clf:", best_clf, "| BEST reg:", best_reg, flush=True)

    for exp, name, pipe, reg_name, feats in [
        (CLF_EXP, best_clf, clf_pipes[best_clf], CLF_REG_NAME, C.CLF_FEATURES),
        (REG_EXP, best_reg, reg_pipes[best_reg], REG_REG_NAME, C.REG_FEATURES)]:
        mlflow.set_experiment(exp)
        with mlflow.start_run(run_name=f"REGISTER_best_{name}"):
            mlflow.set_tag("selected_by", "validation set")
            info = mlflow.sklearn.log_model(pipe, name="model", registered_model_name=reg_name,
                                            input_example=df.loc[idx_train, feats].head(3),
                                            serialization_format="cloudpickle")
            version = info.registered_model_version
            client.set_registered_model_alias(reg_name, "champion", version)
            print(f"registered {reg_name} v{version} ({name})", flush=True)

    joblib.dump(clf_pipes[best_clf], C.CLF_MODEL_PATH, compress=3)
    joblib.dump(reg_pipes[best_reg], C.REG_MODEL_PATH, compress=3)
    metrics = {
        "split_sizes": {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)},
        "classification": {"best": best_clf, "models": clf_results},
        "regression": {"best": best_reg, "models": reg_results, "baseline_fixed_8pct_test": baseline},
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(C.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print("TRAINING DONE in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()

"""
src/evaluate.py  —  Final Evaluation + GridSearchCV Hyperparameter Tuning

Run from the project root with your venv active:
    python src/evaluate.py

Prerequisite: run  python src/model_a_train.py  first.

What this script does:
  1. Loads every trained Model A classifier from models/model_a/traditional/
  2. Loads pre-built val & test feature matrices saved by model_a_train.py
  3. Evaluates every model on the TEST set:
       Binary Accuracy, Macro F1, Question-Level Accuracy, Inference Time
  4. Saves confusion matrix PNGs for each model
  5. Runs GridSearchCV on an LR pipeline to tune C and vocab size
  6. Prints a final comparison table (copy-paste into your report)
  7. Saves  data/processed/evaluation_results.csv

Outputs
-------
  data/processed/evaluation_results.csv   — numeric results for all models
  data/processed/cm_test_*.png            — confusion matrix PNGs (test set)
  data/processed/gridsearch_results.csv   — GridSearchCV CV scores
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models" / "model_a" / "traditional"

# ── Check prerequisites ───────────────────────────────────────────────────────
_needed_models = ["ohe_vectorizer.pkl", "lr_model.pkl", "svm_model.pkl",
                  "nb_model.pkl", "ensemble_model.pkl"]
_missing = [f for f in _needed_models if not (MODELS_DIR / f).exists()]
if _missing:
    print(f"\nMissing model files: {_missing}")
    print("Fix: run  python src/model_a_train.py  first.")
    sys.exit(1)

_needed_data = ["X_val.npz", "y_val.npy", "val_expanded.csv",
                "test_expanded.csv", "test_clean.csv", "train_expanded.csv"]
_missing_data = [f for f in _needed_data if not (DATA_PROC / f).exists()]
if _missing_data:
    print(f"\nMissing data files in data/processed/: {_missing_data}")
    print("Fix: run  python src/preprocessing.py  then  python src/model_a_train.py.")
    sys.exit(1)

# ── Package imports ───────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import normalize
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    )
except ImportError as e:
    print(f"\nMissing package: {e}")
    print("Fix: make sure venv is active, then:  pip install -r requirements.txt")
    sys.exit(1)

# ── Pickle compatibility shim ─────────────────────────────────────────────────
# The ensemble model was pickled when SoftVotingEnsemble lived in __main__
# (model_a_train.py). Defining it here lets joblib.load resolve the class.
class SoftVotingEnsemble:
    """Averages predicted probabilities from multiple models."""
    def __init__(self, models, names):
        self.models = models
        self.names = names

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models])
        return probs.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ── Configuration ─────────────────────────────────────────────────────────────
GRIDSEARCH_TRAIN_ROWS = 15_000   # subset of training rows for GridSearchCV
GRIDSEARCH_CV         = 3        # k-fold CV (3 is fast; increase to 5 for final report)
RANDOM_STATE          = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FEATURE HELPERS  (same logic as model_a_train.py so test set uses
#     the IDENTICAL transformations — no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def _row_cosine(A, B):
    A_n = normalize(A, norm="l2")
    B_n = normalize(B, norm="l2")
    sims = A_n.multiply(B_n).sum(axis=1)
    return np.asarray(sims).flatten()


def build_cosine_features(ohe, clean_df):
    """Cosine similarity between article and each option — returns (4*n, 1) sparse."""
    art_vecs = ohe.transform(clean_df["article_clean"].fillna("").tolist())
    cos_cols = []
    for opt in ["A", "B", "C", "D"]:
        opt_vecs = ohe.transform(clean_df[f"{opt}_clean"].fillna("").tolist())
        cos_cols.append(_row_cosine(art_vecs, opt_vecs))

    n = len(clean_df)
    interleaved = np.empty(n * 4)
    for j, col in enumerate(cos_cols):
        interleaved[j::4] = col   # A→0,4,8…; B→1,5,9…; C→2,6,10…; D→3,7,11…
    return sp.csr_matrix(interleaved.reshape(-1, 1))


def build_all_features(ohe, exp_df, clean_df):
    """OHE + cosine similarity + numeric features → single sparse matrix."""
    X_ohe = ohe.transform(exp_df["combined"])
    X_cos = build_cosine_features(ohe, clean_df)
    X_num = sp.csr_matrix(exp_df[["article_length", "q_length"]].values.astype(float))
    return sp.hstack([X_ohe, X_cos, X_num], format="csr")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def question_level_accuracy(model, X, exp_df):
    """
    For every group of 4 consecutive rows (= one question with options A–D),
    check whether the model assigns the highest confidence score to the correct option.
    This is the most meaningful metric for a multiple-choice task.
    Random baseline = 0.25 (1 in 4 chance of picking right answer by luck).
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    else:
        scores = model.predict(X).astype(float)

    labels  = exp_df["is_correct"].values
    n_q     = len(exp_df) // 4
    correct = 0
    for i in range(n_q):
        grp_scores = scores[i * 4: i * 4 + 4]
        grp_labels = labels[i * 4: i * 4 + 4]
        if grp_labels.sum() > 0:
            if np.argmax(grp_scores) == np.argmax(grp_labels):
                correct += 1
    return correct / max(n_q, 1)


def evaluate_model(name, model, X, y, exp_df, split="test"):
    """Run all metrics and return a results dict."""
    t0     = time.time()
    y_pred = model.predict(X)
    elapsed = time.time() - t0

    acc    = accuracy_score(y, y_pred)
    f1_mac = f1_score(y, y_pred, average="macro",  zero_division=0)
    f1_bin = f1_score(y, y_pred, average="binary", zero_division=0)
    q_acc  = question_level_accuracy(model, X, exp_df)

    print(f"\n  ── {name} ──")
    print(f"    Binary Acc  : {acc:.4f}   (majority-class baseline ≈ 0.75)")
    print(f"    Macro F1    : {f1_mac:.4f}   (target > 0.50)")
    print(f"    Binary F1   : {f1_bin:.4f}")
    print(f"    Q-Level Acc : {q_acc:.4f}   (random baseline = 0.25, target > 0.40)")
    print(f"    Infer. time : {elapsed:.2f}s  on {len(y):,} rows")

    return {
        "Model":          name,
        "Split":          split,
        "Binary Acc":     round(acc,    4),
        "Macro F1":       round(f1_mac, 4),
        "Binary F1":      round(f1_bin, 4),
        "Q-Level Acc":    round(q_acc,  4),
        "Infer. Time(s)": round(elapsed, 2),
    }


def save_cm(name, model, X, y, split="test"):
    """Save a confusion matrix PNG for a given split."""
    y_pred = model.predict(X)
    cm  = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Incorrect (0)", "Correct (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {name}\n({split} set)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = DATA_PROC / f"cm_{split}_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved : data/processed/cm_{split}_{safe}.png")


def _section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load:
      - OHE vectorizer fitted during training
      - Pre-built val feature matrix (X_val.npz / y_val.npy saved by model_a_train.py)
      - Test expanded + clean CSV (to build test features fresh — same as training procedure)
      - Train expanded CSV (small subset for GridSearchCV)
    """
    _section("Loading data and models")

    ohe = joblib.load(MODELS_DIR / "ohe_vectorizer.pkl")
    print("  Loaded: ohe_vectorizer.pkl")

    # ── Val features (pre-built by model_a_train.py) ──────────────────────────
    X_val = sp.load_npz(DATA_PROC / "X_val.npz")
    y_val = np.load(DATA_PROC / "y_val.npy")
    val_exp = pd.read_csv(DATA_PROC / "val_expanded.csv")
    print(f"  Loaded: X_val {X_val.shape}, y_val {y_val.shape}")

    # ── Test features — build now using saved OHE (never fit on test!) ─────────
    print("\n  Building test features (this takes ~30 s) ...")
    test_exp = pd.read_csv(DATA_PROC / "test_expanded.csv")

    # Matching val.csv / dev.csv naming for test_clean
    _tc_path = DATA_PROC / "test_clean.csv"
    test_clean = pd.read_csv(_tc_path, index_col=0)
    # Fill any missing clean columns (safety guard)
    for col in ["article_clean", "A_clean", "B_clean", "C_clean", "D_clean"]:
        if col not in test_clean.columns:
            test_clean[col] = ""

    t0 = time.time()
    X_test = build_all_features(ohe, test_exp, test_clean)
    y_test = test_exp["is_correct"].values.astype(int)
    print(f"  Test features built in {time.time()-t0:.1f}s  |  shape: {X_test.shape}")

    # ── Small training subset for GridSearchCV ────────────────────────────────
    print(f"\n  Sampling {GRIDSEARCH_TRAIN_ROWS:,} training rows for GridSearchCV ...")
    train_exp = pd.read_csv(DATA_PROC / "train_expanded.csv")
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(train_exp), size=min(GRIDSEARCH_TRAIN_ROWS, len(train_exp)),
                     replace=False)
    idx.sort()   # keep consecutive-row grouping mostly intact for question-level logic
    gs_exp  = train_exp.iloc[idx].reset_index(drop=True)
    gs_text = gs_exp["combined"].tolist()
    gs_y    = gs_exp["is_correct"].values.astype(int)
    print(f"  GridSearchCV subset: {len(gs_exp):,} rows")

    return ohe, X_val, y_val, val_exp, X_test, y_test, test_exp, gs_text, gs_y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOAD ALL TRAINED MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    """Load every .pkl file saved by model_a_train.py.  Skip silently if absent."""
    _section("Loading trained models")

    models = {}

    def _load(key, filename, label=None):
        path = MODELS_DIR / filename
        if path.exists():
            models[key] = joblib.load(path)
            print(f"  Loaded: {filename}")
        else:
            print(f"  SKIP (not found): {filename}")

    _load("lr",       "lr_model.pkl",         "Logistic Regression")
    _load("svm",      "svm_model.pkl",         "LinearSVC (SVM)")
    _load("nb",       "nb_model.pkl",          "ComplementNB")
    _load("rf",       "rf_model.pkl",          "Random Forest")
    _load("xgb",      "xgb_model.pkl",         "XGBoost")
    _load("kmeans",   "kmeans_model.pkl",       "KMeans")
    _load("gmm",      "gmm_model.pkl",          "GMM")
    _load("lp",       "label_prop_model.pkl",   "LabelPropagation")
    _load("ensemble", "ensemble_model.pkl",     "Ensemble (LR+NB)")

    # SVD transformers needed for unsupervised models
    if "gmm" in models and (MODELS_DIR / "gmm_svd.pkl").exists():
        models["gmm_svd"] = joblib.load(MODELS_DIR / "gmm_svd.pkl")
        print("  Loaded: gmm_svd.pkl")
    if "lp" in models and (MODELS_DIR / "label_prop_svd.pkl").exists():
        models["lp_svd"] = joblib.load(MODELS_DIR / "label_prop_svd.pkl")
        print("  Loaded: label_prop_svd.pkl")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATE SUPERVISED MODELS ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def eval_supervised(models, X_test, y_test, test_exp):
    """Evaluate LR, SVM, NB, RF, XGBoost, Ensemble on the test set."""
    _section("Supervised Model Evaluation — TEST SET")

    results = []
    supervised = [
        ("lr",       "Logistic Regression"),
        ("svm",      "LinearSVC (SVM)"),
        ("nb",       "ComplementNB"),
        ("rf",       "Random Forest"),
        ("xgb",      "XGBoost"),
        ("ensemble", "Ensemble (LR+NB)"),
    ]

    for key, name in supervised:
        if key not in models:
            continue
        model = models[key]
        res = evaluate_model(name, model, X_test, y_test, test_exp, split="test")
        save_cm(name, model, X_test, y_test, split="test")
        results.append(res)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  EVALUATE UNSUPERVISED / SEMI-SUPERVISED ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def eval_unsupervised(models, X_test, y_test, test_exp):
    """Evaluate KMeans, GMM, LabelPropagation.  These need the SVD transform first."""
    _section("Unsupervised / Semi-Supervised Evaluation — TEST SET")

    results = []

    # ── KMeans ────────────────────────────────────────────────────────────────
    if "kmeans" in models:
        print("\n  ── KMeans ──")
        km = models["kmeans"]
        # KMeans clusters don't have a fixed label assignment; we find the best
        # mapping between cluster IDs and true labels (0/1) using majority vote.
        labels_km = km.predict(X_test)
        # Map each cluster to majority class
        mapping = {}
        for c in np.unique(labels_km):
            mask = labels_km == c
            mapping[c] = int(np.round(y_test[mask].mean()))

        y_pred_km = np.array([mapping[c] for c in labels_km])
        acc    = accuracy_score(y_test, y_pred_km)
        f1_mac = f1_score(y_test, y_pred_km, average="macro",  zero_division=0)
        f1_bin = f1_score(y_test, y_pred_km, average="binary", zero_division=0)

        # Question-level: use distance to nearest centroid as proxy score
        # (closer to the centroid that maps to 1 → higher probability of correct)
        dists = km.transform(X_test)   # shape (n, n_clusters)
        correct_cluster = [c for c, v in mapping.items() if v == 1]
        if correct_cluster:
            # Use negative distance to correct cluster as score (higher = better match)
            scores = -dists[:, correct_cluster[0]]
        else:
            scores = -dists.min(axis=1)

        n_q = len(test_exp) // 4
        q_correct = 0
        labels_arr = test_exp["is_correct"].values
        for i in range(n_q):
            grp_s = scores[i * 4: i * 4 + 4]
            grp_l = labels_arr[i * 4: i * 4 + 4]
            if grp_l.sum() > 0 and np.argmax(grp_s) == np.argmax(grp_l):
                q_correct += 1
        q_acc = q_correct / max(n_q, 1)

        print(f"    Binary Acc  : {acc:.4f}")
        print(f"    Macro F1    : {f1_mac:.4f}")
        print(f"    Q-Level Acc : {q_acc:.4f}")

        # Save CM
        cm  = confusion_matrix(y_test, y_pred_km)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Incorrect (0)", "Correct (1)"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title("Confusion Matrix — KMeans (test)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(DATA_PROC / "cm_test_kmeans.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Plot saved : data/processed/cm_test_kmeans.png")

        results.append({
            "Model": "KMeans", "Split": "test",
            "Binary Acc": round(acc, 4), "Macro F1": round(f1_mac, 4),
            "Binary F1":  round(f1_bin, 4), "Q-Level Acc": round(q_acc, 4),
            "Infer. Time(s)": 0.0,
        })

    # ── GMM ───────────────────────────────────────────────────────────────────
    if "gmm" in models and "gmm_svd" in models:
        print("\n  ── GaussianMixture (GMM) ──")
        gmm     = models["gmm"]
        gmm_svd = models["gmm_svd"]

        X_test_svd = gmm_svd.transform(X_test)
        labels_gmm = gmm.predict(X_test_svd)

        mapping_gmm = {}
        for c in np.unique(labels_gmm):
            mask = labels_gmm == c
            mapping_gmm[c] = int(np.round(y_test[mask].mean()))

        y_pred_gmm = np.array([mapping_gmm[c] for c in labels_gmm])
        acc    = accuracy_score(y_test, y_pred_gmm)
        f1_mac = f1_score(y_test, y_pred_gmm, average="macro",  zero_division=0)
        f1_bin = f1_score(y_test, y_pred_gmm, average="binary", zero_division=0)

        # Q-level: use GMM probability of the "correct" component
        proba_gmm = gmm.predict_proba(X_test_svd)
        correct_comp = [c for c, v in mapping_gmm.items() if v == 1]
        if correct_comp:
            scores_gmm = proba_gmm[:, correct_comp[0]]
        else:
            scores_gmm = proba_gmm.max(axis=1)

        q_correct = 0
        labels_arr = test_exp["is_correct"].values
        for i in range(n_q := len(test_exp) // 4):
            grp_s = scores_gmm[i * 4: i * 4 + 4]
            grp_l = labels_arr[i * 4: i * 4 + 4]
            if grp_l.sum() > 0 and np.argmax(grp_s) == np.argmax(grp_l):
                q_correct += 1
        q_acc = q_correct / max(n_q, 1)

        print(f"    Binary Acc  : {acc:.4f}")
        print(f"    Macro F1    : {f1_mac:.4f}")
        print(f"    Q-Level Acc : {q_acc:.4f}")

        cm  = confusion_matrix(y_test, y_pred_gmm)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Incorrect (0)", "Correct (1)"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title("Confusion Matrix — GMM (test)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(DATA_PROC / "cm_test_gmm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Plot saved : data/processed/cm_test_gmm.png")

        results.append({
            "Model": "GMM", "Split": "test",
            "Binary Acc": round(acc, 4), "Macro F1": round(f1_mac, 4),
            "Binary F1":  round(f1_bin, 4), "Q-Level Acc": round(q_acc, 4),
            "Infer. Time(s)": 0.0,
        })

    # ── LabelPropagation ──────────────────────────────────────────────────────
    if "lp" in models and "lp_svd" in models:
        print("\n  ── LabelPropagation ──")
        lp     = models["lp"]
        lp_svd = models["lp_svd"]
        X_test_svd = lp_svd.transform(X_test)
        res = evaluate_model("LabelPropagation", lp, X_test_svd, y_test, test_exp, split="test")
        save_cm("LabelPropagation", lp, X_test_svd, y_test, split="test")
        results.append(res)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRIDSEARCHCV — HYPERPARAMETER TUNING FOR LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def run_gridsearch(gs_text, gs_y):
    """
    Run GridSearchCV on a Pipeline(CountVectorizer → LogisticRegression).
    Uses raw 'combined' text from a 15k-row training subset so the search
    also tunes the vocabulary size alongside C.
    """
    _section(f"GridSearchCV — LR Pipeline  [{GRIDSEARCH_TRAIN_ROWS:,} training rows, "
             f"{GRIDSEARCH_CV}-fold CV]")

    pipe = Pipeline([
        ("vec", CountVectorizer(
            binary=True,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )),
        ("clf", LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=500,
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "vec__max_features": [5_000, 10_000, 20_000],
        "clf__C":            [0.01, 0.1, 1.0, 10.0],
    }

    print(f"\n  Grid: {param_grid}")
    print(f"  Total fits: {GRIDSEARCH_CV} × {len(param_grid['vec__max_features'])} × "
          f"{len(param_grid['clf__C'])} = "
          f"{GRIDSEARCH_CV * len(param_grid['vec__max_features']) * len(param_grid['clf__C'])}")

    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=GRIDSEARCH_CV,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    gs.fit(gs_text, gs_y)
    elapsed = time.time() - t0

    print(f"\n  GridSearchCV finished in {elapsed:.1f}s")
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")

    # Save CV results
    cv_df = pd.DataFrame(gs.cv_results_)[
        ["param_vec__max_features", "param_clf__C", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    cv_df.columns = ["max_features", "C", "Mean CV F1", "Std CV F1", "Rank"]
    cv_path = DATA_PROC / "gridsearch_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"  Saved: data/processed/gridsearch_results.csv")

    print("\n  Top 5 parameter combinations:")
    print(cv_df.head(5).to_string(index=False))

    # ── Plot GridSearch heatmap ────────────────────────────────────────────────
    pivot = cv_df.pivot(index="max_features", columns="C", values="Mean CV F1")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("C (regularisation strength)")
    ax.set_ylabel("max_features (vocab size)")
    ax.set_title("GridSearchCV — Mean CV Macro F1", fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center",
                    color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(DATA_PROC / "gridsearch_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: data/processed/gridsearch_heatmap.png")

    return gs.best_params_, gs.best_score_


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results):
    """Pretty-print a results table for the report."""
    _section("Final Model Comparison Table")

    if not results:
        print("  (no results to display)")
        return

    # Column widths
    cols = ["Model", "Binary Acc", "Macro F1", "Q-Level Acc", "Infer. Time(s)"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in results))
              for c in cols}

    header = "  " + "  ".join(str(c).ljust(widths[c]) for c in cols)
    sep    = "  " + "  ".join("-" * widths[c] for c in cols)
    print("\n" + header)
    print(sep)
    for r in results:
        row = "  " + "  ".join(str(r.get(c, "N/A")).ljust(widths[c]) for c in cols)
        print(row)
    print()
    print("  Baselines: Binary Acc majority-class = 0.75, Q-Level random = 0.25")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  VAL SET SANITY CHECK  (mirrors what model_a_train.py already computed)
# ─────────────────────────────────────────────────────────────────────────────

def eval_val_sanity(models, X_val, y_val, val_exp):
    """
    Quick validation-set check to confirm loaded models still perform as expected.
    Runs only LR and Ensemble (fastest) — just a sanity check, not the main results.
    """
    _section("Validation-Set Sanity Check (LR + Ensemble only)")

    for key, name in [("lr", "Logistic Regression"), ("ensemble", "Ensemble (LR+NB)")]:
        if key in models:
            evaluate_model(name, models[key], X_val, y_val, val_exp, split="val")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RACE Project — Evaluation & GridSearchCV")
    print("=" * 60)

    # ---- Load everything ----
    (ohe, X_val, y_val, val_exp,
     X_test, y_test, test_exp,
     gs_text, gs_y) = load_data()

    models = load_models()

    # ---- Val sanity check ----
    eval_val_sanity(models, X_val, y_val, val_exp)

    # ---- Test set evaluation ----
    results_sup   = eval_supervised(models, X_test, y_test, test_exp)
    results_unsup = eval_unsupervised(models, X_test, y_test, test_exp)

    all_results = results_sup + results_unsup

    # ---- GridSearchCV ----
    best_params, best_cv_f1 = run_gridsearch(gs_text, gs_y)

    # ---- Save results CSV ----
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = DATA_PROC / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved: data/processed/evaluation_results.csv")

    # ---- Print comparison table ----
    print_table(all_results)

    # ---- Summary for report ----
    _section("Summary for Report")
    print(f"  GridSearchCV best params : {best_params}")
    print(f"  GridSearchCV best CV F1  : {best_cv_f1:.4f}")
    if all_results:
        best = max(all_results, key=lambda r: r["Q-Level Acc"])
        print(f"  Best model (Q-Level Acc) : {best['Model']}  →  {best['Q-Level Acc']:.4f}")
        best_f1 = max(all_results, key=lambda r: r["Macro F1"])
        print(f"  Best model (Macro F1)    : {best_f1['Model']}  →  {best_f1['Macro F1']:.4f}")

    print("\n" + "=" * 60)
    print("  DONE — Evaluation complete!")
    print("=" * 60)
    print("  Files written to data/processed/:")
    print("    evaluation_results.csv   — all model metrics")
    print("    gridsearch_results.csv   — GridSearchCV CV scores")
    print("    gridsearch_heatmap.png   — heatmap of C vs max_features")
    print("    cm_test_*.png            — confusion matrices (test set)")
    print()
    print("  Next step: run  python ui/app.py  (Streamlit UI)")
    print("=" * 60)


if __name__ == "__main__":
    main()

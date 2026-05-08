"""
src/model_a_train.py  —  Model A Training Pipeline (Classical ML Only)

Run from the project root with your venv active:
    python src/model_a_train.py

Prerequisite: run  python src/preprocessing.py  first.

This script trains EVERY model required by the project roadmap for Model A
(answer verification task: given article + question + one option, predict
whether that option is the correct answer — binary classification).

  SUPERVISED  (LR, SVM, NB on full 351k rows; RF and XGBoost on a subset):
    Logistic Regression, LinearSVC, ComplementNB, RandomForest, XGBoost

  UNSUPERVISED  (on a subset + dimensionality reduction):
    KMeans (n_clusters=4), GaussianMixture (n_components=4)

  SEMI-SUPERVISED  (on a small subset):
    LabelPropagation

  ENSEMBLE:
    Soft-voting over LR + NB  (reuses already-trained models, no retraining)

Saved files:
    models/model_a/traditional/ohe_vectorizer.pkl   (MUST be loaded at inference)
    models/model_a/traditional/lr_model.pkl
    models/model_a/traditional/svm_model.pkl
    models/model_a/traditional/nb_model.pkl
    models/model_a/traditional/rf_model.pkl
    models/model_a/traditional/xgb_model.pkl        (if xgboost installed)
    models/model_a/traditional/kmeans_model.pkl
    models/model_a/traditional/gmm_model.pkl  +  gmm_svd.pkl
    models/model_a/traditional/label_prop_model.pkl  +  label_prop_svd.pkl
    models/model_a/traditional/ensemble_model.pkl

    data/processed/cm_*.png                          (confusion matrix plots)
    data/processed/X_val.npz  +  y_val.npy           (for evaluate.py)
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
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Check prerequisites ───────────────────────────────────────────────────────
_needed = ["train_expanded.csv", "val_expanded.csv", "test_expanded.csv",
           "train_clean.csv",    "val_clean.csv",    "test_clean.csv"]
_missing = [f for f in _needed if not (DATA_PROC / f).exists()]
if _missing:
    print(f"\nMissing files in data/processed/: {_missing}")
    print("Fix: run  python src/preprocessing.py  first, then retry.")
    sys.exit(1)

# ── Package imports ───────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import joblib
    import matplotlib
    matplotlib.use("Agg")   # headless — saves PNGs without a GUI window
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import ComplementNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.semi_supervised import LabelPropagation
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        ConfusionMatrixDisplay, silhouette_score, classification_report,
    )
except ImportError as e:
    print(f"\nMissing package: {e}")
    print("Fix: make sure venv is active, then:  pip install -r requirements.txt")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Note: xgboost not installed — XGBoost model will be skipped.")

# ── Configuration  (change these if you need to speed up or use more data) ───
OHE_MAX_FEATURES = 10_000   # vocabulary size for CountVectorizer
SUBSET_RF        = 30_000   # rows used for RandomForest  (sparse, memory-safe)
SUBSET_XGB       = 30_000   # rows used for XGBoost
SUBSET_CLUSTER   = 10_000   # rows for KMeans / GMM
SUBSET_LP        =  5_000   # rows for LabelPropagation
LP_LABELED_FRAC  =  0.10    # 10% labeled, 90% unlabeled for LabelPropagation
N_CLUSTERS       = 4        # must match A / B / C / D answer classes
SVD_DIM          = 50       # TruncatedSVD components (for GMM and LP)
RANDOM_STATE     = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def fit_ohe_vectorizer(train_exp):
    """
    Fit CountVectorizer (One-Hot Encoding style) on training combined text.
    CRITICAL: fit on training data ONLY.  Transform val/test separately.
    """
    print("\n  [OHE] Fitting CountVectorizer on train combined text ...")
    print(f"        max_features={OHE_MAX_FEATURES}, binary=True, stop_words='english'")
    ohe = CountVectorizer(
        max_features=OHE_MAX_FEATURES,
        stop_words="english",
        binary=True,        # 1 if word present, 0 if not  (classic OHE style)
        min_df=2,           # ignore words appearing in < 2 documents
        max_df=0.95,        # ignore words appearing in > 95% of documents
    )
    t0 = time.time()
    X = ohe.fit_transform(train_exp["combined"])
    print(f"        Done in {time.time()-t0:.1f}s  |  Vocab size: {len(ohe.vocabulary_):,}")
    print(f"        Matrix shape: {X.shape}  |  NNZ: {X.nnz:,}  |  dtype: {X.dtype}")

    joblib.dump(ohe, MODELS_DIR / "ohe_vectorizer.pkl")
    print("        Saved: models/model_a/traditional/ohe_vectorizer.pkl")
    return ohe, X


def _row_cosine(A, B):
    """
    Element-wise cosine similarity between corresponding rows of sparse matrices.
    A[i] vs B[i] for every i.  Returns numpy array of shape (n_rows,).
    """
    A_n = normalize(A, norm="l2")
    B_n = normalize(B, norm="l2")
    sims = A_n.multiply(B_n).sum(axis=1)
    return np.asarray(sims).flatten()


def build_cosine_features(ohe, clean_df):
    """
    For each original question row in clean_df, compute cosine similarity
    between the article and each of the 4 answer options (A, B, C, D).

    Returns a sparse column matrix of shape (4 * len(clean_df), 1).
    The rows are interleaved: [sim_A0, sim_B0, sim_C0, sim_D0,
                                sim_A1, sim_B1, sim_C1, sim_D1, ...]
    — matching the row order produced by expand_for_model_a() in preprocessing.py.
    """
    art_vecs = ohe.transform(clean_df["article_clean"].fillna("").tolist())
    cos_cols = []
    for opt in ["A", "B", "C", "D"]:
        opt_texts = clean_df[f"{opt}_clean"].fillna("").tolist()
        opt_vecs = ohe.transform(opt_texts)
        cos_cols.append(_row_cosine(art_vecs, opt_vecs))   # shape: (n_orig,)

    n = len(clean_df)
    interleaved = np.empty(n * 4)
    for j, col in enumerate(cos_cols):
        interleaved[j::4] = col      # A at 0,4,8,…; B at 1,5,9,…; etc.
    return sp.csr_matrix(interleaved.reshape(-1, 1))


def build_all_features(ohe, exp_df, clean_df):
    """
    Stack three feature types into one sparse matrix:
      - OHE (CountVectorizer on combined text)     shape: (n, 10000)
      - cosine similarity (article vs option)      shape: (n, 1)
      - numerical (article_length, q_length)       shape: (n, 2)
    """
    X_ohe = ohe.transform(exp_df["combined"])
    X_cos = build_cosine_features(ohe, clean_df)
    X_num = sp.csr_matrix(exp_df[["article_length", "q_length"]].values.astype(float))
    return sp.hstack([X_ohe, X_cos, X_num], format="csr")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def question_level_accuracy(model, X, exp_df):
    """
    For each original question (4 consecutive rows = 4 options),
    check whether the model assigns the highest score to the correct option.
    This is the most meaningful metric for a multiple-choice task.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        rng = raw.max() - raw.min()
        scores = (raw - raw.min()) / (rng + 1e-9)
    else:
        scores = model.predict(X).astype(float)

    labels = exp_df["is_correct"].values
    n_q = len(exp_df) // 4
    correct = 0
    for i in range(n_q):
        grp_scores = scores[i * 4: i * 4 + 4]
        grp_labels = labels[i * 4: i * 4 + 4]
        if grp_labels.sum() > 0:                       # skip malformed groups
            if np.argmax(grp_scores) == np.argmax(grp_labels):
                correct += 1
    return correct / max(n_q, 1)


def evaluate_model(name, model, X_val, y_val, val_exp):
    """Compute and print all metrics.  Returns a result dict for the summary table."""
    t0 = time.time()
    y_pred = model.predict(X_val)
    inf_time = time.time() - t0

    acc    = accuracy_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average="macro")
    f1_bin = f1_score(y_val, y_pred, average="binary", zero_division=0)
    q_acc  = question_level_accuracy(model, X_val, val_exp)

    print(f"\n  Results — {name}")
    print(f"    Binary accuracy   : {acc:.4f}  (random baseline = 0.75 if always predict 0)")
    print(f"    Macro F1          : {f1_mac:.4f}  (target > 0.50)")
    print(f"    Binary F1         : {f1_bin:.4f}")
    print(f"    Question-level    : {q_acc:.4f}  (target > 0.40; random baseline = 0.25)")
    print(f"    Inference time    : {inf_time:.2f}s  on val set ({len(y_val):,} rows)")

    return {
        "Model":          name,
        "Binary Acc":     f"{acc:.4f}",
        "Macro F1":       f"{f1_mac:.4f}",
        "Q-Level Acc":    f"{q_acc:.4f}",
        "Inf. Time (s)":  f"{inf_time:.2f}",
    }


def save_cm_plot(name, model, X_val, y_val):
    """Save a confusion matrix PNG to data/processed/."""
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Incorrect (0)", "Correct (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = DATA_PROC / f"cm_{safe}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved plot: data/processed/cm_{safe}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUPERVISED MODELS
# ─────────────────────────────────────────────────────────────────────────────

def _header(title, note="full dataset"):
    print(f"\n{'='*60}\n  {title}  [{note}]\n{'='*60}")


def train_lr(X_train, y_train, X_val, y_val, val_exp):
    _header("Logistic Regression (LR)")
    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="saga",           # fastest solver for large sparse data
        class_weight="balanced", # compensates for 75/25 class imbalance
        n_jobs=-1,               # use all CPU cores
    )
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    result = evaluate_model("Logistic Regression", model, X_val, y_val, val_exp)
    save_cm_plot("Logistic Regression", model, X_val, y_val)
    joblib.dump(model, MODELS_DIR / "lr_model.pkl")
    print("  Saved: models/model_a/traditional/lr_model.pkl")
    return model, result


def train_svm(X_train, y_train, X_val, y_val, val_exp):
    _header("LinearSVC (SVM)")
    t0 = time.time()
    model = LinearSVC(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    result = evaluate_model("LinearSVC (SVM)", model, X_val, y_val, val_exp)
    save_cm_plot("LinearSVC (SVM)", model, X_val, y_val)
    joblib.dump(model, MODELS_DIR / "svm_model.pkl")
    print("  Saved: models/model_a/traditional/svm_model.pkl")
    return model, result


def train_nb(X_train, y_train, X_val, y_val, val_exp):
    _header("ComplementNB (Naive Bayes)")
    # ComplementNB needs non-negative features — all our features are >= 0 (ok)
    t0 = time.time()
    model = ComplementNB(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    result = evaluate_model("ComplementNB (NB)", model, X_val, y_val, val_exp)
    save_cm_plot("ComplementNB (NB)", model, X_val, y_val)
    joblib.dump(model, MODELS_DIR / "nb_model.pkl")
    print("  Saved: models/model_a/traditional/nb_model.pkl")
    return model, result


def train_rf(X_train, y_train, X_val, y_val, val_exp):
    _header("Random Forest", note=f"subset = {SUBSET_RF:,} rows")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_train.shape[0], min(SUBSET_RF, X_train.shape[0]), replace=False)
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train[idx], y_train[idx])
    print(f"  Trained in {time.time()-t0:.1f}s")
    result = evaluate_model("Random Forest", model, X_val, y_val, val_exp)
    save_cm_plot("Random Forest", model, X_val, y_val)
    joblib.dump(model, MODELS_DIR / "rf_model.pkl")
    print("  Saved: models/model_a/traditional/rf_model.pkl")
    return model, result


def train_xgb(X_train, y_train, X_val, y_val, val_exp):
    if not HAS_XGB:
        print("\n  [XGBoost] Skipped — xgboost not installed.")
        return None, None
    _header("XGBoost", note=f"subset = {SUBSET_XGB:,} rows")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_train.shape[0], min(SUBSET_XGB, X_train.shape[0]), replace=False)
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train[idx], y_train[idx])
    print(f"  Trained in {time.time()-t0:.1f}s")
    result = evaluate_model("XGBoost", model, X_val, y_val, val_exp)
    save_cm_plot("XGBoost", model, X_val, y_val)
    joblib.dump(model, MODELS_DIR / "xgb_model.pkl")
    print("  Saved: models/model_a/traditional/xgb_model.pkl")
    return model, result


# ─────────────────────────────────────────────────────────────────────────────
# 4. UNSUPERVISED MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_kmeans(X_train, y_train):
    _header("K-Means Clustering", note=f"subset = {SUBSET_CLUSTER:,} rows, n_clusters={N_CLUSTERS}")
    print("  Note: n_clusters=4 mirrors the 4 answer classes A/B/C/D.")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_train.shape[0], min(SUBSET_CLUSTER, X_train.shape[0]), replace=False)
    X_sub = X_train[idx]
    y_sub = y_train[idx]

    t0 = time.time()
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_sub)
    print(f"  Trained in {time.time()-t0:.1f}s")

    sil = silhouette_score(X_sub, labels, sample_size=2000, random_state=RANDOM_STATE)

    # Purity: for each cluster, count the majority class
    purity_count = 0
    for c in range(N_CLUSTERS):
        mask = labels == c
        if mask.sum() > 0:
            purity_count += np.bincount(y_sub[mask]).max()
    purity = purity_count / len(y_sub)

    print(f"  Silhouette Score  : {sil:.4f}  (range -1 to 1; > 0.1 is reasonable)")
    print(f"  Cluster Purity    : {purity:.4f}  (fraction assigned to majority class)")
    print("  (Include these numbers in your report under the Unsupervised section)")

    joblib.dump(km, MODELS_DIR / "kmeans_model.pkl")
    print("  Saved: models/model_a/traditional/kmeans_model.pkl")
    return km, {
        "Model": "K-Means",
        "Silhouette": f"{sil:.4f}",
        "Purity":     f"{purity:.4f}",
        "Extra":      "N/A",
    }


def train_gmm(X_train, y_train):
    _header("Gaussian Mixture Model (GMM)",
            note=f"subset = {SUBSET_CLUSTER:,} rows, SVD({SVD_DIM}), n_components={N_CLUSTERS}")
    print("  Note: GMM needs dense input — reducing dims with TruncatedSVD first.")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_train.shape[0], min(SUBSET_CLUSTER, X_train.shape[0]), replace=False)
    X_sub = X_train[idx]
    y_sub = y_train[idx]

    svd = TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_STATE)
    X_dense = svd.fit_transform(X_sub)
    print(f"  Explained variance ratio (SVD): {svd.explained_variance_ratio_.sum():.4f}")

    t0 = time.time()
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="diag",   # faster than 'full' for high-dim data
        random_state=RANDOM_STATE,
        max_iter=200,
    )
    labels = gmm.fit_predict(X_dense)
    print(f"  Trained in {time.time()-t0:.1f}s")

    bic = gmm.bic(X_dense)
    sil = silhouette_score(X_dense, labels, sample_size=2000, random_state=RANDOM_STATE)
    purity_count = 0
    for c in range(N_CLUSTERS):
        mask = labels == c
        if mask.sum() > 0:
            purity_count += np.bincount(y_sub[mask]).max()
    purity = purity_count / len(y_sub)

    print(f"  BIC Score         : {bic:.1f}  (lower = better fit; compare between models)")
    print(f"  Silhouette Score  : {sil:.4f}")
    print(f"  Cluster Purity    : {purity:.4f}")

    joblib.dump(gmm, MODELS_DIR / "gmm_model.pkl")
    joblib.dump(svd, MODELS_DIR / "gmm_svd.pkl")
    print("  Saved: models/model_a/traditional/gmm_model.pkl  +  gmm_svd.pkl")
    return gmm, {
        "Model": "GMM",
        "Silhouette": f"{sil:.4f}",
        "Purity":     f"{purity:.4f}",
        "Extra":      f"BIC={bic:.0f}",
    }


def train_label_propagation(X_train, y_train):
    _header("Label Propagation (Semi-Supervised)",
            note=f"subset = {SUBSET_LP:,} rows, {LP_LABELED_FRAC*100:.0f}% labeled")
    print("  Note: requires dense input — using TruncatedSVD first.")

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_train.shape[0], min(SUBSET_LP, X_train.shape[0]), replace=False)
    X_sub = X_train[idx]
    y_sub = y_train[idx].copy()

    svd = TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_STATE)
    X_dense = svd.fit_transform(X_sub)

    # Mark the majority of rows as unlabeled (label = -1)
    n_labeled = int(SUBSET_LP * LP_LABELED_FRAC)
    y_semi = y_sub.copy().astype(int)
    unlabeled_idx = rng.choice(SUBSET_LP, SUBSET_LP - n_labeled, replace=False)
    y_semi[unlabeled_idx] = -1

    labeled_mask = y_semi != -1
    print(f"  Labeled  : {labeled_mask.sum():,}  |  Unlabeled: {(~labeled_mask).sum():,}")

    t0 = time.time()
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=500)
    lp.fit(X_dense, y_semi)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Evaluate on the labeled samples only
    y_pred_lp = lp.predict(X_dense[labeled_mask])
    f1_semi   = f1_score(y_sub[labeled_mask], y_pred_lp, average="macro")
    acc_semi  = accuracy_score(y_sub[labeled_mask], y_pred_lp)

    # Compare against a fully supervised model trained on the same labeled subset
    from sklearn.linear_model import LogisticRegression as _LR
    lr_ref = _LR(max_iter=300, solver="saga", class_weight="balanced", n_jobs=-1)
    lr_ref.fit(X_dense[labeled_mask], y_sub[labeled_mask])
    f1_sup = f1_score(y_sub[labeled_mask], lr_ref.predict(X_dense[labeled_mask]), average="macro")

    print(f"  Semi-supervised Macro F1  : {f1_semi:.4f}")
    print(f"  Supervised baseline F1    : {f1_sup:.4f}  (LR trained on same {n_labeled} labeled rows)")
    delta = f1_semi - f1_sup
    print(f"  Delta (semi - sup)        : {delta:+.4f}  "
          f"({'semi-supervised helps' if delta > 0 else 'supervised is better here'})")

    joblib.dump(lp,  MODELS_DIR / "label_prop_model.pkl")
    joblib.dump(svd, MODELS_DIR / "label_prop_svd.pkl")
    print("  Saved: models/model_a/traditional/label_prop_model.pkl  +  label_prop_svd.pkl")
    return lp, {
        "Model": "LabelPropagation",
        "Silhouette": "N/A",
        "Purity":     "N/A",
        "Extra":      f"Semi-F1={f1_semi:.4f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

class SoftVotingEnsemble:
    """
    Wraps two already-trained models and averages their predicted probabilities.
    No retraining needed — this reuses lr_model and nb_model as-is.
    """
    def __init__(self, models, names):
        self.models = models
        self.names  = names

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models])
        return probs.mean(axis=0)   # average probabilities across models

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build_ensemble(lr_model, nb_model, X_val, y_val, val_exp):
    _header("Soft-Voting Ensemble (LR + NB)", note="reuses trained models — no retraining")
    ensemble = SoftVotingEnsemble(
        models=[lr_model, nb_model],
        names=["LR", "NB"],
    )
    result = evaluate_model("Ensemble (LR + NB)", ensemble, X_val, y_val, val_exp)
    save_cm_plot("Ensemble (LR+NB)", ensemble, X_val, y_val)
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.pkl")
    print("  Saved: models/model_a/traditional/ensemble_model.pkl")
    return ensemble, result


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(sup_results, unsup_results):
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON TABLE  —  copy this into your final report!")
    print("=" * 72)

    W = [27, 12, 10, 13, 12]
    header  = ["Model", "Binary Acc", "Macro F1", "Q-Level Acc", "Inf. Time(s)"]
    row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in W)

    print(f"\n  Supervised Models (trained on val set):")
    print(row_fmt.format(*header))
    print("  " + "-" * 68)
    for r in sup_results:
        if r is None:
            continue
        print(row_fmt.format(
            r["Model"], r["Binary Acc"], r["Macro F1"],
            r["Q-Level Acc"], r["Inf. Time (s)"],
        ))

    print(f"\n  Unsupervised / Semi-supervised Models:")
    print(f"  {'Model':<27}  {'Silhouette':>12}  {'Purity':>10}  {'Extra (BIC / F1)':>18}")
    print("  " + "-" * 72)
    for r in unsup_results:
        if r is None:
            continue
        print(f"  {r['Model']:<27}  {r['Silhouette']:>12}  {r['Purity']:>10}  {r['Extra']:>18}")

    print("=" * 72)
    print("\n  Key thresholds from your rubric:")
    print("    Binary accuracy > 0.60  (random baseline = 0.25 per-option)")
    print("    Macro F1        > 0.50")
    print("    Q-Level Acc     > 0.40  (random baseline = 0.25)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model A — Training Pipeline")
    print("  RACE Dataset  |  Classical ML Only  |  No Neural Networks")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6]  Loading processed data ...")
    train_exp   = pd.read_csv(DATA_PROC / "train_expanded.csv")
    val_exp     = pd.read_csv(DATA_PROC / "val_expanded.csv")
    test_exp    = pd.read_csv(DATA_PROC / "test_expanded.csv")
    train_clean = pd.read_csv(DATA_PROC / "train_clean.csv", index_col=0)
    val_clean   = pd.read_csv(DATA_PROC / "val_clean.csv",   index_col=0)
    test_clean  = pd.read_csv(DATA_PROC / "test_clean.csv",  index_col=0)

    y_train = train_exp["is_correct"].values
    y_val   = val_exp["is_correct"].values

    print(f"  Train expanded : {len(train_exp):,} rows  |  Positive rate: {y_train.mean():.4f}")
    print(f"  Val   expanded : {len(val_exp):,} rows  |  Positive rate: {y_val.mean():.4f}")

    # ── Fit OHE vectorizer (training data only) ───────────────────────────
    print("\n[2/6]  Building OHE feature matrix ...")
    ohe, X_train_ohe = fit_ohe_vectorizer(train_exp)

    # ── Build full feature matrices ───────────────────────────────────────
    print("\n[3/6]  Adding cosine similarity + numerical features ...")
    print("  (Computing cosine similarity between article and each option ...)")
    t0 = time.time()
    X_train = build_all_features(ohe, train_exp, train_clean)
    X_val   = build_all_features(ohe, val_exp,   val_clean)
    X_test  = build_all_features(ohe, test_exp,  test_clean)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  X_train shape : {X_train.shape}  (OHE + cosine_sim + 2 numerical = {X_train.shape[1]} features)")

    # Save val features for evaluate.py
    sp.save_npz(DATA_PROC / "X_val.npz", X_val)
    np.save(DATA_PROC / "y_val.npy", y_val)
    sp.save_npz(DATA_PROC / "X_test.npz", X_test)
    np.save(DATA_PROC / "y_test.npy", test_exp["is_correct"].values)
    print("  Saved X_val.npz + y_val.npy + X_test.npz + y_test.npy")

    # ── Supervised training ───────────────────────────────────────────────
    print("\n[4/6]  Training supervised models ...")
    sup_results = []

    lr_model,  lr_r   = train_lr(X_train, y_train, X_val, y_val, val_exp)
    svm_model, svm_r  = train_svm(X_train, y_train, X_val, y_val, val_exp)
    nb_model,  nb_r   = train_nb(X_train, y_train, X_val, y_val, val_exp)
    rf_model,  rf_r   = train_rf(X_train, y_train, X_val, y_val, val_exp)
    xgb_model, xgb_r  = train_xgb(X_train, y_train, X_val, y_val, val_exp)

    for r in [lr_r, svm_r, nb_r, rf_r, xgb_r]:
        if r is not None:
            sup_results.append(r)

    # ── Unsupervised + semi-supervised ────────────────────────────────────
    print("\n[5/6]  Training unsupervised / semi-supervised models ...")
    unsup_results = []

    km_model,  km_r  = train_kmeans(X_train, y_train)
    gmm_model, gmm_r = train_gmm(X_train, y_train)
    lp_model,  lp_r  = train_label_propagation(X_train, y_train)

    for r in [km_r, gmm_r, lp_r]:
        if r is not None:
            unsup_results.append(r)

    # ── Ensemble ──────────────────────────────────────────────────────────
    print("\n[6/6]  Building ensemble ...")
    ens_model, ens_r = build_ensemble(lr_model, nb_model, X_val, y_val, val_exp)
    sup_results.append(ens_r)

    # ── Summary ───────────────────────────────────────────────────────────
    print_table(sup_results, unsup_results)

    print("\n  All models saved to: models/model_a/traditional/")
    print("  Next step: run  python src/model_b_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

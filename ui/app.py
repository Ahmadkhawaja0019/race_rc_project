"""
ui/app.py  —  RACE Reading Comprehension & Quiz Generator  (Streamlit UI)

Run from the project root with your venv active:
    streamlit run ui/app.py

Prerequisite: run model_a_train.py and model_b_train.py first so all model
files exist in models/.

Four screens
------------
  Screen 1 — Article Input     : paste an article, pick a question, submit
  Screen 2 — Quiz View         : see the 4 options; Model A ranks them and
                                  highlights the predicted best answer
  Screen 3 — Hints Panel       : 3 graduated hints (Model B) + distractor list
  Screen 4 — Developer Dashboard : model comparison table, confusion matrix
                                   images, evaluation CSV download
"""

# CRITICAL: st.set_page_config() MUST be the very first Streamlit call
import streamlit as st
st.set_page_config(
    page_title="RACE RC System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --rc-muted: #9aa0a6;
      --rc-border: rgba(127, 127, 127, 0.4);
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
      background-color: var(--secondary-background-color) !important;
      color: var(--text-color) !important;
      border-color: var(--rc-border) !important;
    }

    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stTextInput"] input::placeholder {
      color: var(--rc-muted) !important;
    }

    .rc-article,
    .rc-option,
    .rc-hint,
    .rc-pill {
      color: var(--text-color);
    }

    .rc-article {
      background: var(--secondary-background-color);
      border-left: 4px solid var(--primary-color);
      border-radius: 4px;
      padding: 12px 16px;
      font-size: 0.95em;
    }

    .rc-option {
      background: var(--secondary-background-color);
      border: 1px solid var(--rc-border);
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 10px;
    }

    .rc-option--top {
      border: 3px solid #4caf50;
      background: rgba(76, 175, 80, 0.12);
    }

    .rc-score {
      color: var(--rc-muted);
      font-size: 0.85em;
      white-space: nowrap;
      margin-left: 12px;
    }

    .rc-bar-track {
      background: rgba(127, 127, 127, 0.35);
      border-radius: 4px;
      height: 6px;
      margin-top: 8px;
    }

    .rc-bar-fill {
      background: #4caf50;
      height: 6px;
      border-radius: 4px;
    }

    .rc-hint {
      background: var(--secondary-background-color);
      border-left: 5px solid var(--primary-color);
      padding: 12px 16px;
      border-radius: 4px;
      margin-bottom: 14px;
    }

    .rc-hint--1 {
      border-left-color: #ff9800;
      background: rgba(255, 152, 0, 0.12);
    }

    .rc-hint--2 {
      border-left-color: #2196f3;
      background: rgba(33, 150, 243, 0.12);
    }

    .rc-hint--3 {
      border-left-color: #4caf50;
      background: rgba(76, 175, 80, 0.12);
    }

    .rc-hint-title {
      font-weight: bold;
      margin-bottom: 4px;
    }

    .rc-hint--1 .rc-hint-title { color: #ffb74d; }
    .rc-hint--2 .rc-hint-title { color: #64b5f6; }
    .rc-hint--3 .rc-hint-title { color: #81c784; }

    .rc-pill {
      background: rgba(233, 30, 99, 0.12);
      border: 1px solid #e91e63;
      border-radius: 20px;
      padding: 8px 16px;
      text-align: center;
      font-weight: bold;
      color: #ff9ac1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import sys
import re
import time
from pathlib import Path

# ── Dynamic project-root detection ────────────────────────────────────────────
# Works whether you launch from the project root, from ui/, or from the worktree.
def _find_root():
    here = Path(__file__).resolve().parent
    for candidate in [here.parent, here, here.parent.parent]:
        if (candidate / "data" / "raw" / "train.csv").exists():
            return candidate
    # Fallback: look two more levels up
    current = here
    for _ in range(5):
        current = current.parent
        if (current / "data" / "raw" / "train.csv").exists():
            return current
    return here.parent   # best guess

PROJECT_ROOT = _find_root()
MODELS_A = PROJECT_ROOT / "models" / "model_a" / "traditional"
MODELS_B = PROJECT_ROOT / "models" / "model_b" / "traditional"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Lazy imports (so Streamlit error page shows if packages missing) ───────────
try:
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import joblib
    from sklearn.preprocessing import normalize
except ImportError as e:
    st.error(f"Missing package: {e}\n\nActivate your venv and run: pip install -r requirements.txt")
    st.stop()

# ── Pickle compatibility: SoftVotingEnsemble must be importable here ──────────
# ensemble_model.pkl was saved when this class lived in model_a_train.py.
# Defining it here (at module level) lets joblib.load resolve the class correctly.
class SoftVotingEnsemble:
    """Averages predicted probabilities from LR and NB — no retraining needed."""
    def __init__(self, models, names):
        self.models = models
        self.names  = names

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models])
        return probs.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ── Minimal text cleaner (same logic as preprocessing.py) ─────────────────────
def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached — loads only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Model A (answer ranking)...")
def load_model_a():
    """Load OHE vectorizer + best classifier (LR preferred, fallback to others)."""
    ohe_path = MODELS_A / "ohe_vectorizer.pkl"
    if not ohe_path.exists():
        return None, None, "OHE vectorizer not found. Run model_a_train.py first."

    ohe = joblib.load(ohe_path)

    for fname, label in [
        ("lr_model.pkl",       "Logistic Regression"),
        ("ensemble_model.pkl", "Ensemble (LR+NB)"),
        ("svm_model.pkl",      "LinearSVC"),
        ("nb_model.pkl",       "ComplementNB"),
    ]:
        path = MODELS_A / fname
        if path.exists():
            model = joblib.load(path)
            return ohe, model, label

    return ohe, None, "No classifier found. Run model_a_train.py."


@st.cache_resource(show_spinner="Loading Model B (hints & distractors)...")
def load_model_b():
    """Load OHE-B vectorizer and Word2Vec model for distractor generation."""
    result = {}
    ohe_b_path = MODELS_B / "ohe_vectorizer_b.pkl"
    w2v_path   = MODELS_B / "word2vec.model"
    freq_path  = MODELS_B / "word_freq.pkl"

    if ohe_b_path.exists():
        result["ohe_b"] = joblib.load(ohe_b_path)
    if freq_path.exists():
        result["freq"]  = joblib.load(freq_path)
    if w2v_path.exists():
        try:
            from gensim.models import Word2Vec
            result["w2v"] = Word2Vec.load(str(w2v_path))
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rank_options(ohe, model, article: str, question: str, options: dict) -> dict:
    """
    Given article + question + 4 options, return a dict of option → confidence score.
    Higher score = model thinks this option is more likely to be correct.

    IMPORTANT: builds the SAME 3-part feature matrix as model_a_train.py:
      OHE (10,000) + cosine similarity (1) + numeric length features (2) = 10,003 cols.
    Using only ohe.transform() gives 10,000 features and causes a shape mismatch.
    """
    art_c = _clean(article)
    q_c   = _clean(question)
    keys  = list(options.keys())   # ["A", "B", "C", "D"]

    # ── Part 1: OHE on the combined text (article + question + option) ─────────
    texts = [art_c + " " + q_c + " " + _clean(options[k]) for k in keys]
    X_ohe = ohe.transform(texts)   # sparse (4, 10000)

    # ── Part 2: cosine similarity between article and each option ──────────────
    art_vec = ohe.transform([art_c])             # sparse (1, 10000)
    art_n   = normalize(art_vec, norm="l2")
    cos_sims = []
    for k in keys:
        opt_vec = ohe.transform([_clean(options[k])])   # sparse (1, 10000)
        opt_n   = normalize(opt_vec, norm="l2")
        sim     = float(art_n.multiply(opt_n).sum())    # dot product of normed vecs
        cos_sims.append(sim)
    X_cos = sp.csr_matrix(np.array(cos_sims, dtype=float).reshape(-1, 1))  # (4, 1)

    # ── Part 3: numeric length features (article_length, q_length) ────────────
    art_len = len(article.split())
    q_len   = len(question.split())
    X_num   = sp.csr_matrix(
        np.array([[art_len, q_len]] * 4, dtype=float)   # same values for all 4 rows
    )   # (4, 2)

    # ── Stack → (4, 10003), matching the training feature matrix ──────────────
    X = sp.hstack([X_ohe, X_cos, X_num], format="csr")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw   = model.decision_function(X)
        probs = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    else:
        probs = model.predict(X).astype(float)

    return {k: float(p) for k, p in zip(keys, probs)}


def get_hints(article: str, question: str) -> dict:
    """
    Extract 3 graduated hints by scoring article sentences on keyword overlap
    with the question.  Does NOT need any trained model.
    """
    import re as _re
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "of", "and", "or", "but", "for", "with", "that", "this",
        "it", "its", "he", "she", "they", "we", "you", "i", "be", "been",
        "has", "have", "had", "do", "does", "did",
    }
    q_words = set(_clean(question).split()) - stop
    sentences = _re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if not sentences:
        return {"Hint 1": "No article text available.",
                "Hint 2": "No article text available.",
                "Hint 3": "No article text available."}

    scored = []
    for sent in sentences:
        words = set(_clean(sent).split())
        overlap = len(q_words & words)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = [s for _, s in scored[:3]]

    while len(top3) < 3:
        top3.append(top3[-1] if top3 else "—")

    return {
        "Hint 1 (least revealing)": top3[2],
        "Hint 2 (moderate)":        top3[1],
        "Hint 3 (most revealing)":  top3[0],
    }


def gen_distractors(article: str, correct_answer: str, model_b: dict) -> list:
    """
    Generate 3 plausible wrong-answer distractors using available Model B artifacts.
    Falls back gracefully if W2V or OHE-B is missing.
    """
    art_c   = _clean(article)
    ans_c   = _clean(correct_answer)
    ans_words = set(ans_c.split())
    art_words = [w for w in art_c.split() if len(w) >= 3 and w not in ans_words]

    distractors = []

    # Method 1: OHE cosine similarity — medium-similarity article words
    if "ohe_b" in model_b and art_words:
        try:
            from sklearn.metrics.pairwise import cosine_similarity as _cos
            ohe_b = model_b["ohe_b"]
            ans_vec  = ohe_b.transform([ans_c])
            cand_vecs = ohe_b.transform(art_words)
            sims = _cos(ans_vec, cand_vecs)[0]
            ranked = sorted(zip(art_words, sims), key=lambda x: x[1], reverse=True)
            # skip top 3 most similar, take next 3
            distractors += [w for w, _ in ranked[3:6]]
        except Exception:
            pass

    # Method 2: Word2Vec — semantically similar words not in article
    if len(distractors) < 3 and "w2v" in model_b:
        try:
            w2v = model_b["w2v"]
            for aw in list(ans_words)[:3]:
                if aw in w2v.wv:
                    similar = w2v.wv.most_similar(aw, topn=20)
                    for word, _ in similar:
                        if (word not in art_c and word not in ans_words
                                and len(word) >= 3 and word not in distractors):
                            distractors.append(word)
                            break
        except Exception:
            pass

    # Method 3: Frequency-based fallback
    if len(distractors) < 3 and "freq" in model_b:
        try:
            freq = model_b["freq"]
            # Find frequency of first answer word
            first_word = list(ans_words)[0] if ans_words else ""
            target_freq = freq.get(first_word, 5)
            lo, hi = target_freq * 0.6, target_freq * 1.6
            cands = [w for w, cnt in freq.items()
                     if lo <= cnt <= hi and w not in ans_words
                     and w not in distractors and len(w) >= 3][:10]
            distractors += cands[:3 - len(distractors)]
        except Exception:
            pass

    # Simple fallback: grab random article words
    if len(distractors) < 3:
        seen = set(distractors) | ans_words
        for w in art_words:
            if w not in seen and len(w) >= 4:
                distractors.append(w)
                seen.add(w)
            if len(distractors) >= 3:
                break

    return list(dict.fromkeys(distractors))[:3]   # deduplicate, cap at 3


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  (navigation)
# ─────────────────────────────────────────────────────────────────────────────

SCREENS = {
    "📝 Article Input":       "input",
    "❓ Quiz View":            "quiz",
    "💡 Hints Panel":          "hints",
    "📊 Developer Dashboard":  "dashboard",
}

with st.sidebar:
    st.title("📚 RACE RC System")
    st.caption("AL2002 AI Lab Project — Spring 2026")
    st.markdown("---")
    screen_label = st.radio("Navigate", list(SCREENS.keys()))
    screen = SCREENS[screen_label]

    st.markdown("---")
    st.markdown("**Models loaded:**")
    ohe_a, model_a, model_a_label = load_model_a()
    model_b = load_model_b()

    if model_a is not None:
        st.success(f"Model A: {model_a_label}")
    else:
        st.warning("Model A not loaded")

    b_parts = []
    if "ohe_b"  in model_b: b_parts.append("OHE-B")
    if "w2v"    in model_b: b_parts.append("W2V")
    if "freq"   in model_b: b_parts.append("Freq")
    if b_parts:
        st.success(f"Model B: {', '.join(b_parts)}")
    else:
        st.warning("Model B not loaded")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  (persists data between screen switches)
# ─────────────────────────────────────────────────────────────────────────────

for key, default in {
    "article":  "",
    "question": "",
    "opt_a":    "",
    "opt_b":    "",
    "opt_c":    "",
    "opt_d":    "",
    "scores":   None,
    "hints":    None,
    "distractors": None,
    "submitted":   False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 1 — ARTICLE INPUT
# ─────────────────────────────────────────────────────────────────────────────

if screen == "input":
    st.title("📝 Screen 1 — Article Input")
    st.markdown(
        "Paste a reading comprehension passage and a multiple-choice question below, "
        "then click **Analyse** to let the AI rank the answer options."
    )

    # ── Sample article (RACE-style) ────────────────────────────────────────────
    SAMPLE_ARTICLE = (
        "Scientists have discovered a new species of deep-sea fish living at depths "
        "of more than 8,000 metres in the Pacific Ocean. The fish, named Pseudoliparis "
        "swirei, belongs to the snailfish family and was found in the Mariana Trench. "
        "Researchers say the discovery highlights how little we know about life in the "
        "deepest parts of the ocean. The fish survive the crushing pressure by having "
        "a soft, gelatinous body that allows them to withstand conditions that would "
        "destroy most other organisms. Unlike many deep-sea creatures, they appear to "
        "be highly active and are believed to be top predators in their environment."
    )
    SAMPLE_QUESTION = "What allows the fish to survive the extreme pressure of the deep sea?"
    SAMPLE_OPTS = {
        "A": "A hard shell around their body",
        "B": "A soft, gelatinous body structure",
        "C": "Special pressure-resistant bones",
        "D": "They migrate to shallower water periodically",
    }

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Load sample article", use_container_width=True):
            st.session_state.article  = SAMPLE_ARTICLE
            st.session_state.question = SAMPLE_QUESTION
            st.session_state.opt_a    = SAMPLE_OPTS["A"]
            st.session_state.opt_b    = SAMPLE_OPTS["B"]
            st.session_state.opt_c    = SAMPLE_OPTS["C"]
            st.session_state.opt_d    = SAMPLE_OPTS["D"]
            st.session_state.submitted = False

    with st.form("input_form"):
        article = st.text_area(
            "Article / Passage",
            value=st.session_state.article,
            height=220,
            placeholder="Paste the reading passage here...",
        )
        question = st.text_input(
            "Question",
            value=st.session_state.question,
            placeholder="Type or paste the multiple-choice question here...",
        )
        st.markdown("**Answer Options**")
        c1, c2 = st.columns(2)
        with c1:
            opt_a = st.text_input("A:", value=st.session_state.opt_a)
            opt_c = st.text_input("C:", value=st.session_state.opt_c)
        with c2:
            opt_b = st.text_input("B:", value=st.session_state.opt_b)
            opt_d = st.text_input("D:", value=st.session_state.opt_d)

        submitted = st.form_submit_button("Analyse", use_container_width=True, type="primary")

    if submitted:
        # Validate
        errors = []
        if not article.strip():
            errors.append("Article cannot be empty.")
        if not question.strip():
            errors.append("Question cannot be empty.")
        if not all([opt_a.strip(), opt_b.strip(), opt_c.strip(), opt_d.strip()]):
            errors.append("All four answer options (A, B, C, D) must be filled in.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # Save to session state
            st.session_state.article  = article
            st.session_state.question = question
            st.session_state.opt_a    = opt_a
            st.session_state.opt_b    = opt_b
            st.session_state.opt_c    = opt_c
            st.session_state.opt_d    = opt_d
            st.session_state.submitted = True

            options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}

            with st.spinner("Running Model A (answer ranking)..."):
                if ohe_a is not None and model_a is not None:
                    st.session_state.scores = rank_options(ohe_a, model_a, article, question, options)
                else:
                    # Fallback: random scores so UI still works without trained models
                    rng = np.random.default_rng(42)
                    st.session_state.scores = {k: float(rng.random()) for k in options}

            with st.spinner("Extracting hints (Model B)..."):
                st.session_state.hints = get_hints(article, question)

            with st.spinner("Generating distractors (Model B)..."):
                # Use the predicted best answer to generate alternative distractors
                best_opt = max(st.session_state.scores, key=st.session_state.scores.get)
                best_text = options[best_opt]
                st.session_state.distractors = gen_distractors(article, best_text, model_b)

            st.success("Analysis complete! Switch to 'Quiz View' in the sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 2 — QUIZ VIEW
# ─────────────────────────────────────────────────────────────────────────────

elif screen == "quiz":
    st.title("❓ Screen 2 — Quiz View")

    if not st.session_state.submitted or st.session_state.scores is None:
        st.info("No analysis yet. Go to **Article Input** and click Analyse first.")
        st.stop()

    st.markdown("### Article")
    st.markdown(
        f'<div class="rc-article">{st.session_state.article}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(f"### Question\n**{st.session_state.question}**")
    st.markdown("---")

    scores = st.session_state.scores
    best   = max(scores, key=scores.get)

    option_texts = {
        "A": st.session_state.opt_a,
        "B": st.session_state.opt_b,
        "C": st.session_state.opt_c,
        "D": st.session_state.opt_d,
    }

    st.markdown("### Answer Options")
    st.caption(f"Model A ({model_a_label}) confidence scores — highest = predicted correct answer")

    for opt, text in option_texts.items():
        score  = scores.get(opt, 0.0)
        is_top = opt == best

        icon      = "✅" if is_top else "⬜"
        bar_width = int(score * 100)
        opt_class = "rc-option rc-option--top" if is_top else "rc-option"

        st.markdown(
            f"""
            <div class="{opt_class}">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:{'bold' if is_top else 'normal'};font-size:1em;">
                  {icon} <b>{opt}.</b> {text}
                </span>
                <span class="rc-score">score: {score:.3f}</span>
              </div>
              <div class="rc-bar-track">
                <div class="rc-bar-fill" style="width:{bar_width}%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"**Model A prediction:** Option **{best}** — "
        f'*"{option_texts[best]}"* (confidence: {scores[best]:.3f})'
    )
    st.caption(
        "Note: Model A is trained on whether each option is correct (binary classification). "
        "The option with the highest confidence score is predicted as the correct answer."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 3 — HINTS PANEL
# ─────────────────────────────────────────────────────────────────────────────

elif screen == "hints":
    st.title("💡 Screen 3 — Hints Panel")

    if not st.session_state.submitted or st.session_state.hints is None:
        st.info("No analysis yet. Go to **Article Input** and click Analyse first.")
        st.stop()

    st.markdown(f"**Question:** {st.session_state.question}")
    st.markdown("---")

    # ── Graduated Hints ────────────────────────────────────────────────────────
    st.markdown("### Graduated Hints")
    st.caption(
        "Sentences from the article ranked by relevance to the question. "
        "Hint 1 is least revealing; Hint 3 gives the most direct clue."
    )

    HINT_CLASSES = {
        "Hint 1 (least revealing)": "rc-hint--1",
        "Hint 2 (moderate)":        "rc-hint--2",
        "Hint 3 (most revealing)":  "rc-hint--3",
    }

    hints = st.session_state.hints
    for label, hint_text in hints.items():
        hint_class = HINT_CLASSES.get(label, "")
        st.markdown(
            f"""
            <div class="rc-hint {hint_class}">
              <div class="rc-hint-title">{label}</div>
              <div style="font-size:0.95em;">{hint_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Distractors ────────────────────────────────────────────────────────────
    st.markdown("### Generated Distractors")
    st.caption(
        "Plausible-but-wrong alternative keywords generated by Model B "
        "(OHE cosine similarity + Word2Vec + frequency matching)."
    )

    distractors = st.session_state.distractors or []
    if distractors:
        cols = st.columns(len(distractors))
        for i, d in enumerate(distractors):
            with cols[i]:
                st.markdown(
                    f'<div class="rc-pill">{d}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No distractors generated — check that Model B files are present.")

    st.markdown("---")
    st.markdown(
        "**How distractors are generated:** Model B uses three methods — "
        "(1) OHE cosine similarity to find medium-similarity article words, "
        "(2) Word2Vec to find semantically related words outside the article, "
        "(3) frequency matching to find corpus words with similar occurrence rates. "
        "The final distractors are deduplicated results from all three methods."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 4 — DEVELOPER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

elif screen == "dashboard":
    st.title("📊 Screen 4 — Developer Dashboard")
    st.caption("Model performance metrics and evaluation artifacts from evaluate.py")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab_results, tab_cm, tab_gs, tab_about = st.tabs([
        "Model Comparison",
        "Confusion Matrices",
        "GridSearchCV",
        "About",
    ])

    # ── Tab 1: Model comparison table ─────────────────────────────────────────
    with tab_results:
        st.subheader("Model Comparison — Test Set")
        csv_path = DATA_PROC / "evaluation_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Highlight best value per numeric column
            numeric_cols = [c for c in df.columns
                            if c not in ("Model", "Split")]
            st.dataframe(
                df,
                use_container_width=True,
                height=350,
            )
            st.download_button(
                label="Download results CSV",
                data=csv_path.read_bytes(),
                file_name="evaluation_results.csv",
                mime="text/csv",
            )

            # Bar chart of Q-Level Accuracy
            if "Q-Level Acc" in df.columns:
                st.subheader("Question-Level Accuracy by Model")
                chart_df = df.set_index("Model")[["Q-Level Acc"]].sort_values(
                    "Q-Level Acc", ascending=False
                )
                st.bar_chart(chart_df)
                st.caption("Random baseline = 0.25 | Target = 0.40+")

            if "Macro F1" in df.columns:
                st.subheader("Macro F1 by Model")
                chart_df2 = df.set_index("Model")[["Macro F1"]].sort_values(
                    "Macro F1", ascending=False
                )
                st.bar_chart(chart_df2)
        else:
            st.info(
                "evaluation_results.csv not found. "
                "Run `python src/evaluate.py` first."
            )

    # ── Tab 2: Confusion matrices ──────────────────────────────────────────────
    with tab_cm:
        st.subheader("Confusion Matrices (Test Set)")

        cm_files = sorted(DATA_PROC.glob("cm_test_*.png"))
        if cm_files:
            cols_per_row = 2
            for i in range(0, len(cm_files), cols_per_row):
                row_files = cm_files[i:i + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, fpath in zip(cols, row_files):
                    with col:
                        label = fpath.stem.replace("cm_test_", "").replace("_", " ").title()
                        st.markdown(f"**{label}**")
                        st.image(str(fpath), use_column_width=True)
        else:
            st.info(
                "No confusion matrix images found in data/processed/. "
                "Run `python src/evaluate.py` to generate them."
            )

    # ── Tab 3: GridSearchCV ────────────────────────────────────────────────────
    with tab_gs:
        st.subheader("GridSearchCV — LR Hyperparameter Tuning")
        gs_csv  = DATA_PROC / "gridsearch_results.csv"
        gs_heat = DATA_PROC / "gridsearch_heatmap.png"

        if gs_csv.exists():
            gs_df = pd.read_csv(gs_csv)
            st.dataframe(gs_df, use_container_width=True)

            if gs_heat.exists():
                st.image(str(gs_heat), caption="GridSearchCV Heatmap (C vs max_features)", width=600)
        else:
            st.info(
                "gridsearch_results.csv not found. "
                "Run `python src/evaluate.py` to generate it."
            )

        st.markdown(
            "**What is GridSearchCV?**  \n"
            "It systematically tests every combination of hyperparameters and uses "
            "cross-validation (3-fold here) to estimate which combination will generalise "
            "best to unseen data.  For Logistic Regression we tune:  \n"
            "- `C` — regularisation strength (smaller = more regularised)  \n"
            "- `max_features` — vocabulary size for the OHE CountVectorizer"
        )

    # ── Tab 4: About ───────────────────────────────────────────────────────────
    with tab_about:
        st.subheader("Project Information")
        st.markdown(
            """
| Field | Detail |
|-------|--------|
| **Project** | RACE Reading Comprehension & Quiz Generation System |
| **Course** | AL2002 — Artificial Intelligence Lab, Spring 2026 |
| **University** | NUCES (FAST), Islamabad |
| **Dataset** | RACE — 87,866 training articles |
| **Model A task** | Answer option ranking (binary is_correct classification) |
| **Model B task** | Distractor generation + hint extraction |
| **Features** | OHE (CountVectorizer binary=True) + cosine similarity + numerical |
| **Allowed models** | LR, SVM, Naive Bayes, Random Forest, XGBoost, KMeans, GMM, LabelPropagation |
            """
        )

        st.markdown("### How to run the full pipeline")
        st.code(
            """# 1. Activate virtual environment
venv\\Scripts\\activate

# 2. Preprocess data (creates data/processed/)
python src/preprocessing.py

# 3. Train Model A (creates models/model_a/)
python src/model_a_train.py

# 4. Train Model B (creates models/model_b/)
python src/model_b_train.py

# 5. Evaluate all models (creates evaluation_results.csv)
python src/evaluate.py

# 6. Launch this Streamlit app
streamlit run ui/app.py""",
            language="bash",
        )

        st.markdown("### Model files")
        for folder, label in [
            (MODELS_A, "Model A (Answer Ranking)"),
            (MODELS_B, "Model B (Distractors & Hints)"),
        ]:
            st.markdown(f"**{label}** — `{folder.relative_to(PROJECT_ROOT)}`")
            if folder.exists():
                files = list(folder.glob("*.pkl")) + list(folder.glob("*.model")) + list(folder.glob("*.npz"))
                if files:
                    for f in sorted(files):
                        size_kb = f.stat().st_size / 1024
                        st.markdown(f"  - `{f.name}` ({size_kb:.0f} KB)")
                else:
                    st.caption("  (no model files yet — run the training scripts)")
            else:
                st.caption("  (folder not found — run the training scripts)")

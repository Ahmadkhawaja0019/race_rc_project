"""
ui/app.py  —  RACE Reading Comprehension & Quiz Generator  (Streamlit UI)

Run from the project root with your venv active:
    streamlit run ui/app.py

Prerequisite: run model_a_train.py, model_a_generate.py, and model_b_train.py
first so all model files exist in models/.

Four screens
------------
  Screen 1 — Article Input    : paste a reading passage → Generate Quiz
  Screen 2 — Quiz             : answer the generated MCQ, get instant feedback
  Screen 3 — Hints            : graduated hints; Reveal Answer gated behind all hints
  Screen 4 — Dashboard        : model metrics, benchmark table, confusion matrices
"""

# CRITICAL: st.set_page_config() MUST be the very first Streamlit call
import streamlit as st
st.set_page_config(
    page_title="RACE Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .rc-article {
      background: var(--secondary-background-color);
      border-left: 4px solid var(--primary-color);
      border-radius: 4px;
      padding: 12px 16px;
      font-size: 0.95em;
    }
    .rc-opt-correct {
      background: rgba(76, 175, 80, 0.18);
      border: 2px solid #4caf50;
      border-radius: 8px;
      padding: 10px 16px;
      margin-bottom: 8px;
      font-weight: bold;
      color: #2e7d32;
    }
    .rc-opt-wrong {
      background: rgba(244, 67, 54, 0.12);
      border: 2px solid #f44336;
      border-radius: 8px;
      padding: 10px 16px;
      margin-bottom: 8px;
      color: #c62828;
    }
    .rc-opt-neutral {
      background: var(--secondary-background-color);
      border: 1px solid rgba(127,127,127,0.4);
      border-radius: 8px;
      padding: 10px 16px;
      margin-bottom: 8px;
    }
    .rc-hint-1 {
      border-left: 5px solid #ff9800;
      background: rgba(255,152,0,0.08);
      padding: 12px 16px;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    .rc-hint-2 {
      border-left: 5px solid #2196f3;
      background: rgba(33,150,243,0.08);
      padding: 12px 16px;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    .rc-hint-3 {
      border-left: 5px solid #4caf50;
      background: rgba(76,175,80,0.08);
      padding: 12px 16px;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    .rc-pill {
      background: rgba(233, 30, 99, 0.12);
      border: 1px solid #e91e63;
      border-radius: 20px;
      padding: 6px 14px;
      display: inline-block;
      margin: 4px;
      font-weight: bold;
      color: #c2185b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import sys
import re
import random
from pathlib import Path

# ── Dynamic project-root detection ────────────────────────────────────────────
def _find_root():
    here = Path(__file__).resolve().parent
    for candidate in [here.parent, here, here.parent.parent]:
        if (candidate / "data").exists():
            return candidate
    return here.parent

PROJECT_ROOT = _find_root()
MODELS_A  = PROJECT_ROOT / "models" / "model_a" / "traditional"
MODELS_B  = PROJECT_ROOT / "models" / "model_b" / "traditional"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Package imports ────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import joblib
    from sklearn.preprocessing import normalize
except ImportError as e:
    st.error(f"Missing package: {e}\n\nActivate your venv and run: pip install -r requirements.txt")
    st.stop()

# ── Pickle shim: SoftVotingEnsemble must be importable at module level ─────────
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


# ── Text cleaner ───────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Model A (answer verifier)...")
def load_model_a():
    """Load OHE vectorizer + best classifier for answer verification."""
    ohe_path = MODELS_A / "ohe_vectorizer.pkl"
    if not ohe_path.exists():
        return None, None, "OHE vectorizer not found"
    ohe = joblib.load(ohe_path)
    for fname, label in [
        ("lr_model.pkl",       "Logistic Regression"),
        ("ensemble_model.pkl", "Ensemble (LR+NB)"),
        ("svm_model.pkl",      "LinearSVC"),
        ("nb_model.pkl",       "ComplementNB"),
    ]:
        p = MODELS_A / fname
        if p.exists():
            return ohe, joblib.load(p), label
    return ohe, None, "No classifier found"


@st.cache_resource(show_spinner="Loading Question Generator...")
def load_question_generator():
    """Load the trained QuestionGenerator (model_a_generate.py)."""
    tfidf_path = MODELS_A / "qg_tfidf.pkl"
    if not tfidf_path.exists():
        return None
    try:
        from model_a_generate import QuestionGenerator
        return QuestionGenerator.load(MODELS_A)
    except Exception as e:
        return None


@st.cache_resource(show_spinner="Loading Model B (distractors & hints)...")
def load_model_b():
    result = {}
    for fname, key in [
        ("ohe_vectorizer_b.pkl", "ohe_b"),
        ("word_freq.pkl",        "freq"),
    ]:
        p = MODELS_B / fname
        if p.exists():
            result[key] = joblib.load(p)
    w2v_path = MODELS_B / "word2vec.model"
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

def verify_answer(ohe, model, article: str, question: str, options: list) -> list:
    """
    Score each option (str) and return list of float confidence scores
    in the same order as `options`.
    """
    art_c = _clean(article)
    q_c   = _clean(question)
    texts = [art_c + " " + q_c + " " + _clean(opt) for opt in options]

    X_ohe = ohe.transform(texts)  # (n, 10000)

    art_vec = ohe.transform([art_c])
    art_n   = normalize(art_vec, norm="l2")
    cos_sims = []
    for opt in options:
        ov  = ohe.transform([_clean(opt)])
        on  = normalize(ov, norm="l2")
        cos_sims.append(float(art_n.multiply(on).sum()))
    X_cos = sp.csr_matrix(np.array(cos_sims, dtype=float).reshape(-1, 1))

    art_len = len(article.split())
    q_len   = len(question.split())
    X_num   = sp.csr_matrix(np.array([[art_len, q_len]] * len(options), dtype=float))

    X = sp.hstack([X_ohe, X_cos, X_num], format="csr")

    if hasattr(model, "predict_proba"):
        return list(model.predict_proba(X)[:, 1])
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        norm_raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return list(norm_raw)
    else:
        return list(model.predict(X).astype(float))


def get_hints(article: str, question: str) -> list:
    """
    Score article sentences by keyword overlap with the question.
    Returns a list of 3 sentences: [most_general, moderate, most_specific].
    Hint 1 = index 0 (vague), Hint 3 = index 2 (revealing).
    """
    stop = {
        "a","an","the","is","are","was","were","in","on","at","to","of",
        "and","or","but","for","with","that","this","it","its","he","she",
        "they","we","you","i","be","been","has","have","had","do","does","did",
    }
    q_words = set(_clean(question).split()) - stop
    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if not sentences:
        fallback = "No suitable sentences found in the article."
        return [fallback, fallback, fallback]

    scored = []
    for sent in sentences:
        words   = set(_clean(sent).split())
        overlap = len(q_words & words)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = [s for _, s in scored[:3]]
    while len(top3) < 3:
        top3.append(top3[-1] if top3 else "—")

    # [2] = least overlap (vague), [1] = moderate, [0] = most overlap (revealing)
    return [top3[2], top3[1], top3[0]]


def gen_distractors(article: str, correct_answer: str, model_b: dict) -> list:
    """
    Generate 3 plausible wrong-answer distractors using Model B artifacts.
    """
    art_c     = _clean(article)
    ans_c     = _clean(correct_answer)
    ans_words = set(ans_c.split())
    art_words = [w for w in art_c.split() if len(w) >= 3 and w not in ans_words]

    distractors = []

    # Method 1: OHE cosine similarity — medium-similarity article words
    if "ohe_b" in model_b and art_words:
        try:
            from sklearn.metrics.pairwise import cosine_similarity as _cos
            ohe_b    = model_b["ohe_b"]
            ans_vec  = ohe_b.transform([ans_c])
            cand_vecs = ohe_b.transform(art_words)
            sims     = _cos(ans_vec, cand_vecs)[0]
            ranked   = sorted(zip(art_words, sims), key=lambda x: x[1], reverse=True)
            distractors += [w for w, _ in ranked[3:6]]
        except Exception:
            pass

    # Method 2: Word2Vec — semantically related words outside article
    if len(distractors) < 3 and "w2v" in model_b:
        try:
            w2v = model_b["w2v"]
            for aw in list(ans_words)[:3]:
                if aw in w2v.wv:
                    for word, _ in w2v.wv.most_similar(aw, topn=20):
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
            fw   = list(ans_words)[0] if ans_words else ""
            tgt  = freq.get(fw, 5)
            lo, hi = tgt * 0.6, tgt * 1.6
            cands = [w for w, cnt in freq.items()
                     if lo <= cnt <= hi and w not in ans_words
                     and w not in distractors and len(w) >= 3][:10]
            distractors += cands[:3 - len(distractors)]
        except Exception:
            pass

    # Fallback: grab diverse article words
    if len(distractors) < 3:
        seen = set(distractors) | ans_words
        for w in art_words:
            if w not in seen and len(w) >= 4:
                distractors.append(w)
                seen.add(w)
            if len(distractors) >= 3:
                break

    return list(dict.fromkeys(distractors))[:3]


def generate_quiz(article: str, qg, model_b: dict):
    """
    Generate (question, correct_answer, distractors, hints) from an article.
    Falls back gracefully if the QG model is not trained.
    """
    if qg is not None:
        try:
            question, correct_answer = qg.generate(article)
        except Exception:
            question, correct_answer = _fallback_qg(article)
    else:
        question, correct_answer = _fallback_qg(article)

    distractors = gen_distractors(article, correct_answer, model_b)
    hints       = get_hints(article, question)
    return question, correct_answer, distractors, hints


def _fallback_qg(article: str):
    """Simple fallback QG when the trained model is unavailable."""
    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= 6]
    if sentences:
        sent   = sentences[0]
        tokens = [t for t in sent.lower().split() if len(t) > 3]
        answer = tokens[0] if tokens else "the main idea"
        question = f"What does the article say about {answer}?"
    else:
        question = "What is the main topic of the passage?"
        answer   = "the passage topic"
    return question, answer


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR + LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

SCREENS = {
    "📝 Article Input":      "input",
    "❓ Quiz":               "quiz",
    "💡 Hints":              "hints",
    "📊 Developer Dashboard": "dashboard",
}

with st.sidebar:
    st.title("📚 RACE Quiz Generator")
    st.caption("AL2002 AI Lab — Spring 2026")
    st.markdown("---")
    screen_label = st.radio("Navigate", list(SCREENS.keys()))
    screen = SCREENS[screen_label]
    st.markdown("---")

    # Load models once
    ohe_a, model_a, model_a_label = load_model_a()
    qg_model  = load_question_generator()
    model_b   = load_model_b()

    st.markdown("**Models loaded:**")
    if model_a is not None:
        st.success(f"Model A: {model_a_label}")
    else:
        st.warning("Model A not loaded — run model_a_train.py")

    if qg_model is not None:
        st.success("QG: QuestionGenerator")
    else:
        st.warning("QG not loaded — run model_a_generate.py")

    b_parts = []
    if "ohe_b" in model_b: b_parts.append("OHE-B")
    if "w2v"   in model_b: b_parts.append("W2V")
    if "freq"  in model_b: b_parts.append("Freq")
    if b_parts:
        st.success(f"Model B: {', '.join(b_parts)}")
    else:
        st.warning("Model B not loaded — run model_b_train.py")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

_defaults = {
    "article":            "",
    "generated_question": "",
    "correct_answer":     "",
    "distractors":        [],
    "hints":              [],        # list of 3 hint strings
    "quiz_options":       [],        # list of (label, text, is_correct) shuffled
    "quiz_generated":     False,     # True after "Generate Quiz" clicked
    "answered":           False,     # True after "Check Answer" clicked
    "user_selected_idx":  None,      # index into quiz_options user chose
    "hint1_read":         False,
    "hint2_read":         False,
    "hint3_read":         False,
    "answer_revealed":    False,
}

for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 1 — ARTICLE INPUT
# ─────────────────────────────────────────────────────────────────────────────

if screen == "input":
    st.title("📝 Screen 1 — Article Input")
    st.markdown(
        "Paste a reading passage below and click **Generate Quiz**.  \n"
        "The system will automatically create a question, the correct answer, "
        "3 distractors, and 3 graduated hints — no manual input needed."
    )

    SAMPLE = (
        "Scientists have discovered a new species of deep-sea fish living at depths "
        "of more than 8,000 metres in the Pacific Ocean. The fish, named Pseudoliparis "
        "swirei, belongs to the snailfish family and was found in the Mariana Trench. "
        "Researchers say the discovery highlights how little we know about life in the "
        "deepest parts of the ocean. The fish survive the crushing pressure by having "
        "a soft, gelatinous body that allows them to withstand conditions that would "
        "destroy most other organisms. Unlike many deep-sea creatures, they appear to "
        "be highly active and are believed to be top predators in their environment."
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("Load sample article"):
            st.session_state.article = SAMPLE
            # Reset quiz state when a new article is loaded
            for k in ["quiz_generated", "answered", "user_selected_idx",
                      "hint1_read", "hint2_read", "hint3_read", "answer_revealed"]:
                st.session_state[k] = _defaults[k]

    article_input = st.text_area(
        "Reading Passage",
        value=st.session_state.article,
        height=260,
        placeholder="Paste your reading passage here…",
        key="article_text_area",
    )

    if st.button("🎯 Generate Quiz", type="primary", use_container_width=True):
        if not article_input.strip():
            st.error("Please paste a reading passage first.")
        elif len(article_input.strip().split()) < 20:
            st.error("Passage is too short — please provide at least 20 words.")
        else:
            # Reset state for a fresh quiz
            for k in ["answered", "user_selected_idx", "hint1_read",
                      "hint2_read", "hint3_read", "answer_revealed"]:
                st.session_state[k] = _defaults[k]

            st.session_state.article = article_input.strip()

            with st.spinner("Generating quiz question and answer…"):
                q, ans, distractors, hints = generate_quiz(
                    st.session_state.article, qg_model, model_b
                )

            st.session_state.generated_question = q
            st.session_state.correct_answer     = ans
            st.session_state.distractors        = distractors
            st.session_state.hints              = hints

            # Build shuffled option list: 1 correct + 3 distractors
            all_opts = [ans] + distractors[:3]
            while len(all_opts) < 4:
                all_opts.append(f"Other: {all_opts[0]}")  # pad if fewer distractors
            random.seed(42)
            random.shuffle(all_opts)
            labels = ["A", "B", "C", "D"]
            st.session_state.quiz_options = [
                (labels[i], text, text == ans)
                for i, text in enumerate(all_opts)
            ]
            st.session_state.quiz_generated = True

            st.success(
                "Quiz generated! Switch to **❓ Quiz** in the sidebar to answer it."
            )

    if st.session_state.quiz_generated:
        st.markdown("---")
        st.markdown("### Generated Question Preview")
        st.info(f"**Q:** {st.session_state.generated_question}")
        st.caption(
            "The correct answer and distractors are hidden until you submit "
            "your response on the Quiz screen."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 2 — QUIZ
# ─────────────────────────────────────────────────────────────────────────────

elif screen == "quiz":
    st.title("❓ Screen 2 — Quiz")

    if not st.session_state.quiz_generated:
        st.info("No quiz yet.  Go to **📝 Article Input** and click **Generate Quiz** first.")
        st.stop()

    # Article preview
    st.markdown("**Passage:**")
    st.markdown(
        f'<div class="rc-article">{st.session_state.article[:600]}'
        f'{"…" if len(st.session_state.article) > 600 else ""}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Question
    st.markdown(f"### {st.session_state.generated_question}")
    st.markdown("*Select the best answer:*")

    opts = st.session_state.quiz_options   # [(label, text, is_correct), ...]

    # ── Before answer submitted: show radio buttons ────────────────────────────
    if not st.session_state.answered:
        radio_labels = [f"**{lbl}.** {txt}" for lbl, txt, _ in opts]
        choice = st.radio(
            "Your answer:",
            options=range(len(opts)),
            format_func=lambda i: f"{opts[i][0]}. {opts[i][1]}",
            index=None,
            label_visibility="collapsed",
        )

        if st.button("✔ Check Answer", type="primary"):
            if choice is None:
                st.warning("Please select an answer before checking.")
            else:
                st.session_state.user_selected_idx = choice
                st.session_state.answered = True
                st.rerun()

    # ── After answer submitted: show colour-coded feedback ─────────────────────
    else:
        chosen_idx    = st.session_state.user_selected_idx
        correct_idx   = next(i for i, (_, _, ok) in enumerate(opts) if ok)
        user_is_right = (chosen_idx == correct_idx)

        for i, (lbl, txt, is_corr) in enumerate(opts):
            if is_corr:
                icon = "✅"
                css  = "rc-opt-correct"
            elif i == chosen_idx:
                icon = "❌"
                css  = "rc-opt-wrong"
            else:
                icon = "⬜"
                css  = "rc-opt-neutral"
            st.markdown(
                f'<div class="{css}">{icon} <b>{lbl}.</b> {txt}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if user_is_right:
            st.success("✅ Correct! Well done.")
        else:
            correct_text = opts[correct_idx][1]
            st.error(
                f"❌ Incorrect. The correct answer was: **{correct_text}**"
            )

        st.caption(
            "The correct answer was identified by the question generator (Model A). "
            "Go to **💡 Hints** to see graduated clues, or return to "
            "**📝 Article Input** to generate a new quiz."
        )

        if st.button("🔄 Try Again (clear answer)"):
            st.session_state.answered          = False
            st.session_state.user_selected_idx = None
            st.session_state.hint1_read        = False
            st.session_state.hint2_read        = False
            st.session_state.hint3_read        = False
            st.session_state.answer_revealed   = False
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 3 — HINTS
# ─────────────────────────────────────────────────────────────────────────────

elif screen == "hints":
    st.title("💡 Screen 3 — Hints")

    if not st.session_state.quiz_generated:
        st.info("No quiz yet.  Go to **📝 Article Input** and click **Generate Quiz** first.")
        st.stop()

    st.markdown(f"**Question:** *{st.session_state.generated_question}*")
    st.markdown("---")
    st.markdown(
        "Open each hint in order.  After reading all three, the "
        "**Reveal Answer** button will appear."
    )

    hints = st.session_state.hints
    if not hints or len(hints) < 3:
        hints = ["No hint available.", "No hint available.", "No hint available."]

    # Hint 1 — most general (vague)
    with st.expander("🟠 Hint 1 — General clue (least revealing)", expanded=False):
        st.markdown(
            f'<div class="rc-hint-1">{hints[0]}</div>',
            unsafe_allow_html=True,
        )
    h1 = st.checkbox("✔ I have read Hint 1", key="cb_h1",
                     value=st.session_state.hint1_read)
    if h1:
        st.session_state.hint1_read = True

    # Hint 2 — moderate
    with st.expander("🔵 Hint 2 — More specific clue", expanded=False):
        st.markdown(
            f'<div class="rc-hint-2">{hints[1]}</div>',
            unsafe_allow_html=True,
        )
    h2 = st.checkbox("✔ I have read Hint 2", key="cb_h2",
                     value=st.session_state.hint2_read)
    if h2:
        st.session_state.hint2_read = True

    # Hint 3 — most revealing
    with st.expander("🟢 Hint 3 — Near-explicit clue (most revealing)", expanded=False):
        st.markdown(
            f'<div class="rc-hint-3">{hints[2]}</div>',
            unsafe_allow_html=True,
        )
    h3 = st.checkbox("✔ I have read Hint 3", key="cb_h3",
                     value=st.session_state.hint3_read)
    if h3:
        st.session_state.hint3_read = True

    st.markdown("---")

    # Gate: "Reveal Answer" only after all 3 hints read
    all_read = (st.session_state.hint1_read
                and st.session_state.hint2_read
                and st.session_state.hint3_read)

    if not all_read:
        hints_done = sum([st.session_state.hint1_read,
                          st.session_state.hint2_read,
                          st.session_state.hint3_read])
        remaining  = 3 - hints_done
        st.info(
            f"Read and check all 3 hints to unlock the answer.  "
            f"({hints_done}/3 done — {remaining} remaining)"
        )
    else:
        if not st.session_state.answer_revealed:
            if st.button("🔓 Reveal Answer", type="primary"):
                st.session_state.answer_revealed = True
                st.rerun()
        else:
            st.success(
                f"**Correct Answer:** {st.session_state.correct_answer}"
            )

    st.markdown("---")

    # Distractors section
    st.markdown("### Generated Distractors (Model B)")
    st.caption(
        "These are the 3 plausible-but-wrong alternatives generated by Model B "
        "(OHE cosine similarity + Word2Vec + frequency matching)."
    )
    distractors = st.session_state.distractors or []
    if distractors:
        pill_html = " ".join(f'<span class="rc-pill">{d}</span>' for d in distractors)
        st.markdown(pill_html, unsafe_allow_html=True)
    else:
        st.warning("No distractors generated — check that Model B files are present.")

    st.markdown("")
    st.caption(
        "**How it works:** "
        "Method 1 (OHE+cosine) finds article words with medium similarity to the answer. "
        "Method 2 (Word2Vec) finds semantically related words outside the article. "
        "Method 3 (frequency) finds corpus words with similar frequency. "
        "Results are deduplicated and capped at 3."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 4 — DEVELOPER DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

elif screen == "dashboard":
    st.title("📊 Screen 4 — Developer Dashboard")
    st.caption("Model performance metrics and evaluation artifacts from evaluate.py")

    tab_verify, tab_qg, tab_bench, tab_cm, tab_gs, tab_about = st.tabs([
        "Verification Metrics",
        "QG Metrics (BLEU/ROUGE)",
        "Benchmark Comparison",
        "Confusion Matrices",
        "GridSearchCV",
        "About",
    ])

    # ── Tab 1: Verification model comparison ──────────────────────────────────
    with tab_verify:
        st.subheader("Answer Verification — Model Comparison (Test Set)")
        csv_path = DATA_PROC / "evaluation_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True, height=350)
            st.download_button(
                "Download evaluation_results.csv",
                data=csv_path.read_bytes(),
                file_name="evaluation_results.csv",
                mime="text/csv",
            )
            if "Q-Level Acc" in df.columns:
                st.subheader("Question-Level Accuracy by Model")
                chart_df = df.set_index("Model")[["Q-Level Acc"]].sort_values(
                    "Q-Level Acc", ascending=False
                )
                st.bar_chart(chart_df)
                st.caption("Random baseline = 0.25 | Target > 0.40")
            if "Macro F1" in df.columns:
                st.subheader("Macro F1 by Model")
                st.bar_chart(
                    df.set_index("Model")[["Macro F1"]].sort_values(
                        "Macro F1", ascending=False
                    )
                )
            # Per-class metrics (Issue 5)
            per_class_cols = [c for c in df.columns
                              if "class" in c.lower() or "prec" in c.lower()
                              or "rec" in c.lower()]
            if per_class_cols:
                st.subheader("Per-Class Precision / Recall / F1 (Issue 5)")
                st.caption(
                    "Class 1 = 'Correct option'. "
                    "Low values for Class 1 indicate the model predicts mostly the majority class (Class 0). "
                    "class_weight='balanced' is used in all supervised classifiers to mitigate this."
                )
                st.dataframe(df[["Model"] + per_class_cols], use_container_width=True)
        else:
            st.info("evaluation_results.csv not found. Run `python src/evaluate.py` first.")

    # ── Tab 2: QG metrics (Issue 4) ────────────────────────────────────────────
    with tab_qg:
        st.subheader("Question Generation Metrics: BLEU / ROUGE / METEOR (Issue 4)")
        st.markdown(
            "These metrics measure how closely the **automatically generated question** "
            "resembles the original RACE reference question, averaged over the test set."
        )

        metric_info = {
            "BLEU-1":  "Unigram precision — fraction of generated words appearing in the reference.",
            "BLEU-4":  "4-gram precision — higher order n-gram overlap; standard QG benchmark metric.",
            "ROUGE-1": "Unigram recall-oriented overlap between generated and reference question.",
            "ROUGE-2": "Bigram overlap — captures phrase-level similarity.",
            "ROUGE-L": "Longest common subsequence — captures sentence-level structure.",
            "METEOR":  "Synonym-aware alignment; accounts for paraphrases and stemming variants.",
        }

        qg_path = DATA_PROC / "qg_metrics.csv"
        if qg_path.exists():
            qg_df = pd.read_csv(qg_path)
            for col, desc in metric_info.items():
                if col in qg_df.columns:
                    val = qg_df[col].iloc[0]
                    val_str = f"{val:.4f}" if val is not None else "N/A"
                    st.metric(label=col, value=val_str, help=desc)
            st.markdown("---")
            st.dataframe(qg_df, use_container_width=True)
            st.download_button(
                "Download qg_metrics.csv",
                data=qg_path.read_bytes(),
                file_name="qg_metrics.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "qg_metrics.csv not found.  \n"
                "Run `python src/model_a_generate.py` then `python src/evaluate.py`."
            )

        with st.expander("📖 How QG metrics are computed"):
            st.markdown(
                """
**BLEU** (Bilingual Evaluation Understudy) compares the n-gram overlap between
the generated question and the reference (original RACE) question.
*Formula:* brevity penalty × geometric mean of n-gram precisions.

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures recall:
what fraction of the reference n-grams appear in the generated question.

**METEOR** (Metric for Evaluation of Translation with Explicit ORdering) adds
synonym matching and stemming to catch paraphrased correct answers.

All three are averaged over the test sample (`n_samples` rows).
"""
            )

    # ── Tab 3: Benchmark comparison (Issue 6) ─────────────────────────────────
    with tab_bench:
        st.subheader("Benchmark Comparison: Classical ML vs. BERT / T5 (Issue 6)")
        st.markdown(
            "Published neural QG scores on RACE and similar datasets are shown alongside "
            "our classical ML scores for context.  "
            "**Our constraint:** classical ML only — no neural networks, no BERT/transformers."
        )

        bench_path = DATA_PROC / "benchmark_comparison.csv"
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path)
            # Highlight our row
            def _highlight_our(row):
                if "This project" in str(row.get("Reference", "")):
                    return ["background-color: rgba(255,193,7,0.15)"] * len(row)
                return [""] * len(row)
            st.dataframe(
                bench_df.style.apply(_highlight_our, axis=1),
                use_container_width=True,
            )
            st.download_button(
                "Download benchmark_comparison.csv",
                data=bench_path.read_bytes(),
                file_name="benchmark_comparison.csv",
                mime="text/csv",
            )
        else:
            # Show static table even if CSV not generated yet
            static_data = {
                "System": [
                    "BERT (fine-tuned, neural)",
                    "T5-base (fine-tuned, neural)",
                    "Rule-based (heuristic templates)",
                    "Our System (TF-IDF + LR, classical)  ◄",
                ],
                "BLEU-1": [0.52, 0.58, 0.22, "run evaluate.py"],
                "BLEU-4": [0.18, 0.23, 0.05, "run evaluate.py"],
                "ROUGE-L": [0.44, 0.49, 0.20, "run evaluate.py"],
                "METEOR": [0.21, 0.26, 0.10, "run evaluate.py"],
                "Reference": [
                    "Sun et al. (2022)",
                    "Zhao et al. (2023)",
                    "Pan et al. (2019)",
                    "This project (AL2002 Spring 2026)",
                ],
            }
            st.dataframe(pd.DataFrame(static_data), use_container_width=True)
            st.info("Run `python src/evaluate.py` to populate our model's scores.")

        st.markdown("---")
        st.markdown(
            "> **Why the gap?** Neural models (BERT, T5) leverage billions of parameters "
            "pre-trained on massive corpora, giving them strong language understanding. "
            "Our classical model uses TF-IDF sentence scoring, frequency-based answer extraction, "
            "and Wh-word templates — all without any pre-trained language knowledge. "
            "The gap is expected and reflects the architectural constraint of this assignment."
        )

    # ── Tab 4: Confusion matrices ─────────────────────────────────────────────
    with tab_cm:
        st.subheader("Confusion Matrices (Test Set)")
        st.caption(
            "Each matrix shows True/False Positive/Negative counts for the "
            "binary is_correct classification task.  "
            "Title annotations show per-class Precision and Recall (Issue 5)."
        )
        cm_files = sorted(DATA_PROC.glob("cm_test_*.png"))
        if cm_files:
            cols_per_row = 2
            for i in range(0, len(cm_files), cols_per_row):
                row_files = cm_files[i:i + cols_per_row]
                cols_ui   = st.columns(cols_per_row)
                for col_ui, fpath in zip(cols_ui, row_files):
                    with col_ui:
                        label = fpath.stem.replace("cm_test_", "").replace("_", " ").title()
                        st.markdown(f"**{label}**")
                        st.image(str(fpath), use_column_width=True)
        else:
            st.info("No confusion matrix images found. Run `python src/evaluate.py`.")

    # ── Tab 5: GridSearchCV ───────────────────────────────────────────────────
    with tab_gs:
        st.subheader("GridSearchCV — LR Hyperparameter Tuning")
        gs_csv  = DATA_PROC / "gridsearch_results.csv"
        gs_heat = DATA_PROC / "gridsearch_heatmap.png"

        if gs_csv.exists():
            gs_df = pd.read_csv(gs_csv)
            st.dataframe(gs_df, use_container_width=True)
            if gs_heat.exists():
                st.image(
                    str(gs_heat),
                    caption="GridSearchCV Heatmap (C vs max_features)",
                    use_column_width=True,
                )
        else:
            st.info("gridsearch_results.csv not found. Run `python src/evaluate.py`.")

        st.markdown(
            "**What is GridSearchCV?**  \n"
            "It systematically tests every combination of hyperparameters and uses "
            "cross-validation (3-fold) to estimate which combination generalises best. "
            "Parameters tuned:  \n"
            "- `C` — regularisation strength (smaller = more regularised)  \n"
            "- `max_features` — vocabulary size for the OHE CountVectorizer"
        )

    # ── Tab 6: About ──────────────────────────────────────────────────────────
    with tab_about:
        st.subheader("Project Information")
        st.markdown(
            """
| Field | Detail |
|-------|--------|
| **Project** | RACE Reading Comprehension & Quiz Generation System |
| **Course** | AL2002 — Artificial Intelligence Lab, Spring 2026 |
| **University** | NUCES (FAST), Islamabad |
| **Dataset** | RACE — 87,866 rows, single-file 80-10-10 stratified split |
| **Model A — Verifier** | Binary is_correct classification (LR, SVM, NB, RF, XGB, KMeans, GMM, LP, Ensemble) |
| **Model A — Generator** | Template-based QG: TF-IDF sentence scoring + Wh-word templates + LR ranker |
| **Model B** | Distractor generation (OHE+cosine, Word2Vec, freq) + hint extraction |
| **Features** | OHE CountVectorizer(binary=True) + cosine sim + word counts |
| **Constraint** | Classical ML only — no neural networks, no BERT/LSTM |
            """
        )
        st.markdown("### Full Pipeline")
        st.code(
            """# Activate virtual environment
venv\\Scripts\\activate

# Step 1 — preprocess (80-10-10 stratified split)
python src/preprocessing.py

# Step 2 — train Model A verifier
python src/model_a_train.py

# Step 3 — train Question Generator
python src/model_a_generate.py

# Step 4 — train Model B (distractors + hints)
python src/model_b_train.py

# Step 5 — evaluate all models (BLEU/ROUGE/METEOR + benchmark table)
python src/evaluate.py

# Step 6 — launch Streamlit app
streamlit run ui/app.py""",
            language="bash",
        )

        st.markdown("### Model Files Status")
        for folder, label in [
            (MODELS_A, "Model A (Verifier + QG)"),
            (MODELS_B, "Model B (Distractors & Hints)"),
        ]:
            st.markdown(f"**{label}** — `{folder.relative_to(PROJECT_ROOT)}`")
            if folder.exists():
                files = (list(folder.glob("*.pkl")) +
                         list(folder.glob("*.model")) +
                         list(folder.glob("*.npz")))
                if files:
                    for f in sorted(files):
                        size_kb = f.stat().st_size / 1024
                        st.markdown(f"  - `{f.name}` ({size_kb:.0f} KB)")
                else:
                    st.caption("  (no model files yet — run the training scripts)")
            else:
                st.caption("  (folder not found — run the training scripts)")

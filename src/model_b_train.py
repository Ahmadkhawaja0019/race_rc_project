"""
src/model_b_train.py  —  Model B Training Pipeline (Distractor & Hint Generator)

Run from the project root with your venv active:
    python src/model_b_train.py

Prerequisite: run  python src/preprocessing.py  first.
(model_a_train.py is optional — this script loads its OHE vectorizer if available,
 otherwise it fits a fresh one.)

Model B has TWO components:

─────────────────────────────────────────────────────────────────────────────
COMPONENT 1 — DISTRACTOR GENERATION  (creates 3 wrong-but-plausible options)
─────────────────────────────────────────────────────────────────────────────
  Method 1 — OHE + Cosine Similarity  (PRIMARY — required by rubric)
    Find all unique words in the article.
    Vectorize each word and the correct answer using the OHE vectorizer.
    Compute cosine similarity between answer and every candidate word.
    Pick words with MEDIUM similarity (not too close = answer, not too far = random).

  Method 2 — Word2Vec Nearest Neighbours
    Train a Word2Vec model on all 87k training articles.
    For each correct answer word, find top-N closest words in embedding space.
    Filter out words already in the article (they'd be too obvious as distractors).

  Method 3 — Frequency-Based Substitution
    Count how often every word appears across all articles.
    Find words whose frequency is similar to the correct answer words.
    These are in the same "importance tier" in the domain — plausible alternatives.

─────────────────────────────────────────────────────────────────────────────
COMPONENT 2 — HINT EXTRACTION  (graduated, 3-level hints)
─────────────────────────────────────────────────────────────────────────────
  Rule-Based Keyword Overlap
    Split article into sentences.
    Score each sentence by overlap with question keywords.
    Hint 3 (most revealing) = sentence with highest overlap
    Hint 2 = second-best overlap
    Hint 1 (most general) = third-best overlap

Evaluation metrics:
  Distractors : Precision, Recall, F1 (word-level overlap with true wrong options)
  Hints       : Hint-3 coverage (does Hint 3 contain a word from the correct answer text?)

Saved files:
    models/model_b/traditional/ohe_vectorizer_b.pkl   OHE vectorizer for Model B
    models/model_b/traditional/word2vec.model          trained Word2Vec
    models/model_b/traditional/word_freq.pkl           corpus word frequency Counter
"""

import sys
import re
import time
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_A     = PROJECT_ROOT / "models" / "model_a" / "traditional"
MODELS_B     = PROJECT_ROOT / "models" / "model_b" / "traditional"
MODELS_B.mkdir(parents=True, exist_ok=True)

# ── Check prerequisites ───────────────────────────────────────────────────────
for _f in ["train_clean.csv", "val_clean.csv"]:
    if not (DATA_PROC / _f).exists():
        print(f"\nMissing: data/processed/{_f}")
        print("Fix: run  python src/preprocessing.py  first.")
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
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    from sklearn.metrics import f1_score, precision_score, recall_score
    from tqdm import tqdm
except ImportError as e:
    print(f"\nMissing package: {e}")
    print("Fix: make sure venv is active, then:  pip install -r requirements.txt")
    sys.exit(1)

try:
    from gensim.models import Word2Vec
    HAS_W2V = True
except ImportError:
    HAS_W2V = False
    print("Note: gensim not found — Word2Vec method will be skipped.")

try:
    import nltk
    nltk.data.find("corpora/stopwords")
    from nltk.corpus import stopwords as _sw
    STOPWORDS = set(_sw.words("english"))
except Exception:
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "shall",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "i", "my", "your", "his", "her", "our", "their",
        "not", "no", "so", "if", "as", "up", "out", "about", "into",
    }

# ── Configuration ─────────────────────────────────────────────────────────────
OHE_MAX_FEATURES  = 10_000
W2V_VECTOR_SIZE   = 100
W2V_WINDOW        = 5
W2V_MIN_COUNT     = 2
W2V_EPOCHS        = 5
W2V_WORKERS       = 4
DISTRACTOR_SKIP   = 3       # skip top-N most similar (too close to the answer)
DISTRACTOR_TAKE   = 3       # number of distractors to return
FREQ_TOL          = 0.40    # frequency tolerance band: ±40% of the answer word's freq
EVAL_SAMPLE       = 500     # rows of val set used for evaluation (speed)
RANDOM_STATE      = 42
MIN_WORD_LEN      = 3       # ignore very short words as distractor candidates


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1/6]  Loading clean datasets ...")
    train = pd.read_csv(DATA_PROC / "train_clean.csv", index_col=0)
    val   = pd.read_csv(DATA_PROC / "val_clean.csv",   index_col=0)
    print(f"  Train: {len(train):,} rows  |  Val: {len(val):,} rows")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# 2. OHE VECTORIZER
# ─────────────────────────────────────────────────────────────────────────────

def get_ohe_vectorizer(train_clean):
    """
    Load OHE vectorizer from model_a if it exists.
    Otherwise fit a fresh one on training article text.
    """
    print("\n[2/6]  Preparing OHE vectorizer ...")
    existing = MODELS_A / "ohe_vectorizer.pkl"
    if existing.exists():
        ohe = joblib.load(existing)
        print(f"  Loaded existing vectorizer from model_a  "
              f"(vocab size: {len(ohe.vocabulary_):,})")
    else:
        print("  model_a vectorizer not found — fitting a new one on article text ...")
        ohe = CountVectorizer(
            max_features=OHE_MAX_FEATURES,
            stop_words="english",
            binary=True,
            min_df=2,
            max_df=0.95,
        )
        ohe.fit(train_clean["article_clean"].tolist())
        print(f"  Fitted.  Vocab size: {len(ohe.vocabulary_):,}")

    joblib.dump(ohe, MODELS_B / "ohe_vectorizer_b.pkl")
    print("  Saved: models/model_b/traditional/ohe_vectorizer_b.pkl")
    return ohe


# ─────────────────────────────────────────────────────────────────────────────
# 3. WORD2VEC
# ─────────────────────────────────────────────────────────────────────────────

def train_word2vec(train_clean):
    """Train Word2Vec on all training article sentences."""
    if not HAS_W2V:
        return None

    print("\n[3/6]  Training Word2Vec on article corpus ...")
    print(f"  vector_size={W2V_VECTOR_SIZE}, window={W2V_WINDOW}, "
          f"min_count={W2V_MIN_COUNT}, epochs={W2V_EPOCHS}")

    sentences = [text.split() for text in train_clean["article_clean"].tolist()]
    t0 = time.time()
    w2v = Word2Vec(
        sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        epochs=W2V_EPOCHS,
        seed=RANDOM_STATE,
    )
    print(f"  Trained in {time.time()-t0:.1f}s  |  Vocab size: {len(w2v.wv):,} words")

    save_path = MODELS_B / "word2vec.model"
    w2v.save(str(save_path))
    print(f"  Saved: models/model_b/traditional/word2vec.model")
    return w2v


# ─────────────────────────────────────────────────────────────────────────────
# 4. WORD FREQUENCY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_freq_table(train_clean):
    """Count how often every word appears across all training articles."""
    print("\n[4/6]  Building corpus word-frequency table ...")
    t0 = time.time()
    freq = Counter()
    for text in train_clean["article_clean"]:
        freq.update(str(text).split())
    print(f"  Done in {time.time()-t0:.1f}s  |  Unique words: {len(freq):,}")
    print(f"  Top-10 words: {freq.most_common(10)}")

    joblib.dump(freq, MODELS_B / "word_freq.pkl")
    print("  Saved: models/model_b/traditional/word_freq.pkl")
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# 5. DISTRACTOR GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _article_words(article_text, exclude_words=None):
    """Extract unique meaningful words from article text."""
    words = list(set(str(article_text).split()))
    words = [w for w in words if len(w) >= MIN_WORD_LEN]
    words = [w for w in words if w not in STOPWORDS]
    if exclude_words:
        excl = set(str(exclude_words).split())
        words = [w for w in words if w not in excl]
    return words


def gen_distractors_ohe(article, correct_text, ohe):
    """
    Method 1 — OHE + Cosine Similarity  (PRIMARY METHOD)

    Strategy: find article words with MEDIUM cosine similarity to the correct answer.
    - Too high similarity → basically the same word as the answer (bad distractor)
    - Too low  similarity → completely unrelated (not a plausible distractor)
    - Medium   similarity → plausible alternative  ← this is what we want

    We skip the top DISTRACTOR_SKIP most similar words and take the next 3.
    """
    if pd.isna(correct_text):
        correct_text = ""
    correct_text = str(correct_text).strip()
    if not correct_text:
        return []
    
    candidates = _article_words(article, exclude_words=correct_text)
    if len(candidates) < DISTRACTOR_SKIP + DISTRACTOR_TAKE:
        return candidates[:DISTRACTOR_TAKE]

    # Vectorize correct answer and all candidate words in one batch
    try:
        answer_vec = ohe.transform([correct_text])          # (1, vocab)
        cand_vecs  = ohe.transform(candidates)              # (n_cands, vocab)
        sims       = cosine_similarity(answer_vec, cand_vecs)[0]  # (n_cands,)
    except Exception:
        return candidates[:DISTRACTOR_TAKE]

    # Sort by similarity (descending), skip the top few, take the next 3
    ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
    distractors = [w for w, _ in ranked[DISTRACTOR_SKIP:]][:DISTRACTOR_TAKE]
    return distractors


def gen_distractors_w2v(article, correct_text, w2v):
    """
    Method 2 — Word2Vec Nearest Neighbours

    Strategy: embed the correct answer word(s) in W2V space; find semantically
    similar words that do NOT already appear in the article (article words are
    too easy for students to spot — they'd just scan the text).
    """
    if w2v is None:
        return []

    if pd.isna(correct_text):
        correct_text = ""
    correct_text = str(correct_text).strip()
    if not correct_text:
        return []

    article_words = set(_article_words(article))
    answer_words  = [w for w in correct_text.split()
                     if w in w2v.wv and len(w) >= MIN_WORD_LEN]

    if not answer_words:
        return []

    # Collect candidates from all answer words, deduplicate
    candidates = {}
    for aw in answer_words:
        try:
            similar = w2v.wv.most_similar(aw, topn=20)
            for word, score in similar:
                if (word not in article_words
                        and word not in STOPWORDS
                        and len(word) >= MIN_WORD_LEN
                        and word != aw):
                    candidates[word] = max(candidates.get(word, 0), score)
        except KeyError:
            continue

    # Sort by similarity score, return top-3
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:DISTRACTOR_TAKE]]


def gen_distractors_freq(article, correct_text, freq_table):
    """
    Method 3 — Frequency-Based Substitution

    Strategy: find the frequency of words in the correct answer, then look for
    corpus words in the same frequency band (±FREQ_TOL).  Words with similar
    frequency are in the same "importance tier" in the domain — they sound
    plausible but refer to different things.
    """
    if pd.isna(correct_text):
        correct_text = ""
    correct_text = str(correct_text).strip()
    if not correct_text:
        return []
    
    answer_words = [w for w in correct_text.split()
                    if w in freq_table and len(w) >= MIN_WORD_LEN]
    if not answer_words:
        return []

    article_word_set = set(_article_words(article))
    answer_set       = set(correct_text.split())

    # Use the frequency of the most common word in the answer as our target
    target_freq = max(freq_table[w] for w in answer_words)

    lo = target_freq * (1 - FREQ_TOL)
    hi = target_freq * (1 + FREQ_TOL)

    candidates = [
        w for w, cnt in freq_table.items()
        if lo <= cnt <= hi
        and w not in article_word_set
        and w not in answer_set
        and w not in STOPWORDS
        and len(w) >= MIN_WORD_LEN
    ]

    if not candidates:
        return []

    # Shuffle deterministically and take top 3
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(candidates)
    return candidates[:DISTRACTOR_TAKE]


# ─────────────────────────────────────────────────────────────────────────────
# 6. HINT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def get_hints(article, question):
    """
    Rule-based hint extraction.

    Scores each sentence in the article by how many question keywords it
    contains. Returns 3 hints in graduated order:
      Hint 1 — least revealing  (3rd-best overlap)
      Hint 2 — moderate         (2nd-best overlap)
      Hint 3 — most helpful     (best overlap, near the answer)

    Why graduated? A student should try Hint 1 first. Only if they're still
    stuck do they reveal Hint 2, then Hint 3.
    """
    sentences = [s.strip() for s in str(article).split(".") if len(s.strip()) > 10]
    if not sentences:
        return {"Hint 1": "", "Hint 2": "", "Hint 3": ""}

    q_words = set(question.lower().split()) - STOPWORDS

    scored = []
    for sent in sentences:
        s_words = set(sent.lower().split())
        overlap = len(q_words & s_words)
        scored.append((sent, overlap))

    # Sort by overlap descending, keep unique sentences with overlap > 0 first
    scored.sort(key=lambda x: x[1], reverse=True)

    # Pad if fewer than 3 sentences
    while len(scored) < 3:
        scored.append(("(No additional hint available.)", 0))

    top3 = [s for s, _ in scored[:3]]
    return {
        "Hint 1": top3[2],   # least revealing
        "Hint 2": top3[1],
        "Hint 3": top3[0],   # most revealing
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _word_overlap_f1(generated_list, true_list):
    """
    Word-level F1 between generated distractors and true wrong options.

    True distractor words = union of all words in the 3 actual wrong options.
    Generated distractor words = union of all words in the 3 generated items.

    Precision = |gen_words ∩ true_words| / |gen_words|
    Recall    = |gen_words ∩ true_words| / |true_words|
    F1        = harmonic mean
    """
    def _clean_items(items):
        if items is None:
            return []
        if isinstance(items, (list, tuple, set)):
            seq = list(items)
        elif isinstance(items, str):
            seq = [items]
        else:
            if pd.isna(items):
                return []
            seq = [items]

        cleaned = []
        for x in seq:
            if pd.isna(x):
                continue
            cleaned.append(str(x))
        return cleaned

    gen_words  = set(" ".join(_clean_items(generated_list)).split()) - STOPWORDS
    true_words = set(" ".join(_clean_items(true_list)).split()) - STOPWORDS

    if not gen_words or not true_words:
        return 0.0, 0.0, 0.0

    overlap = gen_words & true_words
    p = len(overlap) / len(gen_words)
    r = len(overlap) / len(true_words)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate_distractors(val_clean, ohe, w2v, freq_table):
    """
    Evaluate all three distractor methods on a sample of the validation set.
    For each sample:
      - true distractors = the 3 actual wrong answer options
      - generated = what each method produces
      - metric = word-level precision, recall, F1
    """
    print("\n" + "=" * 60)
    print("  EVALUATING DISTRACTOR GENERATION")
    print("=" * 60)

    sample = val_clean.sample(min(EVAL_SAMPLE, len(val_clean)),
                               random_state=RANDOM_STATE)

    results = {
        "OHE+Cosine": {"P": [], "R": [], "F1": []},
        "Word2Vec":   {"P": [], "R": [], "F1": []},
        "Freq-Based": {"P": [], "R": [], "F1": []},
    }

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="  Evaluating"):
        correct_opt  = row["answer"]                          # e.g. "A"
        correct_text = row[f"{correct_opt}_clean"]            # text of correct answer
        article      = row["article_clean"]

        # True distractors = the 3 wrong options
        true_distractors = [
            row[f"{opt}_clean"]
            for opt in ["A", "B", "C", "D"]
            if opt != correct_opt
        ]

        # Method 1 — OHE + Cosine
        gen_ohe = gen_distractors_ohe(article, correct_text, ohe)
        p, r, f = _word_overlap_f1(gen_ohe, true_distractors)
        results["OHE+Cosine"]["P"].append(p)
        results["OHE+Cosine"]["R"].append(r)
        results["OHE+Cosine"]["F1"].append(f)

        # Method 2 — Word2Vec
        if w2v is not None:
            gen_w2v = gen_distractors_w2v(article, correct_text, w2v)
            p, r, f = _word_overlap_f1(gen_w2v, true_distractors)
            results["Word2Vec"]["P"].append(p)
            results["Word2Vec"]["R"].append(r)
            results["Word2Vec"]["F1"].append(f)

        # Method 3 — Frequency-Based
        gen_freq = gen_distractors_freq(article, correct_text, freq_table)
        p, r, f = _word_overlap_f1(gen_freq, true_distractors)
        results["Freq-Based"]["P"].append(p)
        results["Freq-Based"]["R"].append(r)
        results["Freq-Based"]["F1"].append(f)

    # Print results table
    print(f"\n  {'Method':<15}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 48)
    for method, scores in results.items():
        if not scores["F1"]:
            continue
        p  = np.mean(scores["P"])
        r  = np.mean(scores["R"])
        f1 = np.mean(scores["F1"])
        print(f"  {method:<15}  {p:>10.4f}  {r:>8.4f}  {f1:>8.4f}")
    print()
    print("  Note: word-level overlap metric.  Higher F1 = generated words")
    print("        appear more often in the true wrong-option text.")
    print("  (Include this table in your report under Model B evaluation)")

    return results


def evaluate_hints(val_clean):
    """
    Evaluate hint quality on a sample of the validation set.

    Metric: Hint-3 Coverage
      For each sample, check whether Hint 3 (the most revealing hint) contains
      at least one word from the correct answer text.  A good Hint 3 should
      strongly point toward the answer without revealing it explicitly.

    Also computes: average sentence overlap score per hint level (1, 2, 3).
    """
    print("\n" + "=" * 60)
    print("  EVALUATING HINT EXTRACTION")
    print("=" * 60)

    sample = val_clean.sample(min(EVAL_SAMPLE, len(val_clean)),
                               random_state=RANDOM_STATE)

    hint3_coverage = []   # does Hint 3 contain an answer word?
    overlap_by_hint = {1: [], 2: [], 3: []}

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="  Evaluating"):
        article      = row["article_clean"]
        question     = row["question_clean"]
        correct_opt  = row["answer"]
        if pd.isna(correct_opt):
            continue
        correct_text = row[f"{correct_opt}_clean"]

        article = "" if pd.isna(article) else str(article)
        question = "" if pd.isna(question) else str(question)
        correct_text = "" if pd.isna(correct_text) else str(correct_text)

        hints = get_hints(article, question)
        ans_words = set(correct_text.split()) - STOPWORDS

        # Hint-3 coverage
        hint3_words = set(hints["Hint 3"].lower().split())
        covered = 1 if (hint3_words & ans_words) else 0
        hint3_coverage.append(covered)

        # Overlap score per hint level (number of shared question keywords)
        q_words = set(question.lower().split()) - STOPWORDS
        for level, key in [(1, "Hint 1"), (2, "Hint 2"), (3, "Hint 3")]:
            h_words = set(hints[key].lower().split())
            overlap_by_hint[level].append(len(q_words & h_words))

    cov = np.mean(hint3_coverage)
    print(f"\n  Hint-3 Coverage       : {cov:.4f}  "
          f"(fraction where Hint 3 contains an answer word)")
    print(f"  Avg keyword overlap per hint level:")
    print(f"    Hint 1 (least revealing) : {np.mean(overlap_by_hint[1]):.2f} keywords")
    print(f"    Hint 2 (moderate)        : {np.mean(overlap_by_hint[2]):.2f} keywords")
    print(f"    Hint 3 (most revealing)  : {np.mean(overlap_by_hint[3]):.2f} keywords")
    print()
    print("  Good sign: Hint 3 should have the highest keyword overlap.")
    print("  (Include this table in your report under Model B evaluation)")

    return cov, overlap_by_hint


def show_examples(val_clean, ohe, w2v, freq_table, n=3):
    """Print n side-by-side examples of generated distractors and hints."""
    print("\n" + "=" * 60)
    print("  SAMPLE OUTPUT EXAMPLES")
    print("=" * 60)

    sample = val_clean.sample(n, random_state=RANDOM_STATE)

    for i, (_, row) in enumerate(sample.iterrows()):
        correct_opt  = row["answer"]
        correct_text = row[f"{correct_opt}_clean"]
        article      = row["article_clean"]
        question     = row["question_clean"]

        true_distractors = [
            row[f"{opt}"]
            for opt in ["A", "B", "C", "D"]
            if opt != correct_opt
        ]

        print(f"\n  ── Example {i+1} ──────────────────────────────────────")
        print(f"  Article (first 200 chars): {str(row['article'])[:200]}...")
        print(f"  Question     : {row['question']}")
        print(f"  Correct (opt {correct_opt}): {row[correct_opt]}")
        print(f"  True distractors : {true_distractors}")
        print()

        d_ohe  = gen_distractors_ohe(article, correct_text, ohe)
        d_w2v  = gen_distractors_w2v(article, correct_text, w2v) if w2v else ["(skipped)"]
        d_freq = gen_distractors_freq(article, correct_text, freq_table)
        hints  = get_hints(article, question)

        print(f"  Generated — OHE+Cosine : {d_ohe}")
        print(f"  Generated — Word2Vec   : {d_w2v}")
        print(f"  Generated — Freq-Based : {d_freq}")
        print(f"  Hint 1 (vague)   : {hints['Hint 1'][:120]}...")
        print(f"  Hint 2 (medium)  : {hints['Hint 2'][:120]}...")
        print(f"  Hint 3 (specific): {hints['Hint 3'][:120]}...")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model B — Training Pipeline")
    print("  Distractor Generation + Hint Extraction")
    print("  RACE Dataset  |  Classical / Rule-Based Methods Only")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    train_clean, val_clean = load_data()

    # ── OHE vectorizer ────────────────────────────────────────────────────
    ohe = get_ohe_vectorizer(train_clean)

    # ── Word2Vec ──────────────────────────────────────────────────────────
    w2v = train_word2vec(train_clean)

    # ── Frequency table ───────────────────────────────────────────────────
    freq_table = build_freq_table(train_clean)

    # ── Distractor evaluation ─────────────────────────────────────────────
    print("\n[5/6]  Evaluating distractor generation ...")
    distractor_results = evaluate_distractors(val_clean, ohe, w2v, freq_table)

    # ── Hint evaluation ───────────────────────────────────────────────────
    print("\n[6/6]  Evaluating hint extraction ...")
    hint_coverage, hint_overlaps = evaluate_hints(val_clean)

    # ── Show examples ─────────────────────────────────────────────────────
    show_examples(val_clean, ohe, w2v, freq_table, n=2)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Model B — Summary")
    print("=" * 60)
    print("  Distractor models saved:")
    print("    models/model_b/traditional/ohe_vectorizer_b.pkl")
    if w2v:
        print("    models/model_b/traditional/word2vec.model")
    print("    models/model_b/traditional/word_freq.pkl")
    print()
    print("  Hint extractor: rule-based (no model file — logic is in get_hints())")
    print()
    print("  Next step: run  python src/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
model_a_generate.py  —  Template-Based Question Generator (Classical ML)

This module trains and saves a QuestionGenerator that:
  1. Scores sentences in an article by TF-IDF importance
  2. Extracts a key answer phrase from the top-ranked sentence
     (frequency + position heuristic)
  3. Applies Wh-word templates to generate question candidates
     by masking the answer phrase
  4. Uses a trained LR ranker to select the best question candidate

Run as a script to train and save the generator:
    python src/model_a_generate.py

The generator is automatically invoked by ui/app.py at quiz-generation time.

Saved artifacts
---------------
  models/model_a/traditional/qg_tfidf.pkl      — TfidfVectorizer (sentences)
  models/model_a/traditional/qg_ranker.pkl     — LogisticRegression question ranker
"""

import re
import sys
import random
import string
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = PROJECT_ROOT / "models" / "model_a" / "traditional"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    import numpy as np
    import pandas as pd
    import joblib
    from tqdm import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"\nMissing package: {e}")
    print("Fix: make sure your venv is active, then run:")
    print("     pip install -r requirements.txt")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Wh-word templates.  {answer} is a placeholder for the masked phrase.
# {rest} is the remainder of the sentence after removing the answer phrase.
WH_TEMPLATES = [
    ("What",  "What {rest}?"),
    ("What",  "What is {answer}?"),         # fallback short form
    ("Who",   "Who {rest}?"),
    ("Where", "Where {rest}?"),
    ("When",  "When {rest}?"),
    ("Why",   "Why {rest}?"),
    ("How",   "How {rest}?"),
    ("Which", "Which {rest}?"),
]

# Words that indicate the answer phrase is likely a person
PERSON_INDICATORS = {
    "he", "she", "his", "her", "him", "mr", "mrs", "ms", "dr",
    "president", "king", "queen", "prince", "professor", "captain",
}

# Words that indicate a location
LOCATION_INDICATORS = {
    "in", "at", "near", "on", "from", "to", "between", "around",
    "country", "city", "town", "village", "place", "region",
}

# Stopwords (minimal set to avoid filtering too aggressively)
STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "in", "on", "at", "to", "of",
    "for", "with", "by", "from", "that", "this", "is", "was", "are",
    "were", "be", "been", "it", "its", "as", "up", "into", "so", "if",
    "about", "after", "before", "has", "have", "had", "do", "does", "did",
    "not", "no", "can", "could", "will", "would", "should", "may", "might",
    "than", "then", "there", "their", "they", "we", "our", "you", "your",
    "he", "she", "him", "her", "his", "hers", "them", "us", "i", "me", "my",
}


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def sent_tokenize(text: str) -> list:
    """
    Split text into sentences using a simple regex.
    Avoids a hard dependency on NLTK punkt models.
    """
    # Split on . ! ? followed by a space and uppercase letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    # Filter out very short sentences (likely noise)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 4]
    return sentences


def word_tokenize(text: str) -> list:
    """Lowercase and split into word tokens, stripping punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def ngrams(tokens: list, n: int) -> list:
    """Return all n-grams from a token list."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# QuestionGenerator class
# ─────────────────────────────────────────────────────────────────────────────

class QuestionGenerator:
    """
    Classical ML Question Generator trained on RACE QA pairs.

    Pipeline
    --------
    fit(train_df)
        1. Fit a TF-IDF vectorizer on all article sentences.
        2. For each training example, find which sentence best overlaps
           the known question + answer — that is the training signal.
        3. Collect (sentence, question, answer) triples and train a
           LogisticRegression ranker to score generated question candidates.

    generate(article) -> (question_str, answer_str)
        1. Score article sentences by cosine similarity to the article
           TF-IDF centroid (most "representative / informative" sentences).
        2. Extract the key answer phrase from the top sentence using
           n-gram frequency + position scoring.
        3. Apply Wh-word templates to generate question candidates.
        4. Score candidates with the trained ranker.
        5. Return the highest-scoring (question, answer) pair.
    """

    def __init__(self):
        self.tfidf          = TfidfVectorizer(
            max_features=15_000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.ranker         = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
        )
        self._corpus_freqs  = Counter()   # word frequencies across all articles
        self._fitted        = False

    # ------------------------------------------------------------------ #
    #  fit
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train the sentence TF-IDF and the question-candidate ranker.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain columns: article, question, answer, A, B, C, D
        """
        print("  [QG] Building corpus word frequencies ...")
        all_words = []
        for article in train_df["article"].dropna():
            all_words.extend(word_tokenize(str(article)))
        self._corpus_freqs = Counter(all_words)

        # ── Fit TF-IDF on individual sentences ──────────────────────
        print("  [QG] Fitting sentence TF-IDF vectorizer ...")
        all_sentences = []
        for article in train_df["article"].dropna():
            sents = sent_tokenize(str(article))
            all_sentences.extend(sents)

        # Cap at 200 k sentences to keep memory manageable
        if len(all_sentences) > 200_000:
            random.seed(42)
            all_sentences = random.sample(all_sentences, 200_000)

        self.tfidf.fit(all_sentences)
        print(f"  [QG] TF-IDF fitted on {len(all_sentences):,} sentences "
              f"| vocab size: {len(self.tfidf.vocabulary_):,}")

        # ── Build ranker training data ───────────────────────────────
        print("  [QG] Building ranker training data ...")
        X_feats, y_labels = [], []

        sample_df = train_df.sample(
            min(20_000, len(train_df)), random_state=42
        ).reset_index(drop=True)

        for _, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc="  [QG] Building ranker",
            leave=True,
        ):
            article  = str(row["article"])
            question = str(row["question"])
            answer_col = str(row["answer"]).strip()  # 'A','B','C','D'
            if answer_col not in ("A", "B", "C", "D"):
                continue
            correct_answer_text = str(row[answer_col])

            sents = sent_tokenize(article)
            if not sents:
                continue

            # Find the gold sentence (highest overlap with question+answer)
            q_tokens = set(word_tokenize(question))
            a_tokens = set(word_tokenize(correct_answer_text))
            target_tokens = q_tokens | a_tokens

            best_idx, best_score = 0, -1
            for idx, sent in enumerate(sents):
                s_tokens = set(word_tokenize(sent))
                score = len(s_tokens & target_tokens) / (len(s_tokens) + 1)
                if score > best_score:
                    best_score, best_idx = score, idx

            gold_sent = sents[best_idx]

            # Generate candidates from the gold sentence
            candidates = self._generate_candidates(gold_sent, article)
            if not candidates:
                continue

            # Positive example: candidate most similar to gold question
            q_vec = self.tfidf.transform([question])
            for cand_q, cand_a, cand_wh in candidates:
                cand_vec = self.tfidf.transform([cand_q])
                sim = float(cosine_similarity(cand_vec, q_vec)[0, 0])
                feats = self._candidate_features(
                    cand_q, cand_a, cand_wh, gold_sent, sim
                )
                # Label = 1 if similarity >= 0.15 (plausible match to real Q)
                label = 1 if sim >= 0.15 else 0
                X_feats.append(feats)
                y_labels.append(label)

        if not X_feats:
            print("  [QG] WARNING: no training examples built — skipping ranker fit.")
            return

        X_arr = np.array(X_feats, dtype=np.float32)
        y_arr = np.array(y_labels, dtype=np.int32)
        pos_rate = y_arr.mean()
        print(f"  [QG] Ranker training: {len(X_arr):,} candidates, "
              f"positive rate: {pos_rate:.3f}")

        self.ranker.fit(X_arr, y_arr)
        self._fitted = True
        print("  [QG] Ranker trained.")

    # ------------------------------------------------------------------ #
    #  generate
    # ------------------------------------------------------------------ #

    def generate(self, article: str) -> tuple:
        """
        Generate a (question, answer_phrase) pair from an article.

        Returns
        -------
        (question_str, answer_str)
        """
        sents = sent_tokenize(article)
        if not sents:
            return "What is the main topic of the passage?", "the passage"

        # ── Score sentences ──────────────────────────────────────────
        top_sent, top_ans = self._pick_top_sentence_and_answer(sents, article)

        # ── Generate candidates ──────────────────────────────────────
        candidates = self._generate_candidates(top_sent, article)
        if not candidates:
            return f"What can be said about {top_ans}?", top_ans

        # ── Rank candidates ──────────────────────────────────────────
        if self._fitted:
            best_q, best_a = self._rank_candidates(candidates, top_sent)
        else:
            # Fallback: pick the first candidate
            best_q, best_a, _ = candidates[0]

        return best_q, best_a

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _pick_top_sentence_and_answer(self, sents: list, article: str) -> tuple:
        """
        Score sentences by TF-IDF cosine similarity to the article centroid.
        Returns (best_sentence, answer_phrase).
        """
        # Article centroid vector
        art_vec = self.tfidf.transform([article[:3000]])   # cap length

        # Score each sentence
        scored = []
        for idx, sent in enumerate(sents):
            s_vec = self.tfidf.transform([sent])
            sim = float(cosine_similarity(s_vec, art_vec)[0, 0])
            # Position bonus: favour sentences in first 60 % of article
            pos_bonus = 0.1 if idx < len(sents) * 0.6 else 0.0
            # Length bonus: prefer sentences of 8-25 words
            wc = len(sent.split())
            len_bonus = 0.05 if 8 <= wc <= 25 else 0.0
            scored.append((sim + pos_bonus + len_bonus, idx, sent))

        scored.sort(reverse=True)
        best_sent = scored[0][2]
        answer_phrase = self._extract_answer_phrase(best_sent, article)
        return best_sent, answer_phrase

    def _extract_answer_phrase(self, sentence: str, article: str) -> str:
        """
        Pick the most informative n-gram from the sentence as the answer phrase.
        Scoring = corpus frequency (lower = more informative) × position weight.
        """
        tokens = word_tokenize(sentence)
        content_tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        if not content_tokens:
            return tokens[0] if tokens else "it"

        best_phrase, best_score = "", -1e9

        # Try unigrams and bigrams
        for n in (2, 1):
            for gram in ngrams(content_tokens, n):
                freq = self._corpus_freqs.get(gram.split()[0], 1)
                # Lower corpus frequency → more informative → higher score
                info_score = 1.0 / (freq + 1)
                # Presence in article = good (it's about the article)
                in_art = 1.0 if gram in article.lower() else 0.0
                score = info_score + 0.5 * in_art
                if score > best_score:
                    best_score, best_phrase = score, gram

        return best_phrase if best_phrase else content_tokens[0]

    def _apply_wh_template(self, template: str, sentence: str, answer: str) -> str:
        """
        Replace the answer phrase in the sentence with a Wh-word and
        reconstruct a grammatical question.
        """
        # Build the 'rest' part: sentence with answer replaced by nothing,
        # then strip leading/trailing words to form a predicate
        sent_lower = sentence.lower().strip().rstrip(".")
        ans_lower  = answer.lower()

        if ans_lower in sent_lower:
            rest = sent_lower.replace(ans_lower, "___", 1).strip()
            rest = re.sub(r"\s+", " ", rest)
            # Remove trailing punctuation
            rest = rest.rstrip(".,;:!?")
        else:
            rest = sent_lower.rstrip(".,;:!?")

        question = template.format(answer=answer, rest=rest)
        # Capitalise first letter
        question = question[0].upper() + question[1:]
        # Ensure ends with ?
        if not question.endswith("?"):
            question = question.rstrip(".!") + "?"
        return question

    def _generate_candidates(self, sentence: str, article: str) -> list:
        """
        Generate a list of (question, answer_phrase, wh_word) tuples
        from a single sentence.
        """
        answer_phrase = self._extract_answer_phrase(sentence, article)
        if not answer_phrase:
            return []

        # Infer most likely Wh-word based on answer phrase content
        tokens = set(word_tokenize(sentence))
        preferred_wh = []
        if tokens & PERSON_INDICATORS:
            preferred_wh = ["Who", "What"]
        elif tokens & LOCATION_INDICATORS:
            preferred_wh = ["Where", "What"]
        else:
            preferred_wh = ["What", "How", "Which"]

        candidates = []
        for wh_word, template in WH_TEMPLATES:
            question = self._apply_wh_template(template, sentence, answer_phrase)
            # Basic quality filter: must be at least 4 words
            if len(question.split()) >= 4:
                # Boost preferred Wh-words by inserting them first
                priority = 0 if wh_word in preferred_wh else 1
                candidates.append((question, answer_phrase, wh_word, priority))

        # Sort by priority (preferred first), then deduplicate
        candidates.sort(key=lambda x: x[3])
        seen = set()
        deduped = []
        for q, a, wh, _ in candidates:
            if q not in seen:
                seen.add(q)
                deduped.append((q, a, wh))

        return deduped

    def _candidate_features(
        self,
        question: str,
        answer: str,
        wh_word: str,
        sentence: str,
        sim_to_ref: float,
    ) -> list:
        """
        Feature vector for the question ranker.

        Features
        --------
        0  sim_to_ref        — cosine similarity to reference question (training only)
        1  q_len             — number of words in question
        2  ans_len           — number of words in answer phrase
        3  wh_is_what        — 1 if Wh-word is "What"
        4  wh_is_who         — 1 if Wh-word is "Who"
        5  wh_is_where       — 1 if Wh-word is "Where"
        6  wh_is_when        — 1 if Wh-word is "When"
        7  wh_is_how         — 1 if Wh-word is "How"
        8  ans_in_sentence   — 1 if answer phrase appears in source sentence
        9  sent_len          — number of words in source sentence
        10 ans_freq          — log corpus frequency of first word of answer
        """
        ans_tokens = word_tokenize(answer)
        q_tokens   = word_tokenize(question)
        ans_freq   = self._corpus_freqs.get(ans_tokens[0] if ans_tokens else "", 1)

        return [
            sim_to_ref,
            len(q_tokens),
            len(ans_tokens),
            1 if wh_word == "What"  else 0,
            1 if wh_word == "Who"   else 0,
            1 if wh_word == "Where" else 0,
            1 if wh_word == "When"  else 0,
            1 if wh_word == "How"   else 0,
            1 if answer.lower() in sentence.lower() else 0,
            len(sentence.split()),
            float(np.log1p(ans_freq)),
        ]

    def _rank_candidates(self, candidates: list, sentence: str) -> tuple:
        """
        Use the trained ranker to score each candidate and return the best.
        Returns (best_question, best_answer).
        """
        feats = []
        for q, a, wh in candidates:
            # sim_to_ref is unavailable at inference time; use 0.0 placeholder
            f = self._candidate_features(q, a, wh, sentence, 0.0)
            feats.append(f)

        X = np.array(feats, dtype=np.float32)
        # predict_proba[:, 1] = probability of being a good question
        probs = self.ranker.predict_proba(X)[:, 1]
        best_idx = int(np.argmax(probs))
        return candidates[best_idx][0], candidates[best_idx][1]

    # ------------------------------------------------------------------ #
    #  save / load
    # ------------------------------------------------------------------ #

    def save(self, model_dir: Path) -> None:
        joblib.dump(self.tfidf,         model_dir / "qg_tfidf.pkl")
        joblib.dump(self.ranker,        model_dir / "qg_ranker.pkl")
        joblib.dump(self._corpus_freqs, model_dir / "qg_corpus_freqs.pkl")
        joblib.dump(self._fitted,       model_dir / "qg_fitted_flag.pkl")
        print(f"  [QG] Saved to {model_dir}/")

    @classmethod
    def load(cls, model_dir: Path) -> "QuestionGenerator":
        qg = cls()
        qg.tfidf          = joblib.load(model_dir / "qg_tfidf.pkl")
        qg.ranker         = joblib.load(model_dir / "qg_ranker.pkl")
        qg._corpus_freqs  = joblib.load(model_dir / "qg_corpus_freqs.pkl")
        qg._fitted        = joblib.load(model_dir / "qg_fitted_flag.pkl")
        return qg


# ─────────────────────────────────────────────────────────────────────────────
# Train script entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Model A — Question Generator Training")
    print("=" * 60)

    train_path = DATA_PROC / "train_clean.csv"
    if not train_path.exists():
        print(f"\nERROR: {train_path} not found.")
        print("Run  python src/preprocessing.py  first.")
        sys.exit(1)

    print(f"\n  Loading {train_path.name} ...")
    train_df = pd.read_csv(train_path, index_col=0)
    print(f"  Rows loaded: {len(train_df):,}")

    # Drop rows missing required columns
    required = ["article", "question", "answer", "A", "B", "C", "D"]
    train_df = train_df.dropna(subset=required)
    train_df = train_df[train_df["answer"].isin(["A", "B", "C", "D"])]
    print(f"  Rows after filtering: {len(train_df):,}")

    print("\n  Training QuestionGenerator ...")
    qg = QuestionGenerator()
    qg.fit(train_df)

    print("\n  Saving model artifacts ...")
    qg.save(MODEL_DIR)

    # ── Quick smoke test ──────────────────────────────────────────
    print("\n  Smoke test (3 examples from training set) ...")
    for i, row in train_df.sample(3, random_state=1).iterrows():
        article = str(row["article"])
        q_gen, a_gen = qg.generate(article)
        q_real = str(row["question"])
        print(f"\n  Article snippet : {article[:120]} ...")
        print(f"  Real question   : {q_real}")
        print(f"  Generated Q     : {q_gen}")
        print(f"  Generated ans   : {a_gen}")

    print("\n" + "=" * 60)
    print("  DONE — QuestionGenerator training complete!")
    print("=" * 60)
    print("  Next step: run  python src/model_b_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

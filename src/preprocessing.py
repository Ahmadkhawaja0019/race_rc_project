"""
preprocessing.py  —  RACE Dataset Preprocessing Pipeline

Run from the project root with your venv active:
    python src/preprocessing.py

What this script does:
  1. Loads  a SINGLE raw CSV file from data/raw/train.csv
  2. Performs an 80-10-10 stratified split by the 'answer' column
       (random_state=42 for reproducibility)
  3. Drops  rows that have missing values in critical columns
  4. Encodes answer labels:  A -> 0,  B -> 1,  C -> 2,  D -> 3
  5. Cleans text  (lowercase, remove punctuation, collapse spaces)
  6. Adds word-count features  (article_length, q_length)
  7. Creates a 'combined' column  (article + question + answer option)
  8. Expands the dataset: 1 original row  ->  4 rows  (one per option)
       Each expanded row has label 'is_correct' = 1 if that option is
       the right answer, else 0.  (Expected positive rate ≈ 0.25)
  9. Saves everything to data/processed/

WHY a single file?
  The RACE download from Kaggle (train.csv / dev.csv / test.csv) often
  contains the same data in all three files.  To guarantee non-overlapping
  train / validation / test sets we load train.csv once and split it here
  with a fixed seed.

Outputs
-------
  data/processed/train_clean.csv      — cleaned 80 % of data
  data/processed/val_clean.csv        — cleaned 10 % (validation)
  data/processed/test_clean.csv       — cleaned 10 % (held-out test)
  data/processed/train_expanded.csv   — 4x expanded train for Model A
  data/processed/val_expanded.csv     — 4x expanded val for Model A
  data/processed/test_expanded.csv    — 4x expanded test for Model A
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------
# Path setup — works whether you run the script from the project
# root OR from inside src/
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW  = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Try to import required packages.  If they are missing, tell the
# user exactly how to fix it rather than crashing with a cryptic error.
# ---------------------------------------------------------------
try:
    import pandas as pd
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"\nMissing package: {e}")
    print("Fix: make sure your venv is active, then run:")
    print("     pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
LABEL_MAP     = {"A": 0, "B": 1, "C": 2, "D": 3}
CRITICAL_COLS = ["article", "question", "A", "B", "C", "D", "answer"]
OPTIONS       = ["A", "B", "C", "D"]
RANDOM_SEED   = 42


# ---------------------------------------------------------------
# Step 1 — Load single file and perform 80-10-10 stratified split
# ---------------------------------------------------------------

def load_and_split() -> tuple:
    """
    Load data/raw/train.csv and split into train / val / test
    using a stratified 80-10-10 split by the 'answer' column.

    Returns
    -------
    train_df, val_df, test_df  — three non-overlapping DataFrames
    """
    # Find the file — prefer train.csv, fall back to any CSV present
    candidates = ["train.csv", "dev.csv", "test.csv"]
    path = None
    for name in candidates:
        candidate = DATA_RAW / name
        if candidate.exists():
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            "No CSV file found in data/raw/  "
            "(looked for train.csv, dev.csv, test.csv).\n"
            "Please place at least one RACE CSV file in data/raw/"
        )

    print(f"  Loading: {path.name}")
    df = pd.read_csv(path, index_col=0)
    print(f"  Raw file: {df.shape[0]:,} rows  x  {df.shape[1]} cols")

    # Drop rows missing critical columns BEFORE splitting so the
    # stratify column has no NaN values.
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLS)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing values before splitting.")

    # Keep only rows whose answer is A / B / C / D
    df = df[df["answer"].isin(["A", "B", "C", "D"])].copy()
    print(f"  Usable rows: {len(df):,}")

    # ── 80 / 20 split first ──────────────────────────────────────
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["answer"],
        random_state=RANDOM_SEED,
    )

    # ── 50 / 50 of the remaining 20 %  →  10 % val, 10 % test ───
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["answer"],
        random_state=RANDOM_SEED,
    )

    print(f"\n  Split summary (stratified by 'answer', seed={RANDOM_SEED}):")
    print(f"    train : {len(train_df):>7,} rows  ({len(train_df)/len(df)*100:.1f} %)")
    print(f"    val   : {len(val_df):>7,} rows  ({len(val_df)/len(df)*100:.1f} %)")
    print(f"    test  : {len(test_df):>7,} rows  ({len(test_df)/len(df)*100:.1f} %)")

    # Verify stratification (answer distribution should be ~25 % each)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = part["answer"].value_counts(normalize=True).sort_index()
        dist_str = "  ".join(f"{k}={v:.3f}" for k, v in dist.items())
        print(f"    {name:5s} answer dist: {dist_str}")

    return train_df, val_df, test_df


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Lowercase -> remove punctuation -> collapse whitespace.
    This is the same function used in the EDA notebook (Cell 8).
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # keep only letters, digits, spaces
    text = re.sub(r"\s+", " ", text).strip()    # collapse multiple spaces
    return text


def preprocess_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Clean one split:
      - encode answer labels (A/B/C/D  ->  0/1/2/3)
      - clean article, question, and option text
      - add word-count columns (article_length, q_length)
    Returns a new DataFrame (original is not mutated).

    Note: NaN-dropping has already been done in load_and_split().
    """
    print(f"\n  --- Preprocessing  {split_name} ---")
    df = df.copy()

    # Encode answer labels
    df["label"] = df["answer"].map(LABEL_MAP)
    unmapped = df["label"].isna().sum()
    if unmapped:
        print(f"  Warning: {unmapped} rows had unknown answer values — dropping.")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Clean text columns
    print(f"  Cleaning text for {len(df):,} rows...", end=" ", flush=True)
    df["article_clean"]  = df["article"].apply(clean_text)
    df["question_clean"] = df["question"].apply(clean_text)
    for opt in OPTIONS:
        df[f"{opt}_clean"] = df[opt].apply(clean_text)
    print("done.")

    # Word count features
    df["article_length"] = df["article"].apply(lambda x: len(str(x).split()))
    df["q_length"]       = df["question"].apply(lambda x: len(str(x).split()))

    print(f"  Result: {len(df):,} rows")
    return df


def expand_for_model_a(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Convert 1 row  ->  4 rows  (one row per answer option).

    New columns:
      combined   — article_clean + ' ' + question_clean + ' ' + option_clean
      option     — which option this row represents: A / B / C / D
      is_correct — 1 if this option is the correct answer, else 0

    The expected positive rate (mean of is_correct) is ~0.25 since
    exactly one of every four options is correct.
    """
    print(f"\n  --- Expanding  {split_name}  (4x rows) ---")
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}", leave=True):
        for opt in OPTIONS:
            combined = (
                row["article_clean"]
                + " "
                + row["question_clean"]
                + " "
                + row[f"{opt}_clean"]
            )
            records.append(
                {
                    "id":             row.get("id", ""),
                    "combined":       combined,
                    "option":         opt,
                    "is_correct":     1 if row["answer"] == opt else 0,
                    "article_length": row["article_length"],
                    "q_length":       row["q_length"],
                }
            )

    expanded = pd.DataFrame(records)
    pos_rate = expanded["is_correct"].mean()
    print(f"  Rows: {len(expanded):,}  |  Positive rate: {pos_rate:.4f}  (expect ~0.2500)")

    if abs(pos_rate - 0.25) > 0.01:
        print("  WARNING: positive rate is far from 0.25 — check the data!")

    return expanded


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("  RACE Dataset Preprocessing Pipeline")
    print("=" * 60)

    # ---- Step 1: Load single file and split 80-10-10 ----
    print("\n[1/5]  Loading raw data and performing 80-10-10 stratified split ...")
    train_raw, val_raw, test_raw = load_and_split()

    # ---- Step 2: Clean ----
    print("\n[2/5]  Cleaning text + encoding labels ...")
    train_clean = preprocess_split(train_raw, "train")
    val_clean   = preprocess_split(val_raw,   "val")
    test_clean  = preprocess_split(test_raw,  "test")

    # ---- Step 3: Save cleaned splits ----
    print("\n[3/5]  Saving cleaned splits ...")
    train_clean.to_csv(DATA_PROC / "train_clean.csv")
    val_clean.to_csv(DATA_PROC   / "val_clean.csv")
    test_clean.to_csv(DATA_PROC  / "test_clean.csv")
    print("  Saved: data/processed/train_clean.csv")
    print("  Saved: data/processed/val_clean.csv")
    print("  Saved: data/processed/test_clean.csv")

    # ---- Step 4: Expand for Model A ----
    print("\n[4/5]  Expanding rows for Model A (1 row -> 4 rows) ...")
    train_exp = expand_for_model_a(train_clean, "train")
    val_exp   = expand_for_model_a(val_clean,   "val")
    test_exp  = expand_for_model_a(test_clean,  "test")

    # ---- Step 5: Save expanded splits ----
    print("\n[5/5]  Saving expanded splits ...")
    train_exp.to_csv(DATA_PROC / "train_expanded.csv", index=False)
    val_exp.to_csv(DATA_PROC   / "val_expanded.csv",   index=False)
    test_exp.to_csv(DATA_PROC  / "test_expanded.csv",  index=False)
    print("  Saved: data/processed/train_expanded.csv")
    print("  Saved: data/processed/val_expanded.csv")
    print("  Saved: data/processed/test_expanded.csv")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  DONE — Preprocessing complete!")
    print("=" * 60)
    print(f"  train_clean    : {len(train_clean):>8,} rows  (~80 %)")
    print(f"  train_expanded : {len(train_exp):>8,} rows  (~4x train_clean)")
    print(f"  val_clean      : {len(val_clean):>8,} rows  (~10 %)")
    print(f"  val_expanded   : {len(val_exp):>8,} rows")
    print(f"  test_clean     : {len(test_clean):>8,} rows  (~10 %)")
    print(f"  test_expanded  : {len(test_exp):>8,} rows")
    print()
    print("  Next step: run  python src/model_a_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

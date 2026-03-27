"""
src.data.preprocessor
---------------------
Cleans, validates, and transforms the raw Book Crossing DataFrames
produced by BookCrossingLoader into analysis-ready artifacts used by
all three downstream modules (recommender, clustering, age estimation).

Design principles
-----------------
* Fit/transform pattern — call fit_transform(raw) once; the fitted
  state (ISBN→index map, user→index map) is stored on the instance so
  that downstream modules can reuse the same mappings.
* Immutable inputs — the raw DataFrames are never modified in-place.
* Transparent logging — every filtering step logs how many rows were
  removed and why, so results are fully reproducible.
* Single source of truth — one preprocessor, three tasks. No divergent
  cleaning logic scattered across notebooks.

Cleaning decisions (grounded in actual data inspection)
--------------------------------------------------------
Users
  • Age coerced to numeric; non-numeric → NaN.
  • Ages outside [5, 100] → NaN  (dataset contains ages 0 and 244).
  • Blank/null user_id rows dropped.
  • Duplicate user_ids dropped (keep first).

Books
  • ISBNs stripped of whitespace.
  • ISBNs shorter than 10 digits zero-padded on the left (dataset
    contains ISBNs like '342310538' which should be '0342310538').
  • Duplicate ISBNs dropped (keep first).
  • 'year' column coerced to int; implausible years (≤ 1000 or ≥ 2025)
    set to NaN.

Ratings
  • Blank user_id / isbn / rating rows dropped.
  • 'rating' coerced to int; values outside [0, 10] dropped.
  • Duplicate (user_id, isbn) pairs dropped (keep first).
  • Ratings whose user_id is not in the cleaned users table dropped.

Outputs produced by export_libsvm()
------------------------------------
  book_ratings.libsvm              — all users, label = 0 (for Task 1 & 2)
  book_ratings_with_user_age.libsvm    — users with known age, label = age
  book_ratings_without_user_age.libsvm — users with missing age, label = 0

Example
-------
>>> from src.data.loader import BookCrossingLoader
>>> from src.data.preprocessor import BookCrossingPreprocessor
>>>
>>> raw  = BookCrossingLoader("data/raw").load_all()
>>> prep = BookCrossingPreprocessor()
>>> clean = prep.fit_transform(raw)
>>>
>>> # Inspect
>>> clean.users.shape
>>> clean.ratings.shape
>>> print(prep.summary())
>>>
>>> # Export libsvm files for all three tasks
>>> prep.export_libsvm(clean, output_dir="data/processed")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import dump_svmlight_file

from .loader import RawDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public container returned by BookCrossingPreprocessor.fit_transform()
# ---------------------------------------------------------------------------

@dataclass
class CleanDataset:
    """
    Container holding the cleaned, analysis-ready DataFrames.

    Attributes
    ----------
    users : pd.DataFrame
        Columns: user_id (str), age (float | NaN).
        Only rows with valid user_id remain.

    users_with_age : pd.DataFrame
        Subset of users where age is not NaN.

    users_without_age : pd.DataFrame
        Subset of users where age is NaN.

    books : pd.DataFrame
        Columns: isbn (str), title (str), author (str),
                 year (float | NaN), publisher (str).
        ISBNs have been zero-padded to 10 characters.

    ratings : pd.DataFrame
        Columns: user_id (str), isbn (str), rating (int).
        Only valid ratings (0–10) for known users.

    isbn_to_idx : dict[str, int]
        Maps each unique ISBN in ratings to a 1-based integer index
        (used as the feature column index in libsvm files).

    user_to_idx : dict[str, int]
        Maps each unique user_id in ratings to a 0-based integer index
        (used as the row index in the sparse matrix).

    idx_to_isbn : dict[int, str]
        Inverse of isbn_to_idx.

    idx_to_user : dict[int, str]
        Inverse of user_to_idx.
    """

    users:               pd.DataFrame
    users_with_age:      pd.DataFrame
    users_without_age:   pd.DataFrame
    books:               pd.DataFrame
    ratings:             pd.DataFrame
    isbn_to_idx:         Dict[str, int] = field(default_factory=dict)
    user_to_idx:         Dict[str, int] = field(default_factory=dict)
    idx_to_isbn:         Dict[int, str] = field(default_factory=dict)
    idx_to_user:         Dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_users_with_age(self) -> int:
        return len(self.users_with_age)

    @property
    def n_users_without_age(self) -> int:
        return len(self.users_without_age)

    @property
    def n_books(self) -> int:
        return len(self.books)

    @property
    def n_ratings(self) -> int:
        return len(self.ratings)

    @property
    def n_unique_rated_books(self) -> int:
        return self.ratings["isbn"].nunique()

    @property
    def sparsity(self) -> float:
        """
        Fraction of the user × book matrix that is empty.
        A value close to 1.0 means almost no ratings exist.
        """
        possible = self.n_users * self.n_unique_rated_books
        if possible == 0:
            return float("nan")
        return 1.0 - (self.n_ratings / possible)

    def __repr__(self) -> str:
        return (
            f"CleanDataset("
            f"users={self.n_users:,}, "
            f"with_age={self.n_users_with_age:,}, "
            f"without_age={self.n_users_without_age:,}, "
            f"books={self.n_books:,}, "
            f"ratings={self.n_ratings:,}, "
            f"sparsity={self.sparsity:.4%})"
        )


# ---------------------------------------------------------------------------
# Age cleaning constants
# ---------------------------------------------------------------------------

_AGE_MIN: int = 5    # below this → invalid (dataset has age=0)
_AGE_MAX: int = 100  # above this → invalid (dataset has age=244)

# Year cleaning constants
_YEAR_MIN: int = 1000
_YEAR_MAX: int = 2025


# ---------------------------------------------------------------------------
# Main preprocessor class
# ---------------------------------------------------------------------------

class BookCrossingPreprocessor:
    """
    Cleans and transforms raw Book Crossing DataFrames into
    analysis-ready artifacts shared by all three pipeline tasks.

    Parameters
    ----------
    age_min : int
        Minimum valid user age (inclusive). Default: 5.
    age_max : int
        Maximum valid user age (inclusive). Default: 100.
    drop_implicit_ratings : bool
        If True, ratings of 0 (implicit/no opinion) are removed before
        building the sparse matrix.  Set False to keep them for Task 1
        (the original recommender kept 0-ratings).
        Default: False (preserve original behaviour).

    Example
    -------
    >>> prep = BookCrossingPreprocessor()
    >>> clean = prep.fit_transform(raw)
    >>> print(prep.summary())
    """

    def __init__(
        self,
        age_min: int = _AGE_MIN,
        age_max: int = _AGE_MAX,
        drop_implicit_ratings: bool = False,
    ) -> None:
        self.age_min = age_min
        self.age_max = age_max
        self.drop_implicit_ratings = drop_implicit_ratings

        # Fitted state — populated during fit_transform()
        self._isbn_to_idx: Dict[str, int] = {}
        self._user_to_idx: Dict[str, int] = {}
        self._fitted: bool = False
        self._stats: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, raw: RawDataset) -> CleanDataset:
        """
        Clean all three DataFrames and build index mappings.

        This is the main entry point.  Call this once; the fitted
        isbn_to_idx and user_to_idx mappings are stored on the instance
        and reused by export_libsvm() and build_sparse_matrix().

        Parameters
        ----------
        raw : RawDataset
            Output of BookCrossingLoader.load_all().

        Returns
        -------
        CleanDataset
            Cleaned DataFrames plus index mappings.
        """
        logger.info("Starting preprocessing pipeline...")

        users   = self._clean_users(raw.users.copy())
        books   = self._clean_books(raw.books.copy())
        ratings = self._clean_ratings(raw.ratings.copy(), users)

        # Build index mappings from the cleaned ratings table
        self._build_index_maps(ratings)
        self._fitted = True

        # Split users by age availability
        users_with_age    = users[users["age"].notna()].reset_index(drop=True)
        users_without_age = users[users["age"].isna()].reset_index(drop=True)

        clean = CleanDataset(
            users=users,
            users_with_age=users_with_age,
            users_without_age=users_without_age,
            books=books,
            ratings=ratings,
            isbn_to_idx=self._isbn_to_idx,
            user_to_idx=self._user_to_idx,
            idx_to_isbn={v: k for k, v in self._isbn_to_idx.items()},
            idx_to_user={v: k for k, v in self._user_to_idx.items()},
        )

        logger.info("Preprocessing complete.\n%s", self.summary(clean))
        return clean

    def build_sparse_matrix(
        self,
        clean: CleanDataset,
        user_subset: Optional[pd.DataFrame] = None,
    ) -> sparse.csr_matrix:
        """
        Build a CSR sparse matrix of shape (n_users, n_books).

        Each row is a user; each column is a book (indexed by isbn_to_idx).
        Cell values are integer ratings (0–10).

        Parameters
        ----------
        clean : CleanDataset
            Output of fit_transform().
        user_subset : pd.DataFrame, optional
            If provided, only include ratings for users in this subset
            (must have a 'user_id' column).  Used to produce the
            with-age and without-age matrices separately.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape: (n_users_in_subset, n_unique_rated_books)
        """
        self._assert_fitted()

        ratings = clean.ratings
        if user_subset is not None:
            valid_ids = set(user_subset["user_id"].astype(str))
            ratings = ratings[ratings["user_id"].isin(valid_ids)]

        rows = ratings["user_id"].map(self._user_to_idx).values
        cols = ratings["isbn"].map(self._isbn_to_idx).values
        vals = ratings["rating"].values.astype(np.float64)

        n_users = len(self._user_to_idx)
        # isbn_to_idx is 1-based (starts at 1), so the matrix needs
        # max_index + 1 columns to avoid an out-of-bounds error.
        n_books = max(self._isbn_to_idx.values()) + 1 if self._isbn_to_idx else 0

        matrix = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_users, n_books),
            dtype=np.float64,
        )

        logger.info(
            "Built sparse matrix: %d × %d  (nnz=%s, sparsity=%.4f%%)",
            matrix.shape[0],
            matrix.shape[1],
            f"{matrix.nnz:,}",
            (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100,
        )
        return matrix

    def export_libsvm(
        self,
        clean: CleanDataset,
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        """
        Write three libsvm files to output_dir.

        Files produced
        --------------
        book_ratings.libsvm
            All users.  Label column = 0 (no target for Task 1 & 2).
            One row per user; features are isbn_idx:rating pairs.

        book_ratings_with_user_age.libsvm
            Users with a known age.  Label = integer age.

        book_ratings_without_user_age.libsvm
            Users with missing age.  Label = 0 (placeholder).

        Parameters
        ----------
        clean : CleanDataset
            Output of fit_transform().
        output_dir : str | Path
            Directory to write the files into (created if absent).

        Returns
        -------
        dict[str, Path]
            Maps each logical name to its output Path.
        """
        self._assert_fitted()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}

        # ── 1. All users (Task 1 & 2) ────────────────────────────────
        p_all = out / "book_ratings.libsvm"
        self._write_libsvm(
            ratings=clean.ratings,
            users_df=clean.users,
            age_col="age",
            use_age_as_label=False,
            output_path=p_all,
        )
        paths["all"] = p_all
        logger.info("Written: %s", p_all)

        # ── 2. Users with known age (Task 3 — training set) ──────────
        p_with = out / "book_ratings_with_user_age.libsvm"
        self._write_libsvm(
            ratings=clean.ratings,
            users_df=clean.users_with_age,
            age_col="age",
            use_age_as_label=True,
            output_path=p_with,
        )
        paths["with_age"] = p_with
        logger.info("Written: %s", p_with)

        # ── 3. Users with missing age (Task 3 — prediction set) ──────
        p_without = out / "book_ratings_without_user_age.libsvm"
        self._write_libsvm(
            ratings=clean.ratings,
            users_df=clean.users_without_age,
            age_col="age",
            use_age_as_label=False,
            output_path=p_without,
        )
        paths["without_age"] = p_without
        logger.info("Written: %s", p_without)

        return paths

    def export_cleaned_csvs(
        self,
        clean: CleanDataset,
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        """
        Write cleaned DataFrames to CSV for inspection / downstream use.

        Files produced (semicolon-delimited, UTF-8)
        -------------------------------------------
        users_cleaned.csv
        users_with_age.csv
        users_without_age.csv
        books_cleaned.csv
        ratings_cleaned.csv
        isbn_to_index_map.csv   — two-column mapping: isbn, index
        user_to_index_map.csv   — two-column mapping: user_id, index

        Returns
        -------
        dict[str, Path]
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}

        def _save(df: pd.DataFrame, name: str) -> Path:
            p = out / name
            df.to_csv(p, sep=";", index=False, encoding="utf-8")
            logger.info("Written: %s  (%s rows)", p, f"{len(df):,}")
            paths[name] = p
            return p

        _save(clean.users,             "users_cleaned.csv")
        _save(clean.users_with_age,    "users_with_age.csv")
        _save(clean.users_without_age, "users_without_age.csv")
        _save(clean.books,             "books_cleaned.csv")
        _save(clean.ratings,           "ratings_cleaned.csv")

        # Index mapping CSVs (useful for debugging and downstream joins)
        isbn_map = pd.DataFrame(
            list(self._isbn_to_idx.items()), columns=["isbn", "index"]
        )
        _save(isbn_map, "isbn_to_index_map.csv")

        user_map = pd.DataFrame(
            list(self._user_to_idx.items()), columns=["user_id", "index"]
        )
        _save(user_map, "user_to_index_map.csv")

        return paths

    def summary(self, clean: Optional[CleanDataset] = None) -> str:
        """
        Return a human-readable preprocessing summary.

        If clean is provided, includes post-cleaning statistics.
        """
        lines = [
            "",
            "=" * 60,
            "  Book Crossing Preprocessor — Summary",
            "=" * 60,
        ]

        # Cleaning step stats (populated during each cleaning method)
        if self._stats:
            lines.append("  Cleaning steps:")
            for key, val in self._stats.items():
                lines.append(f"    {key:<42} {val:>8,}")

        if clean is not None:
            lines += [
                "",
                "  Cleaned dataset:",
                f"    Users total                            {clean.n_users:>8,}",
                f"    Users with age                         {clean.n_users_with_age:>8,}",
                f"    Users without age                      {clean.n_users_without_age:>8,}",
                f"    Books (unique ISBNs)                   {clean.n_books:>8,}",
                f"    Ratings                                {clean.n_ratings:>8,}",
                f"    Unique books in ratings                {clean.n_unique_rated_books:>8,}",
                f"    Matrix sparsity                        {clean.sparsity:>8.4%}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private cleaning methods
    # ------------------------------------------------------------------

    def _clean_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the users DataFrame.

        Steps (in order)
        ----------------
        1. Strip whitespace from all string columns.
        2. Drop rows where user_id is null or blank.
        3. Drop duplicate user_ids (keep first occurrence).
        4. Coerce age to float; set out-of-range values to NaN.
        """
        logger.info("Cleaning users (%s rows)...", f"{len(df):,}")
        initial = len(df)

        # 1. Strip whitespace
        df["user_id"] = df["user_id"].astype(str).str.strip()
        if "age" in df.columns:
            df["age"] = df["age"].astype(str).str.strip()
            df["age"] = df["age"].replace({"nan": pd.NA, "": pd.NA, "None": pd.NA})

        # 2. Drop null / blank user_id
        before = len(df)
        df = df[df["user_id"].notna() & (df["user_id"] != "")]
        self._log_drop("users: null/blank user_id", before, len(df))

        # 3. Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["user_id"], keep="first")
        self._log_drop("users: duplicate user_id", before, len(df))

        # 4. Coerce age to numeric and clamp range
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        before_valid_age = df["age"].notna().sum()
        df.loc[
            (df["age"] < self.age_min) | (df["age"] > self.age_max),
            "age",
        ] = pd.NA
        after_valid_age = df["age"].notna().sum()
        n_clamped = int(before_valid_age - after_valid_age)
        self._stats["users: ages set to NaN (out of range)"] = n_clamped
        logger.debug("  Ages out of [%d, %d] set to NaN: %d", self.age_min, self.age_max, n_clamped)

        self._stats["users: rows after cleaning"] = len(df)
        logger.info("  → %s rows after cleaning (dropped %s)", f"{len(df):,}", f"{initial - len(df):,}")
        return df.reset_index(drop=True)

    def _clean_books(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the books DataFrame.

        Steps
        -----
        1. Strip whitespace from isbn.
        2. Zero-pad ISBNs shorter than 10 characters.
        3. Drop duplicate ISBNs (keep first).
        4. Coerce year to int; set implausible years to NaN.
        """
        logger.info("Cleaning books (%s rows)...", f"{len(df):,}")
        initial = len(df)

        # 1. Strip whitespace
        df["isbn"] = df["isbn"].astype(str).str.strip()

        # 2. Zero-pad short ISBNs
        before_short = (df["isbn"].str.len() < 10).sum()
        df["isbn"] = df["isbn"].apply(self._pad_isbn)
        self._stats["books: ISBNs zero-padded to 10 chars"] = int(before_short)
        if before_short:
            logger.debug("  Zero-padded %d ISBNs to 10 characters", before_short)

        # 3. Drop duplicate ISBNs
        before = len(df)
        df = df.drop_duplicates(subset=["isbn"], keep="first")
        self._log_drop("books: duplicate isbn", before, len(df))

        # 4. Coerce year to numeric; invalidate implausible values
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df.loc[
            (df["year"] <= _YEAR_MIN) | (df["year"] > _YEAR_MAX),
            "year",
        ] = pd.NA
        df["year"] = df["year"].astype("Int64")  # nullable integer

        self._stats["books: rows after cleaning"] = len(df)
        logger.info("  → %s rows after cleaning (dropped %s)", f"{len(df):,}", f"{initial - len(df):,}")
        return df.reset_index(drop=True)

    def _clean_ratings(
        self,
        df: pd.DataFrame,
        clean_users: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Clean the ratings DataFrame.

        Steps
        -----
        1. Strip whitespace from user_id and isbn.
        2. Zero-pad ISBNs shorter than 10 characters.
        3. Drop rows where user_id, isbn, or rating is null/blank.
        4. Coerce rating to numeric; drop rows outside [0, 10].
        5. Drop duplicate (user_id, isbn) pairs (keep first).
        6. Drop ratings for users not in the cleaned users table.
        7. (Optional) drop implicit ratings where rating == 0.
        """
        logger.info("Cleaning ratings (%s rows)...", f"{len(df):,}")
        initial = len(df)

        # 1. Strip whitespace
        df["user_id"] = df["user_id"].astype(str).str.strip()
        df["isbn"]    = df["isbn"].astype(str).str.strip()
        df["rating"]  = df["rating"].astype(str).str.strip()

        # 2. Zero-pad ISBNs (must match books after same padding)
        df["isbn"] = df["isbn"].apply(self._pad_isbn)

        # 3. Drop null / blank in key columns
        before = len(df)
        df = df[
            df["user_id"].notna() & (df["user_id"] != "") &
            df["isbn"].notna()    & (df["isbn"]    != "") &
            df["rating"].notna()  & (df["rating"]  != "")
        ]
        self._log_drop("ratings: null/blank user_id|isbn|rating", before, len(df))

        # 4. Coerce rating to int; drop out-of-range
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        before = len(df)
        df = df[df["rating"].notna() & (df["rating"] >= 0) & (df["rating"] <= 10)]
        df["rating"] = df["rating"].astype(int)
        self._log_drop("ratings: non-numeric or out-of-range rating", before, len(df))

        # 5. Drop duplicate (user_id, isbn) pairs
        before = len(df)
        df = df.drop_duplicates(subset=["user_id", "isbn"], keep="first")
        self._log_drop("ratings: duplicate (user_id, isbn)", before, len(df))

        # 6. Keep only ratings for users present in cleaned users table
        valid_user_ids = set(clean_users["user_id"].astype(str))
        before = len(df)
        df = df[df["user_id"].isin(valid_user_ids)]
        self._log_drop("ratings: user_id not in cleaned users", before, len(df))

        # 7. Optionally drop implicit ratings (rating == 0)
        if self.drop_implicit_ratings:
            before = len(df)
            df = df[df["rating"] > 0]
            self._log_drop("ratings: implicit (rating == 0) dropped", before, len(df))

        self._stats["ratings: rows after cleaning"] = len(df)
        logger.info("  → %s rows after cleaning (dropped %s)", f"{len(df):,}", f"{initial - len(df):,}")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Index map builder
    # ------------------------------------------------------------------

    def _build_index_maps(self, ratings: pd.DataFrame) -> None:
        """
        Build isbn→int and user_id→int mappings from the cleaned ratings.

        ISBNs are indexed 1-based (index 0 reserved for the libsvm
        label column convention used in dump_svmlight_file).
        User IDs are indexed 0-based (row index in the sparse matrix).
        """
        unique_isbns = ratings["isbn"].unique()
        self._isbn_to_idx = {isbn: idx + 1 for idx, isbn in enumerate(unique_isbns)}

        unique_users = ratings["user_id"].unique()
        self._user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}

        logger.info(
            "Index maps built: %s unique ISBNs, %s unique users",
            f"{len(self._isbn_to_idx):,}",
            f"{len(self._user_to_idx):,}",
        )

    # ------------------------------------------------------------------
    # libsvm writer
    # ------------------------------------------------------------------

    def _write_libsvm(
        self,
        ratings: pd.DataFrame,
        users_df: pd.DataFrame,
        age_col: str,
        use_age_as_label: bool,
        output_path: Path,
    ) -> None:
        """
        Write one libsvm file.

        Format per line
        ---------------
            <label> <isbn_idx>:<rating> <isbn_idx>:<rating> ...

        • label  = float age if use_age_as_label else 0.
        • Only users with at least one explicit rating (rating > 0) are
          written.
        • Features are written in ascending index order.

        Parameters
        ----------
        ratings : pd.DataFrame
            Full cleaned ratings table.
        users_df : pd.DataFrame
            Subset of users to include (must have user_id and age cols).
        use_age_as_label : bool
            If True the age value is the label; otherwise 0.
        output_path : Path
            File to write.
        """
        users_set   = set(users_df["user_id"].astype(str))
        age_map: Dict[str, float] = {}

        if use_age_as_label and age_col in users_df.columns:
            age_map = (
                users_df
                .set_index("user_id")[age_col]
                .dropna()
                .to_dict()
            )

        # Pre-filter ratings to only the users we want
        subset = ratings[ratings["user_id"].isin(users_set)].copy()

        # Add isbn index column
        subset["isbn_idx"] = subset["isbn"].map(self._isbn_to_idx)

        rows_written = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for user_id, group in subset.groupby("user_id", sort=False):
                # Only include non-zero (explicit) ratings as features
                explicit = group[group["rating"] > 0].copy()
                if explicit.empty:
                    continue

                label = age_map.get(str(user_id), 0)

                # Sort by feature index (libsvm convention)
                explicit = explicit.sort_values("isbn_idx")
                features = " ".join(
                    f"{int(row.isbn_idx)}:{int(row.rating)}"
                    for _, row in explicit.iterrows()
                )
                fh.write(f"{label} {features}\n")
                rows_written += 1

        logger.info(
            "  %s — %s users written",
            output_path.name,
            f"{rows_written:,}",
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_isbn(isbn: str) -> str:
        """
        Zero-pad an ISBN to exactly 10 characters if it is shorter.
        ISBNs longer than 10 characters (ISBN-13 etc.) are left as-is.

        Examples
        --------
        '342310538'  → '0342310538'
        '20103389'   → '0020103389'   (8 chars → pad 2 zeros)
        '0195153448' → '0195153448'   (already 10)
        '9780195153' → '9780195153'   (13-char ISBN-13, left unchanged)
        """
        isbn = str(isbn).strip()
        if len(isbn) < 10:
            isbn = isbn.zfill(10)
        return isbn

    def _log_drop(self, reason: str, before: int, after: int) -> None:
        """Log how many rows were dropped and why; store in self._stats."""
        dropped = before - after
        self._stats[reason] = dropped
        if dropped:
            logger.debug("  Dropped %d rows — %s", dropped, reason)

    def _assert_fitted(self) -> None:
        """Raise if fit_transform() has not been called yet."""
        if not self._fitted:
            raise RuntimeError(
                "fit_transform() must be called before this method."
            )
"""
src.data.loader
---------------
Responsible for reading the three raw Book Crossing CSV files from disk
and returning them as validated, lightly-typed DataFrames.

No cleaning or transformation is done here — that is the job of
BookCrossingPreprocessor.  The loader's only jobs are:

  1. Find the files (flexible path resolution).
  2. Parse them with the correct delimiter and encoding.
  3. Rename columns to a canonical snake_case schema.
  4. Run lightweight structural validation (required columns present,
     non-empty files) and surface clear error messages.
  5. Return a named container so downstream code never has to guess
     which DataFrame is which.

Original data sources
---------------------
  Users.csv   — semicolon-delimited, columns: User-ID, Age
  Books.csv   — semicolon-delimited, columns: ISBN, Title, Author,
                Year, Publisher
  Ratings.csv — semicolon-delimited, columns: User-ID, ISBN, Rating

All three files use  sep=';'  and may contain UTF-8 or Latin-1
characters (the dataset has known encoding quirks in book titles).

Example
-------
>>> loader = BookCrossingLoader("data/raw")
>>> data = loader.load_all()
>>> data.users.head()
>>> data.books.head()
>>> data.ratings.head()
>>> print(loader.summary())
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public container returned by BookCrossingLoader.load_all()
# ---------------------------------------------------------------------------

@dataclass
class RawDataset:
    """
    Thin container holding the three raw Book Crossing DataFrames.

    Attributes
    ----------
    users : pd.DataFrame
        Columns: user_id (str), age (str | NaN).
        Shape: ~278 K rows × 2 cols.

    books : pd.DataFrame
        Columns: isbn (str), title (str), author (str), year (str),
        publisher (str).
        Shape: ~271 K rows × 5 cols.

    ratings : pd.DataFrame
        Columns: user_id (str), isbn (str), rating (str).
        Shape: ~1.15 M rows × 3 cols.

    data_dir : Path
        Directory from which the files were loaded.
    """

    users: pd.DataFrame
    books: pd.DataFrame
    ratings: pd.DataFrame
    data_dir: Path

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_users(self) -> int:
        """Number of unique users in the users table."""
        return self.users["user_id"].nunique()

    @property
    def n_books(self) -> int:
        """Number of unique ISBNs in the books table."""
        return self.books["isbn"].nunique()

    @property
    def n_ratings(self) -> int:
        """Total number of rating rows."""
        return len(self.ratings)

    def __repr__(self) -> str:
        return (
            f"RawDataset("
            f"users={self.n_users:,}, "
            f"books={self.n_books:,}, "
            f"ratings={self.n_ratings:,}, "
            f"data_dir='{self.data_dir}')"
        )


# ---------------------------------------------------------------------------
# Column schema: original CSV name → canonical internal name
# ---------------------------------------------------------------------------

_USERS_RENAME: dict[str, str] = {
    "User-ID": "user_id",
    "Age":     "age",
}

_BOOKS_RENAME: dict[str, str] = {
    "ISBN":      "isbn",
    "Title":     "title",
    "Author":    "author",
    "Year":      "year",
    "Publisher": "publisher",
}

_RATINGS_RENAME: dict[str, str] = {
    "User-ID": "user_id",
    "ISBN":    "isbn",
    "Rating":  "rating",
}

# Required columns after renaming (used for validation)
_REQUIRED_USERS   = {"user_id", "age"}
_REQUIRED_BOOKS   = {"isbn", "title", "author", "year", "publisher"}
_REQUIRED_RATINGS = {"user_id", "isbn", "rating"}


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class BookCrossingLoader:
    """
    Loads the three raw Book Crossing CSV files from a directory.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing Users.csv, Books.csv, and Ratings.csv.
        The loader performs case-insensitive filename matching so that
        'users.csv', 'Users.csv', and 'USERS.CSV' all resolve correctly.

    users_filename : str, optional
        Override the default filename for the users file.
        Default: 'Users.csv'.

    books_filename : str, optional
        Override the default filename for the books file.
        Default: 'Books.csv'.

    ratings_filename : str, optional
        Override the default filename for the ratings file.
        Default: 'Ratings.csv'.

    encoding : str, optional
        Character encoding passed to pd.read_csv.
        Default: 'latin-1' (handles special characters in book titles).

    Example
    -------
    >>> loader = BookCrossingLoader("data/raw")
    >>> data = loader.load_all()
    >>> print(data)
    RawDataset(users=278,859, books=271,379, ratings=1,149,780, ...)
    """

    # Default filenames (case-insensitive match attempted first)
    _DEFAULT_USERS   = "Users.csv"
    _DEFAULT_BOOKS   = "Books.csv"
    _DEFAULT_RATINGS = "Ratings.csv"

    def __init__(
        self,
        data_dir: str | Path,
        users_filename:   Optional[str] = None,
        books_filename:   Optional[str] = None,
        ratings_filename: Optional[str] = None,
        encoding: str = "latin-1",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._users_fn   = users_filename   or self._DEFAULT_USERS
        self._books_fn   = books_filename   or self._DEFAULT_BOOKS
        self._ratings_fn = ratings_filename or self._DEFAULT_RATINGS
        self.encoding    = encoding

        self._users:   Optional[pd.DataFrame] = None
        self._books:   Optional[pd.DataFrame] = None
        self._ratings: Optional[pd.DataFrame] = None

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"data_dir does not exist: {self.data_dir}\n"
                f"Create it and place Users.csv, Books.csv, and Ratings.csv inside."
            )

        logger.info("BookCrossingLoader initialised — data_dir: %s", self.data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self, verbose: bool = True) -> RawDataset:
        """
        Load all three CSV files and return a RawDataset.

        Parameters
        ----------
        verbose : bool
            If True, log a summary table after loading.

        Returns
        -------
        RawDataset
            Container with .users, .books, .ratings DataFrames.
        """
        self._users   = self._load_users()
        self._books   = self._load_books()
        self._ratings = self._load_ratings()

        dataset = RawDataset(
            users=self._users,
            books=self._books,
            ratings=self._ratings,
            data_dir=self.data_dir,
        )

        if verbose:
            logger.info(self._format_summary(dataset))

        return dataset

    def load_users(self) -> pd.DataFrame:
        """Load and return only the users DataFrame."""
        self._users = self._load_users()
        return self._users

    def load_books(self) -> pd.DataFrame:
        """Load and return only the books DataFrame."""
        self._books = self._load_books()
        return self._books

    def load_ratings(self) -> pd.DataFrame:
        """Load and return only the ratings DataFrame."""
        self._ratings = self._load_ratings()
        return self._ratings

    def summary(self) -> str:
        """
        Return a human-readable summary string of what has been loaded.
        Raises RuntimeError if load_all() has not been called yet.
        """
        if any(df is None for df in [self._users, self._books, self._ratings]):
            raise RuntimeError("Call load_all() before summary().")
        dataset = RawDataset(self._users, self._books, self._ratings, self.data_dir)
        return self._format_summary(dataset)

    # ------------------------------------------------------------------
    # Private loaders — one per file
    # ------------------------------------------------------------------

    def _load_users(self) -> pd.DataFrame:
        """
        Load Users.csv.

        Canonical columns after loading
        --------------------------------
        user_id : str   — unique user identifier
        age     : str   — raw age string; may be empty/NaN
        """
        path = self._resolve_path(self._users_fn)
        logger.info("Loading users from: %s", path)

        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,           # keep everything as strings — let preprocessor cast
            encoding=self.encoding,
            low_memory=False,
        )

        df = self._rename_and_validate(df, _USERS_RENAME, _REQUIRED_USERS, "users")
        logger.info("  → %s rows loaded", f"{len(df):,}")
        return df

    def _load_books(self) -> pd.DataFrame:
        """
        Load Books.csv.

        Canonical columns after loading
        --------------------------------
        isbn      : str — 10-character ISBN (may have length anomalies)
        title     : str
        author    : str — may be NaN for ~2 rows
        year      : str — publication year as string (may be '0' or '2050')
        publisher : str — may be NaN for ~2 rows
        """
        path = self._resolve_path(self._books_fn)
        logger.info("Loading books from: %s", path)

        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            encoding=self.encoding,
            low_memory=False,
        )

        df = self._rename_and_validate(df, _BOOKS_RENAME, _REQUIRED_BOOKS, "books")
        logger.info("  → %s rows loaded", f"{len(df):,}")
        return df

    def _load_ratings(self) -> pd.DataFrame:
        """
        Load Ratings.csv.

        Canonical columns after loading
        --------------------------------
        user_id : str — matches user_id in users
        isbn    : str — matches isbn in books (length may vary 9–13 chars)
        rating  : str — integer 0–10 as string
        """
        path = self._resolve_path(self._ratings_fn)
        logger.info("Loading ratings from: %s", path)

        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            encoding=self.encoding,
            low_memory=False,
        )

        df = self._rename_and_validate(df, _RATINGS_RENAME, _REQUIRED_RATINGS, "ratings")
        logger.info("  → %s rows loaded", f"{len(df):,}")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, filename: str) -> Path:
        """
        Resolve a filename inside data_dir.

        Tries an exact match first, then a case-insensitive scan of the
        directory so that 'users.csv' and 'Users.csv' both work.
        """
        exact = self.data_dir / filename
        if exact.exists():
            return exact

        # Case-insensitive fallback
        lower = filename.lower()
        for entry in self.data_dir.iterdir():
            if entry.name.lower() == lower:
                logger.debug("Resolved '%s' → '%s' (case-insensitive)", filename, entry.name)
                return entry

        raise FileNotFoundError(
            f"Could not find '{filename}' in {self.data_dir}.\n"
            f"Files present: {[e.name for e in self.data_dir.iterdir()]}"
        )

    @staticmethod
    def _rename_and_validate(
        df: pd.DataFrame,
        rename_map: dict[str, str],
        required: set[str],
        name: str,
    ) -> pd.DataFrame:
        """
        Rename columns using rename_map, then verify all required
        canonical column names are present.

        Raises
        ------
        ValueError
            If required columns are missing after renaming.
        """
        # Strip whitespace from column headers (common CSV artefact)
        df.columns = df.columns.str.strip()

        # Rename only columns that actually exist in this file
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=actual_rename)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[{name}] Missing expected columns after rename: {missing}\n"
                f"Columns found: {list(df.columns)}"
            )

        # Warn about unexpected extra columns (do not drop them)
        extra = set(df.columns) - required
        if extra:
            logger.debug("[%s] Extra columns found (kept): %s", name, extra)

        # Guard against completely empty files
        if len(df) == 0:
            raise ValueError(f"[{name}] File loaded but contains zero rows.")

        return df

    @staticmethod
    def _format_summary(dataset: RawDataset) -> str:
        """Format a multiline summary string."""
        age_col = dataset.users["age"]
        n_with_age    = age_col.notna().sum()
        n_without_age = age_col.isna().sum()

        lines = [
            "",
            "=" * 52,
            "  Book Crossing Raw Dataset — Load Summary",
            "=" * 52,
            f"  Source directory : {dataset.data_dir}",
            f"  Users            : {dataset.n_users:>10,}",
            f"    ├─ with age    : {n_with_age:>10,}",
            f"    └─ without age : {n_without_age:>10,}",
            f"  Books            : {dataset.n_books:>10,}",
            f"  Ratings          : {dataset.n_ratings:>10,}",
            "=" * 52,
        ]
        return "\n".join(lines)
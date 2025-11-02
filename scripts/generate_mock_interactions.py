# ==================== scripts/generate_mock_interactions.py ====================
"""
Utility script to seed mock user interaction data into the database.

It generates ratings, reading history, and favorites for a set of users so
that the recommendation pipeline has enough implicit feedback to train.

Usage (inside project root):
    python scripts/generate_mock_interactions.py --ratings-per-user 20 --dry-run
    python scripts/generate_mock_interactions.py --ratings-per-user 20 --insert

Adjust table or column names if your schema differs.
"""

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import get_settings
from src.utils.logging_config import logger


@dataclass
class InteractionBatch:
    ratings: pd.DataFrame
    history: pd.DataFrame
    favorites: pd.DataFrame


def fetch_ids(engine, schema: str, table: str, column: str, where: Optional[str] = None) -> List[int]:
    """Fetch IDs from a table."""
    clause = f" WHERE {where}" if where else ""
    query = text(f"SELECT {column} FROM {schema}.{table}{clause}")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    ids = [row[0] for row in rows]
    if not ids:
        raise ValueError(f"No IDs found in {schema}.{table}.{column}")
    return ids


def choose_books(book_ids: Sequence[int], count: int) -> List[int]:
    """Sample book IDs without exceeding available pool."""
    count = min(count, len(book_ids))
    return random.sample(book_ids, count)


def generate_interactions(
    user_ids: Sequence[int],
    book_ids: Sequence[int],
    ratings_per_user: int,
    history_per_user: int,
    favorites_per_user: int,
    history_ts_column: str,
) -> InteractionBatch:
    """Create mock interactions for each user."""
    now = datetime.utcnow()
    ratings_records = []
    history_records = []
    favorites_records = []

    for user_id in user_ids:
        rating_books = choose_books(book_ids, ratings_per_user)
        history_books = choose_books(book_ids, history_per_user)
        favorite_books = choose_books(book_ids, favorites_per_user)

        for offset, book_id in enumerate(rating_books):
            ts = now - timedelta(days=random.randint(0, 365), minutes=offset)
            ratings_records.append(
                {
                    "user_id": user_id,
                    "book_id": book_id,
                    "rating_value": random.randint(1, 5),
                    "created_at": ts,
                }
            )

        for offset, book_id in enumerate(history_books):
            ts = now - timedelta(days=random.randint(0, 180), minutes=offset)
            history_records.append(
                {
                    "user_id": user_id,
                    "book_id": book_id,
                    "progress": random.randint(10, 100),
                    history_ts_column: ts,
                }
            )

        for offset, book_id in enumerate(favorite_books):
            ts = now - timedelta(days=random.randint(0, 90), minutes=offset)
            favorites_records.append(
                {
                    "user_id": user_id,
                    "book_id": book_id,
                    "added_at": ts,
                }
            )

    ratings_df = pd.DataFrame(ratings_records)
    history_df = pd.DataFrame(history_records)
    favorites_df = pd.DataFrame(favorites_records)

    logger.info(
        "Generated %d ratings, %d history events, %d favorites",
        len(ratings_df),
        len(history_df),
        len(favorites_df),
    )

    return InteractionBatch(ratings=ratings_df, history=history_df, favorites=favorites_df)


def insert_dataframe(df: pd.DataFrame, engine, schema: str, table: str):
    """Append dataframe rows into target table."""
    if df.empty:
        logger.warning("No records to insert for %s", table)
        return
    df.to_sql(table, engine, schema=schema, if_exists="append", index=False, method="multi")
    logger.info("Inserted %d rows into %s.%s", len(df), schema, table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock user interaction data.")
    parser.add_argument("--schema", type=str, help="Database schema (override .env)", default=None)
    parser.add_argument("--ratings-per-user", type=int, default=15, help="Number of ratings per user")
    parser.add_argument("--history-per-user", type=int, default=10, help="Reading history events per user")
    parser.add_argument("--favorites-per-user", type=int, default=5, help="Favorites per user")
    parser.add_argument(
        "--users",
        type=int,
        nargs="*",
        help="Explicit list of user IDs. If omitted, script fetches from users table.",
    )
    parser.add_argument("--users-table", type=str, default="users", help="Source table for user IDs")
    parser.add_argument("--books-table", type=str, default="books", help="Source table for book IDs")
    parser.add_argument(
        "--users-where",
        type=str,
        default=None,
        help="Optional SQL WHERE clause to filter users (without the WHERE keyword)",
    )
    parser.add_argument(
        "--books-where",
        type=str,
        default="is_deleted = FALSE",
        help="Optional SQL WHERE clause to filter books (without the WHERE keyword)",
    )
    parser.add_argument(
        "--history-timestamp-column",
        type=str,
        default="last_read_at",
        help="Column name for reading history timestamps",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only generate data; do not insert")
    parser.add_argument(
        "--insert",
        action="store_true",
        help="Insert generated data into ratings, reading_history, favorites tables",
    )
    parser.add_argument("--rng-seed", type=int, default=None, help="Seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()

    settings = get_settings()
    schema = args.schema or settings.db_schema

    if args.rng_seed is not None:
        random.seed(args.rng_seed)
        np.random.seed(args.rng_seed)

    engine = create_engine(settings.db_uri)

    users_where = args.users_where.strip() if args.users_where else None
    books_where = args.books_where.strip() if args.books_where else None

    if args.users:
        user_ids = args.users
    else:
        user_ids = fetch_ids(engine, schema, args.users_table, "user_id", users_where)

    book_ids = fetch_ids(engine, schema, args.books_table, "book_id", books_where)

    logger.info("Preparing interactions for %d users across %d books", len(user_ids), len(book_ids))
    batch = generate_interactions(
        user_ids=user_ids,
        book_ids=book_ids,
        ratings_per_user=args.ratings_per_user,
        history_per_user=args.history_per_user,
        favorites_per_user=args.favorites_per_user,
        history_ts_column=args.history_timestamp_column,
    )

    if args.dry_run and not args.insert:
        logger.info("Dry run only. Sample ratings head:\n%s", batch.ratings.head())
        logger.info("Sample history head:\n%s", batch.history.head())
        logger.info("Sample favorites head:\n%s", batch.favorites.head())
        return

    if args.insert:
        logger.info("Inserting generated data into database...")
        insert_dataframe(batch.ratings, engine, schema, "ratings")
        insert_dataframe(batch.history, engine, schema, "reading_history")
        insert_dataframe(batch.favorites, engine, schema, "favorites")
        logger.info("Mock interaction seeding complete.")
    else:
        output_dir = Path("artifacts") / "mock_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch.ratings.to_csv(output_dir / f"ratings_{timestamp}.csv", index=False)
        batch.history.to_csv(output_dir / f"history_{timestamp}.csv", index=False)
        batch.favorites.to_csv(output_dir / f"favorites_{timestamp}.csv", index=False)
        logger.info("Saved mock data CSVs under %s", output_dir)


if __name__ == "__main__":
    main()

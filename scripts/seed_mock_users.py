# ==================== scripts/seed_mock_users.py ====================
"""
Seed mock user rows into the database so that the recommender has
enough profiles to train on. The script uses fixed defaults for role
and password so you can simply run:

    python scripts/seed_mock_users.py

It will insert records unless you pass `--dry-run` or `--no-insert`.
"""

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import get_settings
from src.utils.logging_config import logger

# Hard-coded defaults for mandatory columns
ROLE_COLUMN = "role_id"
PASSWORD_COLUMN = "password"
DEFAULT_ROLE_ID = 2
DEFAULT_PASSWORD = "$2a$14$JJnowhI5UDW4iW7bsSEhguYVbZnP.N/bhGONRTpn6yO/9aw4Xo/mG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed mock users into the database.")
    parser.add_argument("--schema", type=str, help="Database schema (override .env)", default=None)
    parser.add_argument("--table", type=str, default="users", help="Target table name")
    parser.add_argument("--count", type=int, default=10, help="Number of users to insert")
    parser.add_argument("--prefix", type=str, default="mockuser", help="Prefix for usernames/emails")
    parser.add_argument("--id-column", type=str, default="user_id", help="Primary key column name (blank to skip)")
    parser.add_argument("--name-column", type=str, default="username", help="Display name column")
    parser.add_argument("--email-column", type=str, default="email", help="Email column")
    parser.add_argument(
        "--created-at-column",
        type=str,
        default="created_at",
        help="Timestamp column (blank to skip)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview generated rows without inserting")
    parser.add_argument(
        "--no-insert",
        action="store_true",
        help="Generate data but write to CSV instead of inserting",
    )
    parser.add_argument("--rng-seed", type=int, default=None, help="Seed for reproducibility")
    return parser.parse_args()


def get_next_id(conn, schema: str, table: str, id_column: Optional[str]) -> int:
    """Return the next available ID (max + 1)."""
    if not id_column:
        return 0
    result = conn.execute(text(f"SELECT COALESCE(MAX({id_column}), 0) FROM {schema}.{table}"))
    max_id = result.scalar_one()
    return int(max_id) + 1


def generate_users(
    start_id: int,
    count: int,
    prefix: str,
    id_column: Optional[str],
    name_column: str,
    email_column: str,
    created_at_column: Optional[str],
) -> List[Dict[str, object]]:
    """Create mock user rows with fixed role/password columns."""
    now = datetime.utcnow()
    rows: List[Dict[str, object]] = []

    for idx in range(count):
        user_number = start_id + idx if id_column else idx
        username = f"{prefix}_{user_number:03d}"
        email = f"{username}@example.com"
        created_at = now - timedelta(days=random.randint(0, 365), minutes=random.randint(0, 1440))

        row: Dict[str, object] = {
            name_column: username,
            email_column: email,
            ROLE_COLUMN: DEFAULT_ROLE_ID,
            PASSWORD_COLUMN: DEFAULT_PASSWORD,
        }

        if id_column:
            row[id_column] = start_id + idx

        if created_at_column:
            row[created_at_column] = created_at

        rows.append(row)

    return rows


def main():
    args = parse_args()

    if args.rng_seed is not None:
        random.seed(args.rng_seed)

    settings = get_settings()
    schema = args.schema or settings.db_schema

    engine = create_engine(settings.db_uri)

    with engine.begin() as conn:
        start_id = get_next_id(conn, schema, args.table, args.id_column)
        logger.info(
            "Generating %d users for %s.%s (starting id %s)",
            args.count,
            schema,
            args.table,
            start_id if args.id_column else "auto",
        )
        rows = generate_users(
            start_id=start_id,
            count=args.count,
            prefix=args.prefix,
            id_column=args.id_column if args.id_column else None,
            name_column=args.name_column,
            email_column=args.email_column,
            created_at_column=args.created_at_column if args.created_at_column else None,
        )

        df = pd.DataFrame(rows)
        logger.info("Preview:\n%s", df.head())

        if args.dry_run:
            logger.info("Dry run requested. No rows inserted.")
            return

        if args.no_insert:
            output_dir = Path("artifacts") / "mock_data"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"users_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            logger.info("Saved mock user CSV to %s", output_file)
            return

        df.to_sql(
            args.table,
            conn,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
        )
        logger.info("Inserted %d mock users into %s.%s", len(df), schema, args.table)


if __name__ == "__main__":
    main()

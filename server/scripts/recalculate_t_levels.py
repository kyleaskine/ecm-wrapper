#!/usr/bin/env python3
"""
Recalculate current_t_level for a list of composites.

This is the targeted alternative to the recalculate-all endpoint: each listed
composite gets the same recalculation a submission runs
(TLevelCalculator.recalculate_composite_t_level), in its own transaction,
holding the composite's row lock as a submission does. target_t_level is left
alone (the admin recalculate-t-level endpoint also resets it).

The default is a dry run: it prints each change and rolls back. Pass --apply
to commit.

Usage (IDS_FILE has one composite id per line; blank lines and # comments are
skipped):
    python3 scripts/recalculate_t_levels.py IDS_FILE [--apply] [--pause SECONDS]

On the droplet, where the image does not contain scripts/:
    docker cp scripts/recalculate_t_levels.py server-api-1:/app/
    docker cp ids.txt server-api-1:/tmp/ids.txt
    docker exec server-api-1 python recalculate_t_levels.py /tmp/ids.txt
    docker exec server-api-1 python recalculate_t_levels.py /tmp/ids.txt --apply
"""

import argparse
import logging
import os
import sys
import time
from typing import List

# Add parent directory to path for imports (also works when copied to /app)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal  # noqa: E402
from app.models import Composite  # noqa: E402
from app.services.t_level_calculator import TLevelCalculator  # noqa: E402


class FailureCounter(logging.Handler):
    """Counts the calculator's "recalculation failed" warnings. A failed
    recalculation keeps the stored value, so its return value alone looks
    like "unchanged"."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "recalculation failed" in record.getMessage():
            self.count += 1


def read_ids(path: str) -> List[int]:
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(int(line))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalculate current_t_level for a list of composites.")
    parser.add_argument("ids_file", help="file with one composite id per line")
    parser.add_argument("--apply", action="store_true",
                        help="commit the changes (default: dry run)")
    parser.add_argument("--pause", type=float, default=0.05,
                        help="seconds to sleep between composites (default 0.05)")
    args = parser.parse_args()

    ids = read_ids(args.ids_file)
    failures = FailureCounter()
    logging.getLogger("app.services.t_level_calculator").addHandler(failures)
    calculator = TLevelCalculator()
    backend = "library" if getattr(calculator, "use_library", False) else "binary"
    print(f"{len(ids)} composites, backend {backend}, "
          f"{'APPLYING' if args.apply else 'dry run'}", flush=True)

    changed = unchanged = missing = failed = 0
    db = SessionLocal()
    try:
        for composite_id in ids:
            try:
                composite = db.query(Composite).filter(
                    Composite.id == composite_id
                ).populate_existing().with_for_update().first()
                if composite is None:
                    missing += 1
                    print(f"{composite_id}: not found", flush=True)
                    db.rollback()
                    continue

                old = composite.current_t_level
                failures_before = failures.count
                new = calculator.recalculate_composite_t_level(db, composite)
                if new is None or failures.count != failures_before:
                    failed += 1
                    print(f"{composite_id}: calculation failed, kept {old}", flush=True)
                    db.rollback()
                    continue

                if old is not None and abs(new - old) < 0.0005:
                    unchanged += 1
                    db.rollback()
                else:
                    changed += 1
                    print(f"{composite_id}: {old} -> {new}", flush=True)
                    if args.apply:
                        db.commit()
                    else:
                        db.rollback()
            except Exception as e:  # keep going; report at the end
                failed += 1
                db.rollback()
                print(f"{composite_id}: error {e}", flush=True)
            if args.pause:
                time.sleep(args.pause)
    finally:
        db.close()

    verb = "updated" if args.apply else "would change"
    print(f"{verb} {changed}, unchanged {unchanged}, not found {missing}, "
          f"failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

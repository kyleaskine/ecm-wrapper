#!/usr/bin/env python3
"""
Calculate missing group orders.

/submit_result calculates a new factor's group order in a background task
after the submission commits (see fill_group_orders in
app/services/factors.py). If that task never runs, because the process was
killed first, or if PARI/GP fails, the factor keeps a NULL group order and
nothing retries it. This script finds those factors and runs the calculation
again, one factor at a time.

Factors without a sigma that were not found by P-1/P+1 have no group order to
calculate and are skipped.

The default is a dry run: it lists what it would calculate. Pass --apply to
run PARI/GP and store the results.

Usage:
    python3 scripts/backfill_group_orders.py [--days N] [--limit N] [--apply] [--pause SECONDS]

On the droplet, where the image does not contain scripts/:
    docker cp scripts/backfill_group_orders.py server-api-1:/app/
    docker exec server-api-1 python backfill_group_orders.py --days 7
    docker exec server-api-1 python backfill_group_orders.py --days 7 --apply
"""

import argparse
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta

# Add parent directory to path for imports (also works when copied to /app)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, engine  # noqa: E402
from app.models import Factor  # noqa: E402
from app.services.factors import FactorService, fill_group_orders  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate missing group orders.")
    parser.add_argument("--days", type=float, default=None,
                        help="only factors created in the last N days (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="at most N factors")
    parser.add_argument("--apply", action="store_true",
                        help="run PARI/GP and store results (default: dry run)")
    parser.add_argument("--pause", type=float, default=0.2,
                        help="seconds to sleep between factors (default 0.2)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    with SessionLocal() as db:
        query = db.query(Factor).filter(Factor.group_order.is_(None))
        if args.days is not None:
            query = query.filter(Factor.created_at >= datetime.utcnow() - timedelta(days=args.days))
        query = query.order_by(Factor.id.desc())
        if args.limit is not None:
            query = query.limit(args.limit)
        jobs = []
        for factor in query.all():
            job = FactorService.missing_group_order_job(db, factor)
            if job is not None:
                jobs.append(job)

    kinds = Counter("ecm" if j.sigma is not None else j.method for j in jobs)
    print(f"{len(jobs)} factors need a group order "
          f"({', '.join(f'{n} {k}' for k, n in sorted(kinds.items())) or 'none'}); "
          f"{'APPLYING' if args.apply else 'dry run'}", flush=True)
    if not args.apply:
        for job in jobs[:20]:
            print(f"  factor {job.factor_id}: {job.factor[:30]} "
                  f"({'sigma ' + job.sigma[:20] + ', param ' + str(job.parametrization) if job.sigma else job.method})")
        if len(jobs) > 20:
            print(f"  ... and {len(jobs) - 20} more")
        return 0

    for job in jobs:
        fill_group_orders(engine, [job])
        if args.pause:
            time.sleep(args.pause)

    with SessionLocal() as db:
        ids = [j.factor_id for j in jobs]
        filled = db.query(Factor).filter(Factor.id.in_(ids), Factor.group_order.isnot(None)).count() if ids else 0
    print(f"stored {filled} of {len(jobs)}; {len(jobs) - filled} still failing", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Server performance plan

Implementation brief for the periodic slowness on the production ECM server.
Written 2026-11-16 from an investigation against a copy of the production
database. Nothing in this document has been implemented yet.

Revised 2026-11-17 after a review pass. Kyle has settled the t-level licence
question (not a blocker). The admin-dashboard auto-refresh claim was wrong and
has been removed. Item 10 (group-order calculation) is new. Items 1, 4, 6, 7
and 8, the decisions in section 4 and the sequencing in section 6 are updated.
Item 1 was implemented on 2026-11-18; see its status note.

You are picking this up without the investigation's context, so this file is
meant to be complete: what is slow, the evidence, what to change, how to prove
each change worked, and what needs Kyle's decision before you touch it.

## 1. Situation

Production is one DigitalOcean droplet: 1 shared vCPU, 1 GB RAM, running nginx,
PostgreSQL (version 17, inferred from settings in the dump; local dev is 16)
and a single uvicorn worker in Docker (`docker/Dockerfile`,
`docker-compose.prod.yml`). Every route does synchronous SQLAlchemy work in the
threadpool. With one vCPU, anything that burns CPU for a second stalls every
other request for that second, so the goal is fewer CPU-seconds per request,
not better concurrency.

Kyle's brief: find pages and queries that could be designed better and indexes
worth adding. A table restructure is acceptable only for a big win. **None is
needed**: every item below is an index, a query rewrite, a cache, or a change
to how the t-level calculation is invoked.

### Production data profile (dump of 2026-11-16)

| Table | Rows | Heap | Indexes | Notes |
|---|---|---|---|---|
| `ecm_attempts` | 385,397 | 97 MB | 120 MB | plus 282 MB TOAST (`raw_output`) |
| `composites` | 204,067 | 100 MB | 59 MB | 96,037 assignable, ~105,000 fully factored |
| `work_assignments` | 363,300 | 66 MB | 50 MB | ~50 active at any time |
| `ecm_residues` | 91,608 | 25 MB | 26 MB | ~50 pending |
| `factors` | 69,433 | 11 MB | 7 MB | |

- Load: 300–765 result submissions per day, about 95% ECM. Busiest hour in the
  last 30 days: 339 submissions. Busiest minute: 10.
- Attempts per composite: median 2, p90 6, p99 22, max 1,181.
- Distinct ECM `(b1, b2, parametrization)` combinations: 511 all-time, 60 in
  the last 30 days. This matters for item 1.
- `raw_output`: present on 23,884 rows (older rows were purged), 10 GB
  uncompressed, average 442 KB, median 2.9 KB, p99 at the 1 MB cap.

### Baseline timings

Measured through the real app (`TestClient`) against the production copy, on a
16-core dev machine with a warm cache and parallel query disabled. The droplet
will be several times slower and far worse when pages are not cached, so treat
these as lower bounds and compare before/after ratios, not absolutes.

| Endpoint | Wall | Dominant cost |
|---|---|---|
| `POST /submit_result` (ECM) | 535–1,450 ms | `t-level` binary: 97% of recalculation time |
| `GET /admin/dashboard` | 1,258 ms | sort of all `work_assignments`; two full aggregations of `ecm_attempts` |
| `GET /dashboard/p1-testing-status` | 1,799 ms | 96k-element `IN` list sent twice; 1,047 ms of Python |
| `GET /dashboard/` | 681 ms | three full scans of `ecm_attempts` |
| `GET /dashboard/curves` | 433 ms | two full aggregations of `ecm_attempts` |
| `GET /dashboard/leaderboard?days=365` | 412 ms | five scans of the window |
| `GET /ecm-work?work_type=progressive` | 306 ms | seq scan and sort of all composites |
| `GET /dashboard/testing-status` | 209 ms | aggregate over all composites |
| `GET /dashboard/factors` | 136 ms | distinct-clients join |
| `GET /p1-work` | 122 ms | reads 96,165 buffers |
| `GET /ecm-work` (standard, the client default) | 82 ms | reads 91,986 buffers |
| Deleting one attempt | 44 ms | three unindexed foreign-key checks |

Buffer counts are the hardware-independent number. 92,000 buffers per work
request is about 720 MB of page accesses on a server with 128 MB of shared
buffers.

## 2. Ground rules

- Routes that touch the database stay `def`, never `async def`.
- Do not call or recommend the recalculate-all-t-levels maintenance endpoint.
  It takes hours on the droplet. Recalculate per composite with
  `CompositeService.update_t_level(db, composite_id)`.
- Schema changes go through Alembic (current head `e5b1c9d7a2f4`) and are
  mirrored in `app/models/`. The test suite runs on in-memory SQLite, so every
  PostgreSQL-only index option needs its SQLite counterpart or must live only
  in the migration. Follow `uq_work_assignments_one_active_per_composite`
  (`postgresql_where` plus `sqlite_where`) and the hash index on
  `current_composite` (migration only).
- Migrations run at container start (`alembic upgrade head`). Every index here
  builds in under 250 ms on the dev machine at production size, so plain
  `CREATE INDEX` inside the migration transaction is fine. Do not use
  `CONCURRENTLY`.
- After code changes run `venv/bin/python -m pytest tests/ -v`,
  `venv/bin/python -m mypy app/` and
  `pyright --pythonpath venv/bin/python app/` from `server/`.
- Update `server/CLAUDE.md` where behaviour it documents changes.
- The locking in `submit.py`, `residue_manager.py` and
  `services/work_assignment.py` (composite row lock, then residue; recheck
  under lock) fixes real production races. Do not reorder locks or move work
  outside the transaction to gain speed. The one exception is item 10's
  group-order calculation, which takes no part in that locking (see item 10).

## 3. Work items

Ordered by expected benefit. Items 1–5 are the plan; 6–10 are smaller.
Each can ship on its own.

### 1. Run the t-level calculation in-process with a cache

**Status (2026-11-18): implemented, not yet deployed.** The code is in
`app/services/t_level_library.py`, wired in through `TLevelCalculator`, with
`T_LEVEL_BACKEND=binary` as the rollback. Tests are in
`tests/test_t_level_library.py`. Results on the production copy:
- Equivalence: the library matched the binary on all 1,028 single-class
  inputs (514 classes × 2 curve counts) and on 201 sampled composites. It
  also matched on the 592 composites whose stored value differs from a fresh
  recalculation.
- Stored values: of 121,800 composites, 592 (0.5%) recalculate more than
  0.0015 away from the stored value, with either backend. 565 come out lower;
  all of those have superseded attempts and no attempt since 2026-07-19.
  This is old drift, not caused by this change.
- Speed: a warm `POST /submit_result` takes 33–40 ms, against 449–1,158 ms
  with the binary. The first submission that needs an uncached class took
  301–531 ms.
- Memory: the package adds about 25 MB after computing every production
  class, flat after the first ~100.

A `/code-review xhigh` pass followed (2026-11-18). All of the changes it led to
are in place, and the equivalence checks above were re-run on the final code:
- The package is imported on first use, so the binary backend never loads it.
  An import-time check confirms the cache actually intercepts the package's
  calls.
- Every package cache is cleared after every calculation.
- Only `sqlite3.Error` retries with the binary. Input-dependent errors fail
  the same way in the binary, so they are not retried.
- A failed calculation now keeps the stored `current_t_level` instead of
  writing the prior. This covers the recalculation and maintenance paths
  (and the legacy `GET /work`, since removed).
- Priors outside 5–100 now give max(attempts, prior). Before, every ECM
  attempt was discarded. No production composite is affected.
- Submissions bound `b1` to 0–1e15 and `b2` to the BIGINT range, and `b2 = -1`
  (the client's default sentinel) is stored as `None`.

**Problem.** Every ECM submission recalculates the composite's t-level by
spawning `bin/t-level` (`TLevelCalculator._calculate_t_level_external`,
`app/services/t_level_calculator.py:296`). That file is a PyInstaller one-file
bundle of a Python program. Each call unpacks 28 MB into `/tmp/_MEI*`, starts
an interpreter, and then spends about 150 ms per distinct `(b1, b2, param)`
combination. It runs inside the submit transaction while holding the composite
row lock.

**Evidence.** For 40 composites sampled from the last 30 days of ECM
submissions: recalculation mean 400 ms, p90 674 ms, max 1,296 ms, of which the
binary is 97%. The base cost with a trivial input is 200–400 ms.

**Change.** Import the library and cache the expensive part.

The source is at `~/code/t-level` and on PyPI as `t-level` (version 1.0.0,
depends on `gmpy2`). The only expensive function is
`t_level.get_ecm_success_probs(b1, b2, param)`, which returns 91 success
probabilities. It depends only on its three arguments, so it can be cached for
the life of the process. Combining the vectors into a t-level is cheap
arithmetic.

A prototype that wraps that function in an `lru_cache` and calls
`t_level.get_t_level(...)` gave results identical to the deployed binary on
every case tried, including `-w`:

| Case | Binary | In-process, first call | In-process, cached |
|---|---|---|---|
| 1 class, default B2 | 209 ms | 4 ms | 0.2 ms |
| stage-1 leftover + stage 2 | 560 ms | 263 ms | 0.9 ms |
| 6 classes | 1,019 ms | 482 ms | 1.1 ms |
| with prior t-level (`-w 45`) | 575 ms | 229 ms | 1.0 ms |

Requirements:

- Add `t-level==1.0.0` to `requirements.txt`. The deployed binary reports
  0.9.6; results matched, but run the equivalence check below before trusting it.
- Put the integration in its own module. Replace the library's
  `get_ecm_success_probs` with a cached wrapper rather than copying its logic.
  The wrapper must return `tuple(...)`: the library returns a one-shot `map`
  object on its table-lookup path.
- Serialise all calls into the library with one module-level
  `threading.Lock`. The library opens a SQLite connection at import and shares
  a single cursor, and routes run on threadpool threads.
- Reproduce what the CLI does with `-w`. See `main()` in
  `~/code/t-level/t_level/__init__.py`: it converts the starting t-level with
  `get_t_level_curves_string(work, precision)` and appends that as an extra
  curve line. It rejects values outside 5–100, and today the server then
  falls back to returning the starting t-level; keep that. Production has no
  composite with a prior t-level below 5.
- Round the result to 3 decimals. The CLI prints 3, so stored values have 3.
- Pass `(curves, b1, b2, param)` tuples directly. Keep the current meanings:
  `b2 is None` is the GMP-ECM default, `b2 == 0` is stage 1 only, and a missing
  parametrization is 3. The aggregation by `(b1, b2, param)` already in
  `_calculate_t_level_external` stays.
- Never pass `param=None` into the library. `get_ecm_success_probs` turns
  `None` into 1, but the server's default is 3, so apply the server's default
  before the call. Pass `b1` and `b2` as `int`, as `_format_number_for_tlevel`
  does today.
- Keep the subprocess path as a fallback, selected by a setting and used
  automatically if the import fails, for at least one release. Rollback is
  then an environment variable.
- Bound memory. `t_level/rho.py` decorates its helpers with unbounded
  `functools.cache`. (Measured 2026-11-18: the earlier "20 → 60 MB, then flat"
  figure came from 20 similar combinations. Over all 514 production classes
  RSS grew to about 218 MB and kept growing. Clearing every `rho` cache after
  each calculation that computed a new class holds it at about 44 MB. A cold
  class then costs a median of 55 ms on the dev box, maximum 286 ms.)
- The first request for each new combination still costs 110–250 ms while
  holding the lock. That is acceptable. Fill the cache only on demand, and do
  not pre-warm at startup. On one vCPU, warming the last 30 days' 60
  combinations is over 10 s of CPU with the library lock held. It would also
  run right after every container restart, which is when the server is
  slowest.
- The cache lasts only as long as the process. Item 7's
  `--limit-max-requests` restarts the process, probably every day or two, and
  each restart empties the cache. Run item 7's log check before this ships. If
  restarts are frequent, ship item 7 in the same release. Only consider
  persisting the vectors (511 × 91 floats, a few hundred KB) if restarts have
  to stay.

**Acceptance.**

- Equivalence with the binary, on identical inputs (the main check):
  - Run every distinct production `(b1, b2, param)` combination (511) through
    both paths as a single-class input.
  - Do the same for a sample of at least 200 composites that covers
    stage-1 leftovers, six or more classes, and a prior t-level (`-w`).
  - The two must agree to the 3 printed decimals.
  - Do not run the binary over every composite: at about 400 ms a call, that
    takes hours.
- Comparison with stored values (informational only): recalculate every
  composite in the production copy in-process and report how many differ from
  the stored `current_t_level` by more than 0.0015.
  - A stored value can be stale for reasons that have nothing to do with this
    change. For example, the param-0 leftover fix (see the comment in
    `recalculate_composite_t_level`) changed results for composites last
    calculated before it.
  - A mismatch is only a defect if the binary and the in-process path disagree
    on that composite.
- `POST /submit_result` for an ECM result drops from 535–1,450 ms to tens of
  milliseconds on the benchmark in the appendix.
- Unit tests cover the wrapper without a database: default B2, stage 1 only,
  explicit B2, each parametrization, and a starting t-level inside and outside
  5–100.

**Licence: decided 2026-11-17.** `t_level/__init__.py` declares
`__license__ = "GPL"` and `rho.py` is LGPL. Kyle accepts the dependency, so
item 1 is not blocked. The repository already commits `server/bin/t-level`, a
PyInstaller build of the same code.

### 2. Partial indexes for work assignment

**Problem.** `/ecm-work` and `/p1-work` order assignable composites by
`target_t_level` and take the first free one. The planner walks
`ix_composites_target_t_level` from the low end, where the ~105,000 fully
factored composites sit, and discards them one heap fetch at a time.
Progressive mode has no usable index and sorts every composite.

**Evidence.**

| Query | Before | After |
|---|---|---|
| `/ecm-work` standard | 91,986 buffers, 105,939 rows discarded | 64 buffers |
| `/ecm-work` progressive | seq scan, 271 ms | 49 buffers |
| `/p1-work` | 96,165 buffers | 62 buffers |

**Change.** One migration, mirrored in `app/models/composites.py` and
`app/models/attempts.py`:

```sql
CREATE INDEX ix_composites_ecm_work_standard ON composites (target_t_level, created_at)
  WHERE is_active AND NOT is_fully_factored AND ecm_progress < 1.0;
CREATE INDEX ix_composites_ecm_work_progressive ON composites (current_t_level, target_t_level, digit_length)
  WHERE is_active AND NOT is_fully_factored AND ecm_progress < 1.0;
CREATE INDEX ix_composites_p1_work ON composites (target_t_level, created_at)
  WHERE is_active AND NOT is_fully_factored;
CREATE INDEX ix_ecm_attempts_p1_coverage ON ecm_attempts (composite_id, method, b1)
  INCLUDE (curves_completed) WHERE method IN ('pm1', 'pp1');
```

They total about 11 MB. The queries in `app/api/v1/ecm_work.py` need no change:
their filters already imply these predicates. `/p1-work` has no
`ecm_progress` filter, which is why it needs its own index. The fourth index
also serves item 4's P-1 page. `/p1-work?work_type=progressive` gets no index
from this item. That is fine unless some client uses it.

**Acceptance.** `EXPLAIN (ANALYZE, BUFFERS)` of each query on the production
copy shows the new index and under 200 buffers. The ordering and the row
returned are unchanged. `tests/test_work_assignment_race.py` passes.

**Implemented 2026-11-23** with item 3 and item 8's unconditional drops, as
migration `f6204945eb74` (CONCURRENTLY, IF [NOT] EXISTS). The SQL captured from
the real endpoints on the copy, run under `EXPLAIN (ANALYZE, BUFFERS)`:

| Query | Buffers | Time (warm) |
|---|---|---|
| `/ecm-work` standard | 80 | 0.4 ms |
| `/ecm-work` progressive | 76 | 0.1 ms |
| `/p1-work` (`method=p1`) | 155,000 → 52,000 | 62 → 30 ms |

`/p1-work` misses the target. The 62 buffers in the evidence table above do
not reproduce against the endpoint's SQL. The scan still passes 13,370
low-target composites that the P-1 sweep has already covered, and each one
costs a coverage probe (40,000 of the 52,000 buffers). An index cannot skip
them: the coverage test compares `b1` with a CASE on the composite's own
`target_t_level`. Fixing it would need a stored "P-1 done to B1" value on
`composites`, or a sweep cursor. That would be a separate item.

### 3. Missing indexes on foreign-key and sort columns

**Problem.** Three columns that reference `ecm_attempts.id` have no index, so
PostgreSQL scans the referencing table for every deleted attempt. Two columns
used for "newest first" listings have no index.

**Evidence.** Deleting one attempt: 44 ms of constraint checks, 0.3 ms with
the indexes. By extrapolation, deleting the composite with 1,181 attempts
costs about 52 s today. The admin dashboard's recent-work query reads 8,996 buffers and spills
to disk; with an index it reads 74.

**Change.**

```sql
CREATE INDEX ix_ecm_attempts_superseded_by ON ecm_attempts (superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX ix_factors_found_by_attempt_id ON factors (found_by_attempt_id);
CREATE INDEX ix_ecm_residues_stage1_attempt_id ON ecm_residues (stage1_attempt_id);
CREATE INDEX ix_work_assignments_created_at ON work_assignments (created_at);
CREATE INDEX ix_composites_created_at ON composites (created_at);
```

`ix_factors_found_by_attempt_id` also serves the
`Factor.found_by_attempt_id.in_(...)` lookups in `web.py`, `query_helpers.py`
and `services/composites.py`.

**Acceptance.** `EXPLAIN ANALYZE DELETE` of one attempt on the production copy
shows each constraint trigger under 1 ms. Met on 2026-11-23 (migration
`f6204945eb74`): the three triggers take 0.19, 0.04 and 0.04 ms.

### 4. Dashboard queries and caching

**Problem.** The public and admin dashboards recompute everything on every
load, and several of those queries scan whole tables.

No page refreshes itself. Commit ff41444 (2025-11-21) removed the dashboard
timers. The "Admin dashboard auto-refreshes every 30 seconds" line under
Dashboard Improvements in `server/CLAUDE.md` is stale; delete it as part of
this item. These pages therefore cost CPU only when a person or a crawler
loads them, which is why this item comes after items 1–3, 5 and 10 in
section 6.

**Changes, in order of benefit.**

a. **`get_aggregated_attempts`** (`app/utils/query_helpers.py:245`). It groups
   all 385k attempts by composite to find the 50 most recently worked, then
   runs the same aggregation again for a total. Both dashboards discard the
   total and pass `attempts_per_composite=0`.
   - Make the total optional and skip it when the caller does not use it.
   - Skip the detail-rows query when `attempts_per_composite` is 0.
   - With no `client_id`, `method`, `priority` or `project_id` filter, take
     candidates from the newest attempts only, widening the window until it
     holds enough distinct composites:

     ```sql
     SELECT composite_id, max(created_at) AS latest
     FROM (SELECT composite_id, created_at FROM ecm_attempts ORDER BY created_at DESC LIMIT 5000) t
     GROUP BY composite_id ORDER BY latest DESC LIMIT 50;
     ```

     The newest 5,000 production attempts cover 2,089 composites. With
     `CREATE INDEX ix_ecm_attempts_created_composite ON ecm_attempts (created_at DESC, composite_id)`
     this is an index-only scan of 23 buffers instead of 12,352. That index
     supersedes `ix_ecm_attempts_created_at`; drop the old one in the same
     migration.
   - Keep the existing query for the filtered cases.

b. **A short-lived cache for public pages.** There is one worker process, so
   an in-process cache is enough. The rules:
   - Cache rendered responses for the public `GET` pages in
     `app/api/v1/web.py` for 30–60 seconds, keyed by path and query string.
   - Only one request should compute a missing entry; others wait for it.
   - Cache only `200` responses (`/composites/find` redirects) and never
     admin pages.
   - Bound the cache by total bytes, not by entry count. Pages differ widely
     in size, and the host has 1 GB.
   - Leave out `/composites/{id}/details` and
     `/composites/{id}/attempts-fragment`. There are 204k possible keys, so
     the hit rate would be near zero and a crawler would churn the cache.
   - Check that no cached template contains per-request content.
     `residue_status.html` renders times relative to `now`; 60 s of staleness
     is acceptable there.

   This protects the server from crawlers and from several people loading the
   same page. Since nothing auto-refreshes, check the nginx access logs for
   crawler traffic before building it. If there is little, do 4a, 4c and 4d
   and skip this part. An nginx `proxy_cache` with `proxy_cache_lock` would do
   the same job without the request reaching Python. The nginx config is not
   in this repository, so that is Kyle's call.

c. **`/dashboard/p1-testing-status`** (`web.py:287`). It loads every active
   composite, sends all ~96,000 ids back in two `IN` lists, and builds, filters
   and paginates the result in Python. Replace the two aggregates with one
   grouped query over `method IN ('pm1', 'pp1')` with no id list, which the
   item 2 index answers in 194 buffers and about 10 ms. The required B1 already
   exists as a SQL expression (`_build_required_b1_case` in `ecm_work.py`), so
   the summary counts, the status filter and the pagination can also move into
   SQL. The page's output must not change.

d. **Counters.** `total_attempts` on the public dashboard scans all of
   `ecm_attempts`. The admin summary runs four separate counts over
   `composites`. Combine counts over the same table into one query using
   `count(*) FILTER (WHERE ...)`. Item b covers the public side.

e. **Leaderboard** (`web.py:761`). Five queries over the same window. Three of
   them (total curves, active clients, top contributors) can be one grouped
   query. Item b makes this less pressing.

**Correctness fixes to fold in** (found by the 2026-11-23 reviews; both are in
`get_aggregated_attempts`, `app/utils/query_helpers.py`):
- The grouped `/curves` view loads the `limit * attempts_per_composite` most
  recent attempts overall, not up to that many per composite. One busy
  composite can fill all 1,250 slots and leave the other rows empty. Rank per
  composite with `row_number() OVER (PARTITION BY composite_id ...)`.
- `?client_id=` only picks which composites appear. The aggregates and the
  detail rows still include every client's attempts. Filter both on
  `client_id` too.

**Acceptance.** On the appendix benchmark: public dashboard under 150 ms
uncached and under 5 ms cached, admin dashboard under 250 ms, P-1 status page
under 200 ms with identical content. These are targets, not measurements. `tests/test_html_pages.py` passes.

### 5. Stop loading `raw_output`

**Problem.** `raw_output` averages 442 KB where present and is capped at 1 MB.
No template or API response reads it, yet these queries fetch it:

- `web.py:69` and `web.py:730`: up to 25 and 50 attempts for the factor lists
  on public pages.
- `query_helpers.py:683` (`batch_fetch_attempts_by_composite`).
- `submit.py:626`: `db.refresh(attempt)` re-reads the row just inserted.
- Point lookups in `residue_manager.py` (lines 649, 694, 728, 794, 843, 866)
  and `t_level_calculator.py` (lines 478, 520, 526), most of them inside the
  submit transaction.

**Change.** Mark the column deferred on the model
(`mapped_column(Text, nullable=True, deferred=True)` in
`app/models/attempts.py`) so no query loads it unless code reads the
attribute. Then remove or narrow the `db.refresh(attempt)` in `submit.py`;
check what SQLAlchemy 2.0.23 reloads on refresh for a deferred attribute that
was set in the constructor. The per-query `defer(ECMAttempt.raw_output)`
options become redundant and can go.

**Acceptance.** With SQL logging on, no `SELECT` issued by `submit_result` or
any dashboard page names `raw_output`. The purge endpoint in
`admin/maintenance.py` still works.

**Implemented 2026-11-23.**
- The column is `deferred=True`, and the 12 per-query `defer()` options are
  gone.
- `db.refresh(attempt)` is removed from submit. SQLAlchemy 2.0's INSERT
  already ends `RETURNING id, created_at, updated_at`, so the refresh re-read
  nothing new. A refresh never reloads a deferred column; it expires it.
- `Query.count()` on the `ECMAttempt` entity still named the column, because
  it wraps every column in a subquery. Four counts now count `ECMAttempt.id`
  instead: the public dashboard total, the `/curves` ungrouped total and the
  two 24-hour counts. On the copy, the public one read only the in-row TOAST
  pointers (12,352 buffers, the heap's size), not the outputs.
- `tests/test_raw_output_deferred.py` checks the acceptance criteria. It
  covers new, factor, duplicate and residue-completing submissions, and 19
  public and admin pages. It also checks that the output is still stored and
  that the cleanup endpoint clears only old rows. All 76 distinct SELECTs from
  18 pages on the copy are clean.
- Use `func.count(ECMAttempt.id)` (or `with_entities`) for future counts, not
  `query(ECMAttempt).count()`.

### 6. One query instead of one per superseded attempt

**Problem.** `recalculate_composite_t_level`
(`t_level_calculator.py:424`) loads every attempt as an ORM object, then runs
one query per superseded stage-1 attempt to find its stage-2 attempt.

**Evidence.** Per submission in production: mean 5 statements, p90 12, max 23.
This is small next to item 1. It becomes the largest remaining cost of a
recalculation once item 1 is done.

**Change.** The calculation needs only curve totals per `(b1, b2, param)`.
Compute those with a grouped query over unsuperseded ECM attempts. Compute the
leftover stage-1 curves with one self-join from each superseded attempt to the
attempt that superseded it, where the latter completed fewer curves. Preserve
the current behaviour exactly, including the skip of attempts with `b1 <= 0`
and the raw parametrization for leftovers.
`get_current_t_level_from_attempts(attempts, starting_t_level)` is also called
from `admin/maintenance.py`; keep it working. (`services/work_assignment.py`
no longer calls it: the legacy `GET /work` was removed on 2026-11-23.)

Restrict the leftover self-join to stage-1 attempts (`b2 = 0`). Completing a
residue also supersedes other clients' stage-2 attempts on the same residue
(the orphan sweep in `complete_residue`). Today such an orphan with more
curves than the winner earns B1-only leftover credit, on top of the real
stage 1's leftover for the same curves. This was found by the 2026-11-23
review. On the copy, none of the 5,122 leftover credits come from an orphan,
so adding the condition changes no stored t-level.

Delete the residue-based branch (the loop over `completed_residues`) rather
than port it.
- It skips every residue whose stage-1 attempt the attempts-based loop already
  handled. It therefore acts only when the residue's `composite_id` differs
  from its stage-1 attempt's `composite_id`.
- In the production copy, 10 of 91,084 completed residues reach it, all through
  that mismatch, and none adds leftover curves. Deleting it changes no stored
  t-level.
- Crediting another composite's stage-1 curves would be wrong anyway.

Re-run this query on the copy before deleting the branch. Expect 10 rows, all
with `adds_leftover = f`:

```sql
SELECT r.id, r.composite_id, s1.composite_id AS s1_composite,
       s2.curves_completed < r.curve_count AS adds_leftover
FROM ecm_residues r
JOIN ecm_attempts s1 ON s1.id = r.stage1_attempt_id
LEFT JOIN ecm_attempts s2 ON s2.id = s1.superseded_by
WHERE r.status = 'completed' AND s1.superseded_by IS NOT NULL
  AND (s1.method IS DISTINCT FROM 'ecm' OR s1.composite_id <> r.composite_id);
```

The mismatch itself is a separate correctness bug and not part of this plan.
`store_residue_file` checks that the stage-1 attempt exists, not that it
belongs to the residue's composite. Completing such a residue then supersedes
the other composite's stage-1 attempt.

**Acceptance.** Recalculation issues a fixed number of statements regardless
of attempt count. On the production copy it gives the same result as the
pre-change code for every composite that has a superseded attempt.
`tests/test_residue_autocomplete.py` passes.

**Implemented 2026-11-23.**
- `_curve_classes_for_composite()` sends two grouped queries: unsuperseded ECM
  curves per (B1, B2, param) class, and leftover stage-1 curves (`b2 = 0`,
  self-join to the superseder). Rows feed `get_current_t_level_from_attempts`
  unchanged. Invalid B1s are still logged and skipped there. The residue
  branch is deleted.
- The pre-deletion query returned 10 rows on the copy, all
  `adds_leftover = f`. The copy still has the five links that the production
  repair removed.
- Compared against the `HEAD` code on the copy: 44,297 composites (all 41,291
  with a superseded attempt, 3,000 random others and the 20 busiest), with
  0 mismatches. SELECTs per recalculation went from a median of 4 (p99 17,
  max 93) to always 2. Over all 44,297, the time fell from 145 s to 97 s
  (library backend).
- `tests/test_t_level_recalculation.py` covers class sums, leftovers
  (including param 0 and merging with a pending stage 1), the orphan rule and
  the fixed statement count. The orphan and statement-count tests fail on the
  old code.
- Follow-up (2026-11-24 review). `get_current_t_level_from_attempts` had two
  more callers: `POST /admin/composites/calculate-t-levels` and the
  background recalculate-all in `admin/maintenance.py`. Both batch-fetched
  unsuperseded attempts and never counted leftovers, so they lowered the 5,013
  composites with a partial stage 2. All paths now call
  `TLevelCalculator.calculate_composite_t_level()`, and the batch fetch helper
  is gone. That costs 2 queries per composite instead of one bulk fetch, which
  matters only for recalculate-all. `tests/test_admin_t_level_recalculation.py`
  fails on the old code.

### 7. Worker recycling

**Problem.** `docker/Dockerfile` starts uvicorn 0.24 with
`--limit-max-requests 5000`. With a single worker and no process manager, the
whole process exits when it reaches the limit:

1. uvicorn stops accepting connections, finishes the requests in flight, and
   exits.
2. `restart: unless-stopped` in `docker-compose.prod.yml` restarts the
   container, which reruns `alembic upgrade head` and re-imports the app.
3. In between, nginx returns 502 for every request. Work requests fail, and
   submissions fall back to the client's retry queue.
4. Every in-process cache is emptied, including item 1's and item 4b's.

Every HTTP request counts toward the limit. The production copy's last 30 days
give a floor of about 2,800 requests a day: about 430 work assignments (around
2 requests each), about 770 submissions, and about 300 residues (around 4
requests each). That is before "no work" polls, page views and crawlers, so
the server probably restarts every day or two.

**Change.** Remove the flag, then watch the container's RSS with
`docker stats` for a week or two. If memory does grow, there are two options.
A much higher limit, such as 100,000, would mean one short outage a month. Or
the app can run under gunicorn with one uvicorn worker: gunicorn keeps the
listening socket open, so connections wait during a worker restart instead of
failing. That costs 20–30 MB for the gunicorn process.

**Verified on the droplet, 2026-11-23** (the container is `server-api-1`).
uvicorn 0.24 logs nothing when it reaches the limit (`Server.on_tick` just
returns "should exit"), so count the starts instead:
`docker logs -t server-api-1 2>&1 | grep "Started server process"`.
- From Nov 9 to Nov 23 there were 20 starts, mostly 16–30 hours apart. Each
  had the same in-container PID (8, sometimes 7), so each was a fresh
  container start.
- 5,000 requests every ~20 hours is about 6,000 requests a day, twice the
  floor estimated above.
- Don't rely on `docker inspect`'s RestartCount. It said 3, because Docker
  resets it whenever the daemon or the container is started fresh (Postgres
  and `ratings` had both been up 3 days).

### 8. Duplicate indexes

Every insert and every non-HOT update maintains all of a table's indexes;
`composites` has 13 and `ecm_attempts` 11.

Drop unconditionally (exact duplicates):

- `ix_ecm_attempts_work_hash` (35 MB): duplicates the unique constraint
  `ecm_attempts_work_hash_key`. It comes from `index=True, unique=True` on the
  same column in `app/models/attempts.py`.
- `ix_ecm_residues_id`: duplicates the primary key.

Drop only if production reports no scans for them (query in section 5):
`ix_composites_is_active`, `ix_composites_priority`,
`ix_composites_target_t_level`, `ix_composites_prior_t_level`,
`ix_ecm_attempts_composite_id`, `ix_ecm_attempts_client_id`,
`ix_ecm_attempts_client_ip`, `ix_ecm_attempts_factor_found`,
`ix_work_assignments_composite_id`, `ix_work_assignments_client_id`,
`ix_work_assignments_priority`. Most are a leading-column prefix of another
index on the same table.

Collect the usage numbers only after items 2–4 are in production. Run
`SELECT pg_stat_reset();` right after that deploy and wait at least a week. The
new indexes take over scans from some of these, such as
`ix_composites_target_t_level`, so counts gathered earlier describe plans that
no longer run.

When mirroring the drops in the models, also remove `index=True` from the
primary keys in `attempts.py`, `composites.py`, `factors.py`, `residues.py`
and `projects.py`. Each declares an `ix_<table>_id` index that duplicates its
primary key. Production has only `ix_ecm_residues_id`, which this item drops.
Alembic autogenerate would otherwise keep proposing the others.

### 9. PostgreSQL settings

The dump does not carry server settings. `docker/postgresql-lowmem.conf` is a
template, and production PostgreSQL runs natively, so it may not be applied.
Ask Kyle for the output of the settings query in section 5. `random_page_cost`
at the default of 4 on SSD storage biases the planner toward sequential scans;
1.1 is the usual value. Several baseline plans spilled sorts and hash
aggregates to disk at `work_mem = 4MB`; items 2–4 remove those queries, so
leave `work_mem` alone.

**Checked 2026-11-23, no change made.** Production reports `shared_buffers`
128MB, `work_mem` 4MB, `random_page_cost` 4, `effective_cache_size` 4GB and
`max_connections` 100. All are defaults. On the copy, every SELECT from the 13
dashboard pages (71 distinct statements) was planned under `random_page_cost`
4 and 1.1, with the other settings as in production. Of the 14 plans that
change, 8 get faster, 1 stays the same and 5 get slower. The 5 include the 365-day
leaderboard (90 → 119 ms) and the admin dashboard counts (14 → 28 ms). The
changed plans total 244 ms at 4 and 278 ms at 1.1. That is no gain, so the
setting stays. The server is shared with aliquot-tracker and opn-tracker; any
later change should be `ALTER DATABASE ecm_distributed SET ...`, not
server-wide.

### 10. Group-order calculation on factor submissions

**Problem.** `FactorService.add_factor` (`app/services/factors.py`, called
only from `submit.py`) runs PARI/GP for every new factor. It does this inside
the submit transaction, while the composite row lock is held.
- An ECM factor spawns `gp` twice: `calculate_group_order` computes the order
  (30 s timeout), then `_factor_group_order` factors it (10 s timeout).
- A P-1/P+1 factor goes through `calculate_p1_order` instead.

**Evidence.** The production copy has 3,370 factors in the last 30 days, about
112 a day (median 20 digits, maximum 65). 3,365 of them have a group order.
On the dev box, five recent production factors took:

| Factor size | Time |
|---|---|
| 19–20 digits | 37–71 ms |
| 45–48 digits | 165–175 ms |
| 56 digits | 334 ms |

The droplet will be several times slower. The appendix benchmark's `submit`
group sends no factors, so this cost is missing from the baseline. Once item 1
lands, it is the largest cost of a submission that finds a factor.

**Change.**
1. Measure first: add a factor-bearing case to the benchmark's `submit`
   group, using a real factor and sigma from the copy.
2. Run the calculation after the submit transaction commits, for example from
   FastAPI `BackgroundTasks`. The step opens its own session and fills
   `factors.group_order` and `group_order_factorization` by factor id in a
   short transaction. This is the one exception to the ground rule about
   moving work out of the transaction. It is safe because the calculation
   reads only the factor, sigma and parametrization, and writes only those
   two columns of a row that already exists.
3. Optionally, have one `gp` process compute and factor the order, which
   halves process start-up.

Keep the integer-only input check in `group_order.py`: it is the injection
boundary for a GP script, and GP has `system()`. If the background step fails,
the factor keeps a NULL group order, as happens today when `gp` fails. Moving
the work out removes it from the locked transaction and from the response
time, but not from the CPU. On one vCPU, other requests still wait while `gp`
runs.

**Acceptance.** On the benchmark, a factor-bearing submission spends no time
in `gp` before it responds. Note that `TestClient` runs background tasks
before it returns, so time the response separately from the background step.
`group_order` is filled in shortly afterwards for ECM, P-1 and P+1 factors.
The existing tests pass.

**Implemented 2026-11-23.**
- `add_factor` no longer runs PARI/GP. Submit collects a `GroupOrderJob` per
  new factor (`FactorService.group_order_job()`, which resolves the
  parametrization as before). Just before returning, it adds
  `fill_group_orders(db.get_bind(), jobs)` as a FastAPI background task.
- That runs after the response is sent, and only for a successful response.
  The submit transaction has committed by then: it commits when its `with`
  block exits. Each factor's update is its own short transaction, matched on
  id, factor value and `group_order IS NULL`.
- Measured end to end with real `gp` (dev box, local PostgreSQL). The
  submission's own commit, which releases the row lock, moved:

  | Factor | Before | After |
  |---|---|---|
  | 20 digits | 44 ms | 15 ms |
  | 45 digits | 129 ms | 15 ms |
  | 56 digits | 304 ms | 15 ms |

  The background step takes the same time as before. On one vCPU it still
  competes for the CPU, but it no longer holds the lock or delays the
  response.
- Step 3 (one `gp` process for the order and its factorization) was not done.
  The two calls have separate timeouts, 30 s and 10 s. A single process that
  timed out while factoring would lose the order as well, which is kept
  today.
- `tests/test_group_order_after_commit.py` covers ECM and P-1, a failed
  calculation, a known factor and a rolled-back submission, plus two tests
  with real `gp` (skipped where it is not installed, as in CI). The
  "after the commit" and "rolled back" tests fail on the old code.
- The review's follow-ups (2026-11-24):
  - A lost background task now leaves a NULL group order that nothing
    retries. Before, the factor and its group order committed together. A
    resubmission cannot heal it: an identical retry takes the duplicate path,
    and a re-found factor is already divided out.
    `scripts/backfill_group_orders.py` recalculates NULLs (dry run first). On
    the copy, 287 factors had none, mostly from failures: 86 single-digit and
    22 two-digit factors, which always fail.
  - A factor's own sigma prefix now sets its curve type. Before, it was
    dropped and the submission's parametrization was used.
  - The pointless re-read of the attempt's parametrization is gone.

## 4. Decisions for Kyle

1. **Licence** of the t-level library (item 1). Decided 2026-11-17: not a
   blocker.
2. **`raw_output` retention.** Submissions carry 442 KB on average, and the
   table holds 10 GB of it uncompressed for six weeks of data. Item 5 removes
   the read cost. Storing less would shrink request bodies, the writes in the
   submit transaction, and backups; purging on a schedule would shrink the
   table. This changes what is kept for debugging, so it is not part of this
   plan. Recommendation: keep the first and last 32 KB. Most outputs are small
   (median 2.9 KB); only the few large ones would be truncated.
3. **Removing `--limit-max-requests`** (item 7), since it was added on purpose.
   Recommendation: remove it once the log check confirms the restarts.
4. **Dropping the conditional indexes** in item 8, after the usage query.
5. **Public page cache** (item 4b): in the app or in nginx, or skipped if the
   access logs show little crawler traffic.

## 5. Checks to run on production

```bash
# Index and table usage (decides item 8, sanity-checks items 2-4)
psql "$DATABASE_URL" -X \
  -c "select relname, seq_scan, seq_tup_read, idx_scan, n_tup_ins, n_tup_upd, n_tup_hot_upd, n_dead_tup, last_autovacuum from pg_stat_user_tables order by seq_tup_read desc" \
  -c "select relname, indexrelname, idx_scan, pg_size_pretty(pg_relation_size(indexrelid)) from pg_stat_user_indexes order by relname, idx_scan"

# Settings (item 9)
psql "$DATABASE_URL" -X -c "show shared_buffers" -c "show work_mem" -c "show random_page_cost" -c "show effective_cache_size" -c "show max_connections"
```

After deploying index migrations, run `ANALYZE` once on `composites`,
`ecm_attempts` and `work_assignments`.

## 6. Suggested sequencing

1. Items 2 and 3 in one migration: indexes only, no code paths change.
   Implemented 2026-11-23 as `f6204945eb74`, together with item 8's
   unconditional drops.
2. Item 1 (deployed 2026-11-23, with the stale t-levels recalculated). The
   restarts happen about every 20 hours, so item 7 can follow on its own.
3. Item 5, then item 6 (both implemented 2026-11-23).
4. Item 10 (implemented 2026-11-23).
5. Item 4.
6. Items 7–9 as Kyle decides. Item 8's unconditional drops can go with step 1.
   Its conditional drops wait for usage numbers collected after step 5.

## Appendix A: the production copy

The 2026-11-16 production dump is loaded in the local `ecm_distributed`
database under the schema `prod_copy` (858 MB). `public` is the normal dev
schema and was not modified. The copy contains real client ids and IP
addresses: keep it local, and do not commit dumps or query output from it.

Point the app, Alembic or `psql` at it by appending a search path to the URL
in `server/.env`:

```bash
export BASE_PGURL=$(grep '^DATABASE_URL=' .env | cut -d= -f2-)
export DATABASE_URL="$BASE_PGURL?options=-csearch_path%3Dprod_copy"
venv/bin/alembic upgrade head     # applies new migrations to the copy; it has its own alembic_version
```

Use `SET max_parallel_workers_per_gather = 0` when reading plans. The droplet
has one vCPU and parallel plans hide the real cost.

Remove it with `DROP SCHEMA prod_copy CASCADE`.

To rebuild it: the dump (`C:\Users\kyle\dump-ecm_distributed-202611161100.tar`)
is a plain-SQL `pg_dumpall` file despite its name. The dev role cannot create
databases, so it was loaded by a filter that keeps only the `ecm_distributed`
section, drops roles, grants and ownership, and rewrites `public.` to
`prod_copy.` everywhere except inside `COPY` data blocks. Creating a database
as the `postgres` superuser and loading with `psql` is simpler if that access
is available.

## Appendix B: benchmark harness

Save outside the repository and run from `server/` with
`BASE_PGURL` set as above: `BENCH_SCHEMA=prod_copy venv/bin/python bench.py pages work`.
It reports wall time, database time, statement count and the slowest
statements per endpoint, taking the best of three runs.

The harness writes to the schema it runs against. The `work` group creates
work assignments and one residue claim for clients named `bench-*`; delete the
assignments and reset the residue to `available` afterwards, or later runs see
those composites as busy. The `submit` group inserts attempts with
`client_id='bench'`; delete them and recalculate the affected composites.
The hot and mid composites are chosen by attempt count each time the harness
starts, so one run's inserts can change the "mid" pick on the next run. List
the composites the `bench` attempts actually touched before cleaning up, and
snapshot the rows first rather than relying on a recalculation.
The copy was left clean after the baseline runs.

```python
"""Time real endpoints against a schema of the local database."""
import os, sys, time, logging

base = os.environ["BASE_PGURL"]
schema = os.environ.get("BENCH_SCHEMA", "prod_copy")
os.environ["DATABASE_URL"] = (
    base + "?options=-csearch_path%3D" + schema + "%20-cmax_parallel_workers_per_gather%3D0"
)
sys.path.insert(0, os.getcwd())
logging.disable(logging.CRITICAL)

from sqlalchemy import event, text
from fastapi.testclient import TestClient
from app.main import app
from app.database import engine, SessionLocal
from app import dependencies as deps

app.dependency_overrides[deps.verify_admin_key] = lambda: True
app.dependency_overrides[deps.verify_admin_key_html] = lambda: True

stats = {"n": 0, "t": 0.0, "slow": []}

@event.listens_for(engine, "before_cursor_execute")
def _before(conn, cur, stmt, params, ctx, many):
    ctx._t0 = time.perf_counter()

@event.listens_for(engine, "after_cursor_execute")
def _after(conn, cur, stmt, params, ctx, many):
    dt = time.perf_counter() - ctx._t0
    stats["n"] += 1
    stats["t"] += dt
    stats["slow"].append((dt, " ".join(stmt.split())[:110]))

client = TestClient(app)

with SessionLocal() as s:
    assert s.execute(text("select current_schema()")).scalar() == schema
    hot_id, hot_n = s.execute(text(
        "select composite_id, count(*) from ecm_attempts group by 1 order by 2 desc limit 1")).one()
    mid_id, mid_n = s.execute(text(
        "select composite_id, count(*) from ecm_attempts group by 1 order by 2 desc offset 200 limit 1")).one()
    comp = lambda i: s.execute(text("select current_composite from composites where id=:i"), {"i": i}).scalar()
    hot_comp, mid_comp = comp(hot_id), comp(mid_id)

def run(label, method, url, show_slow=2, **kw):
    best = None
    for _ in range(3):
        stats.update(n=0, t=0.0, slow=[])
        t0 = time.perf_counter()
        r = client.request(method, url, **kw)
        wall = time.perf_counter() - t0
        snap = (wall, stats["n"], stats["t"], sorted(stats["slow"], reverse=True)[:show_slow], r.status_code, len(r.content))
        if best is None or wall < best[0]:
            best = snap
    wall, n, dbt, slow, code, size = best
    print(f"{label:<44} {code}  wall {wall*1000:8.1f} ms | db {dbt*1000:8.1f} ms in {n:4d} stmts"
          f" | python {max(wall-dbt,0)*1000:7.1f} ms | {size/1024:7.1f} KB")
    for dt, stmt in slow:
        if dt > 0.005:
            print(f"      {dt*1000:8.1f} ms  {stmt}")

only = sys.argv[1:] or None
want = lambda tag: only is None or tag in only
D = "/api/v1/dashboard"

if want("pages"):
    run("public dashboard /", "GET", f"{D}/")
    run("testing-status", "GET", f"{D}/testing-status")
    run("p1-testing-status", "GET", f"{D}/p1-testing-status")
    run("curves (grouped)", "GET", f"{D}/curves")
    run("curves (ungrouped)", "GET", f"{D}/curves?group_by_composite=false")
    run("factors", "GET", f"{D}/factors")
    run("leaderboard (30d)", "GET", f"{D}/leaderboard")
    run("leaderboard (365d)", "GET", f"{D}/leaderboard?days=365")
    run("residue-status", "GET", f"{D}/residue-status")
    run(f"composite details ({hot_n} attempts)", "GET", f"{D}/composites/{hot_id}/details")
    run(f"composite details ({mid_n} attempts)", "GET", f"{D}/composites/{mid_id}/details")
    run("admin dashboard", "GET", "/api/v1/admin/dashboard")
    run("admin residue-status", "GET", "/api/v1/admin/residue-status")

if want("work"):
    run("ecm-work standard", "GET", "/api/v1/ecm-work?client_id=bench-a")
    run("ecm-work progressive", "GET", "/api/v1/ecm-work?client_id=bench-b&work_type=progressive")
    run("ecm-work standard + project", "GET", "/api/v1/ecm-work?client_id=bench-c&project=OPN")
    run("p1-work", "GET", "/api/v1/p1-work?client_id=bench-d")
    run("residues/work", "GET", "/api/v1/residues/work?client_id=bench-e", headers={"X-Client-ID": "bench-e"})

if want("submit"):
    def body(composite, sigma):
        return {"composite": composite, "client_id": "bench", "method": "ecm", "program": "gmp-ecm",
                "program_version": "7.0.6",
                "parameters": {"b1": 11000000, "b2": 35133391030, "curves": 8, "parametrization": 1,
                               "sigma": f"1:{sigma}"},
                "results": {"curves_completed": 8, "execution_time": 100.0},
                "raw_output": "Step 1 took 12345ms\n" * (int(os.environ.get("RAW", "20000")) // 20)}
    n = int(time.time())
    for label, composite, count, offset in (("hot", hot_comp, hot_n, 0), ("mid", mid_comp, mid_n, 100)):
        for i in range(2):
            stats.update(n=0, t=0.0, slow=[])
            t0 = time.perf_counter()
            r = client.post("/api/v1/submit_result", json=body(composite, n + i + offset))
            wall = time.perf_counter() - t0
            print(f"submit_result ({label}: {count} attempts) #{i}  {r.status_code}  wall {wall*1000:8.1f} ms"
                  f" | db {stats['t']*1000:8.1f} ms in {stats['n']:4d} stmts | python {(wall-stats['t'])*1000:7.1f} ms")
```

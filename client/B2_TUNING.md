# Local B2 timing notes

`example_b2_dictionary.txt` targets similar GPU stage-1 and CPU stage-2 elapsed
times for equal curve counts on an RTX 5070 and Ryzen 7 9800X3D with **12 CPU
workers**, subject to the service's minimum B2 of 100 times B1. It covers all 13
standard B1 sizes, including the small bounds. At B1=2.9 billion, the example
now uses a provisional B2=15 trillion, reduced from the measured 20 trillion.
That completed run used eight workers because twelve exceeded the available
RAM; even eight used about 60 GiB resident on its 364-digit input. Timing and
memory at 15 trillion remain untested. The service's B2 minimum makes CPU stage 2
slower than GPU stage 1 at some small bounds. The measurements below use
GMP-ECM 7.0.7 and local logs through October 2, 2026; the reduction was selected
on October 3.

The dictionary contains only `B1 B2`; leaving out the optional `k` column lets
GMP-ECM choose its stage-2 block count. Block count affects time and memory; see
the [GMP-ECM README](https://github.com/sethtroisi/gmp-ecm/blob/main/README).
These bounds are local starting points, not universal ECM optima. Lower B2
reduces work per curve as well as elapsed time.

## Completed service work

The September 29 run at **B1=110 million, B2=88 billion** completed all 3,072
curves on a 280-digit input with 12 workers. Its stage-2 command was observed
with B2=88 billion and no `-k`. Residue 79395 links this run to the exact stage-1
batch from September 28, including sigma range 3245340228–3245343299:

| Stage | Elapsed time | Source |
| --- | ---: | --- |
| GPU stage 1 | 4,362.854 s (72.7 min) | GMP-ECM's GPU time, September 28 at 07:33:47 |
| CPU stage 2 | 4,636.328 s (77.3 min) | First worker start to last worker finish, September 29, 13:04:42–14:21:59 |

Stage 2 took **6.3% longer**. Attempt 370708 was accepted at 14:22:05 with
`residue_completed=True`. This supports B2=88 billion under the observed
background load. Both stages completed without finding a factor.

The completed September 29 run at **B1=43 million, B2=10 billion** provides
another comparison on the same curves. Residue 79512 contained 3,072 curves on a
276-digit input. Its original GPU stage 1 (attempt 370517, sigma range
4063437160–4063440231) took **1,703.548 s (28.4 min)**. With 12 CPU workers,
stage 2 took **1,568.289 s (26.1 min)** from the first worker start at 16:16:39
to the last worker finish at 16:42:47, about **7.9% less time**. All curves
completed despite a finish-after-current request. Attempt 370724 was accepted
with `residue_completed=True` and t-level 49.795. The submitted B2 was verified
against the service record. This supports keeping the example's B2=10 billion.

A separate test at **B1=3 million, B2=300 million** completed and submitted all
2,560 curves of service residue 79430 (201-digit input). Stage 2 took **86.4 s**
with 12 workers. Four historical GPU batches at this B1 on 175–229-digit inputs
have a median of 95.92 s per 3,072 curves, or **79.9 s** scaled to 2,560 curves.
That comparison is an estimate from similar inputs, not the exact stage-1 batch.
Submission attempt 370679 was accepted with `residue_completed=True`.

The command used from `client/` was:

```bash
python3 ecm_client.py --stage2-only --min-b1 3e6 --max-b1 3e6 \
  --b2 3e8 --workers 12 --work-count 1 --exit-on-no-work --progress-interval 50
```

Use `--min-b1` and `--max-b1` to select service residue bounds; `--b1` does not
filter these assignments. This 3-million test preceded the later multi-day
2.9-billion trial described below.

## Small-bound CPU measurements

The September 30 runs between 11:43 and 12:08 provide **11 complete comparisons
on the same inputs and curves**. Each batch contains 3,072 curves and uses 12
CPU workers. GPU time comes from GMP-ECM's reported GPU elapsed time; CPU time
runs from the first worker start to the last worker finish. The uploaded
residue IDs connect the two stages, and the saved worker transcripts confirm
the input, B1, B2, curve counts, and starting sigmas within each GPU batch.

| B1 | B2 | Full pairs | Input digits | GPU seconds | CPU seconds | CPU / GPU, per pair |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 250,000 | 25,000,000 | 4 | 137–192 | 6.13–8.52 | 17.64–24.47 | 2.67–3.01 |
| 1,000,000 | 100,000,000 | 3 | 137–168 | 24.82–31.74 | 41.14–47.98 | 1.51–1.67 |
| 3,000,000 | 300,000,000 | 4 | 137–187 | 73.91–95.98 | 74.33–95.12 | 0.91–1.07 |

The 250,000 pairs are residues 79592, 79593, 79595, and 79603; the 1-million
pairs are 79596–79598; the 3-million pairs are 79599–79602. Every corresponding
stage-2 submission confirmed `residue_completed=True`. Residue 79594 at
B1=1 million also completed successfully, but found a factor and stopped early;
it is excluded from this table. A temporary stage-1 submission rejection for
residue 79595 recovered through the queue at 11:45:59, before its stage 2 ran.

These direct comparisons confirm that 3 million is well balanced at the
service minimum. CPU stage 2 remains slower at 250,000 and 1 million, but
lowering their B2 values would prevent no-factor service completion. Keep all
three entries unchanged. These are observed local timings; other P+1/P-1 work
interleaves with some of the CPU runs. Local verification metadata is saved in
`data/benchmarks/b2_20260930/recent_small_pairs.json`.

Four September 30 service runs verified **B2=25 million at B1=250,000** and
**B2=100 million at B1=1 million**. All four submissions were accepted with
`residue_completed=True`. Three finished all 5,376 curves with 12 workers:

| B1 | B2 | Residue / attempt | CPU seconds / 5,376 curves | CPU seconds / 3,072 curves (scaled) |
| ---: | ---: | --- | ---: | ---: |
| 250,000 | 25,000,000 | 79560 / 370801 | 35.318 | 20.2 |
| 1,000,000 | 100,000,000 | 79556 / 370804 | 93.774 | 53.6 |
| 1,000,000 | 100,000,000 | 79555 / 370805 | 71.480 | 40.8 |

The two 1-million runs used 187- and 143-digit inputs respectively, so these
are separate workloads rather than repeats on the same input. Times run from
the first worker start to the last worker finish and exclude submission time.
The second 250,000 run, residue 79586 / attempt 370802, found a **37-digit
prime factor** and stopped after 11.4 seconds, with 1,452 curves credited.
It is excluded from full-batch timing comparisons. An unrelated P+1 factor
message interleaved with residue 79556 does not belong to that ECM run.

The full runs support keeping both example entries at the service minimum.
Normalized to 3,072 curves, CPU time is about 2.5 times the historical 8.0-second
GPU median at B1=250,000, and 1.3–1.7 times the 31.6-second median at B1=1 million.
Those GPU samples use different inputs and curves; they are approximate
comparisons. Lowering B2 would fall below the service completion threshold.
Public submission records confirm the bounds and completed curve counts;
local verification metadata is under
`data/benchmarks/b2_20260930/completed_user_work_*.json`.

The earlier probes below provide a consistent 250-digit CPU reference.
For B1=11,000 through 1 million, CPU stage 1 generated real RSA-250 residues.
Stage 2 resumed 384 curves per candidate B2, split across 12 workers with 32
curves each. Each candidate was measured twice in reversed order, without
overlapping another probe group. Times below are median wall times scaled to
3,072 curves, including process startup. The 3-million probes used 384 curves
from service residue 79430 instead. Every probe completed all expected curves.

| B1 | Example B2 | CPU seconds / 3,072 curves | Historical GPU seconds / 3,072 curves | GPU sample |
| ---: | ---: | ---: | ---: | --- |
| 11,000 | 1,100,000 | 5.0 | 0.67 | One 85-digit input; weak comparison |
| 50,000 | 5,000,000 | 11.4 | — | No uncontended local measurement |
| 250,000 | 25,000,000 | 35.3 | 8.0 | 39 batches, 137–216 digits |
| 1,000,000 | 100,000,000 | 84.7 | 31.6 | Three batches, 153–189 digits |
| 3,000,000 | 300,000,000 | 105.2 | 95.9 | Four batches, 175–229 digits |

The first four CPU rows use 250-digit RSA-250, so their GPU comparisons are
approximate. The smallest two entries remain provisional. The 3-million CPU
projection agrees reasonably with the completed service batch: 86.4 seconds for
2,560 curves corresponds to 103.7 seconds for 3,072 curves.

For local work, lower B2 choices can bring CPU time closer to GPU time:

| B1 | Local-only B2 | CPU seconds / 3,072 curves |
| ---: | ---: | ---: |
| 11,000 | 500,000 | 3.0 |
| 50,000 | 1,000,000 | 4.4 |
| 250,000 | 5,000,000 | 10.4 |
| 1,000,000 | 40,000,000 | 39.8 |

These lower choices deliberately allow a little extra CPU time at small bounds,
but they cannot complete service residues without a factor. The example uses
the service minimum instead so it can also be used with `--stage2-only`.

A later GPU sweep on September 29 overlapped another GPU application. **All
timings from that sweep are excluded**, including its 50,000 measurements.
The CPU probes in the two tables immediately above were taken earlier, before
the later 110-million CPU job started. Local probe results and saved residues are under the ignored
`data/benchmarks/b2_20260929/` directory; they are not part of the example file.

## Larger bounds

The September 22–24 pipeline logs provide additional completed measurements.
Every batch below used 3,072 curves and 12 CPU workers; times are median minutes.
Stage 1 uses reported GPU elapsed time, not CPU milliseconds from the GPU run.

| Input digits | B1 | B2 | Batches | Stage 1 | Stage 2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 110,000,000 | 55,000,000,000 | 1 | 72.3 | 53.6 |
| 250 | 110,000,000 | 77,000,000,000 | 4 | 73.7 | 66.3 |
| 385 | 110,000,000 | 88,000,000,000 | 8 | 117.7 | 115.6 |
| 451 | 110,000,000 | 77,000,000,000 | 8 | 117.7 | 160.3 |
| 250 | 260,000,000 | 182,000,000,000 | 1 | 171.7 | 100.3 |

The 451-digit results show that one B1-to-B2 table cannot balance every input
size. Older worker logs often omit B2; those timings were not assumed to use
the original dictionary values.

The remaining middle entries use short stage-2 cost estimates on a 250-digit
input, with 12 concurrent workers and automatic `k`:

| B1 | Measured B2 | Curves per worker | Projected CPU seconds / 3,072 curves | Historical GPU seconds / 3,072 curves |
| ---: | ---: | ---: | ---: | ---: |
| 11,000,000 | 1,000,000,000 | 3 | 423.4 | 436.5 |
| 43,000,000 | 10,000,000,000 | 2 | 1,782.7 | 1,705.0 |
| 260,000,000 | 260,000,000,000 | 1 | 11,004.4 | 10,347.3 |

These estimates ran `ecm -param 3 -sigma SIGMA -c C B1-B1 B2` on fresh points,
skipping stage 1. They measure stage-2 cost but do not constitute completed ECM
work; nothing from them was submitted. Projections use the slowest worker time
multiplied by `3072 / (12 * C)`. GPU samples use 227–272, 225–272, and 229–275-digit
inputs respectively (32, 72, and 123 batches). The 43-million entry now has a
completed residue comparison above, and the 260-million entry has the completed
run below.

The September 30 run at **B1=260 million, B2=260 billion** completed all 2,304
curves of residue 79574 on a 305-digit input, with 12 CPU workers. Its B2 was
verified against the submission record. From the first worker start at
08:14:56.888 to the last worker finish at 10:38:01.790, stage 2 took
**8,584.902 seconds (2h 23m 05s)**. Attempt 370796 was accepted with
`residue_completed=True` and t-level 56.622; no factor was found.

Two historical GPU runs at B1=260 million used this exact input, although their
curves differ from residue 79574. Both completed 3,072 curves: July 30 took
10,326.245 seconds, and August 26–27 took 10,924.260 seconds. An interrupted
August 26 start is excluded. Normalizing the CPU run to the same curve count
gives **11,446.536 seconds (3h 10m 47s)**, versus GPU times of
**2h 52m 06s–3h 02m 04s**, about **7.7% slower than their mean**. As a broader
check, 39 GPU batches on 290–310-digit inputs have a median of 10,379.659 seconds
per 3,072 curves, making this CPU run 10.3% slower than that reference.
The log shows a single-worker P+1/P-1 workload overlapping stage 2. This is an
observed local throughput comparison, not an isolated benchmark or a comparison
on the same curves. It supports keeping B2=260 billion. Metadata is saved under
`data/benchmarks/b2_20260930/completed_user_work_79574.json` locally.

The example now uses **B2=1.1 billion at B1=11 million**, the service minimum.
The 423.4-second estimate above was measured at B2=1 billion, so it is not a
measurement of the revised entry. Three September 29 service runs using the
old 1-billion bound submitted their results successfully, but residue completion
was rejected because B2 was below 100 times B1. Retrying those completion
requests cannot change the bounds already recorded in those attempts.

All three residues were subsequently rerun at B2=1.1 billion with 12 workers
and automatic `k`. Each completed all assigned curves without a factor; the
server accepted the new submission and confirmed the residue as completed.
These are reported stage-2 elapsed times under the observed background load:

| Residue | Input digits | Curves | Stage-2 seconds | Accepted attempt |
| ---: | ---: | ---: | ---: | ---: |
| 79526 | 146 | 5,376 | 353.2 | 370719 |
| 79361 | 216 | 2,560 | 267.4 | 370720 |
| 79362 | 218 | 1,792 | 183.2 | 370721 |

The original rejected attempts are superseded through the normal residue
completion path. The recovery journal, saved residues, raw ECM output, and
archived obsolete completion requests are in the ignored local directory
`data/recovery_b2_20260929/`.

Two completed runs on September 29–30 now support **B1=850 million,
B2=1,572,102,932,416**. Both used 12 CPU workers with 192 curves per worker;
both B2 values were verified against their submission records. The first run's
live commands also confirmed no `-k` override. All 2,304 curves per run finished
without a factor, and the server confirmed both residues as completed.

| Residue | Input digits | Stage-2 elapsed | CPU hours / 3,072 curves | Accepted attempt |
| ---: | ---: | ---: | ---: | ---: |
| 79411 | 303 | 6h 41m 30s | 8.92 | 370749 |
| 79251 | 297 | 6h 43m 53s | 8.98 | 370781 |

Elapsed times are measured from the first worker start to the last worker finish:
24,090.290 and 24,232.997 seconds. They differ by only **0.6%**. The first run
also confirms the earlier 6.67–6.83-hour projection from its 50-curve progress
messages. Logs show GPU work overlapping the first run and a single-worker
P+1 process overlapping part of the second; these are timings under the observed
local background load.

The mean normalized CPU time is **8.95 hours per 3,072 curves**. Thirteen
historical GPU batches at B1=850 million on 251–289-digit inputs have a median of
**9.82 hours** for 3,072 curves. CPU stage 2 is about **9% faster** in that
comparison. The input sizes differ and these are not matching stage-1 curves,
so this does not establish an exact GPU/CPU ratio for either input. It does
support retaining the current B2 as a locally measured starting point. Run
metadata and submission evidence are saved locally under the ignored directory
`data/benchmarks/b2_20260930/`.

### B1=2.9 billion: measured reference and provisional reduction

The **B1=2.9 billion, B2=20 trillion (2e13)** run completed all **2,304 curves**
of residue **76119** on a **364-digit input**, using **eight CPU workers** and
automatic `k`. From the first worker start on September 30 at 12:21:15.172 EDT
to the last finish on October 2 at 12:32:56.654 EDT, stage 2 took
**173,501.482 seconds (48h 11m 41s)**. Each worker completed 288 curves without
a factor. The server accepted attempt **371853** at 12:33:00 with
`residue_completed=True` and t-level **65.254**. Its reported duration was
173,501.5 seconds, matching the local elapsed time.

All eight saved worker transcripts confirm the input, bounds, and curve counts.
The public residue and submission records also confirm completion at B2=20
trillion. Stage 1 was attempt 356129; the residue headers identify its producer
as Kyle-PC running GMP-ECM 7.0.6. Its service record has no elapsed time.
The local August 27 stage-1 start on this same input was interrupted and cannot
supply a matching GPU timing.
Verification metadata is saved locally in
`data/benchmarks/b2_20261003/completed_user_work_76119.json`.

Scaling the completed CPU run to 3,072 curves gives **64.26 hours**. Two
historical local GPU batches, each with 3,072 curves, provide a reference:

| GPU batch start | Input digits | GPU stage-1 seconds | Hours |
| --- | ---: | ---: | ---: |
| June 24, 2026 | 344 | 147,728.608 | 41.04 |
| August 2, 2026 | 311 | 145,594.288 | 40.44 |

CPU stage 2 is **57.7% slower** than the mean of those GPU times after matching
curve counts. These are different inputs and runs under observed background
load, not a measurement on the same curves. B2=20 trillion is now a completed
service measurement, but **does not establish equal stage times with eight
workers**. Twelve workers could not be measured to completion because of
memory pressure.

On October 3, the example was reduced to **B2=15 trillion (15e12)** as a
provisional setting, retaining **eight workers and automatic `k`** as the
planned test configuration. Applying the earlier rough `time ∝ B2^p` scaling,
with `p=0.5–0.575`, to the completed 48.2-hour run estimates **41–42 hours for
2,304 curves** on the same input with eight workers. This is about 13–15% less
time, not a 25% runtime reduction. The historical GPU reference scaled to
2,304 curves is about **30.6 hours**, so the reduced bound may still leave CPU
stage 2 slower.

These are estimates, not measurements at 15 trillion. The GPU inputs differ,
and automatic block/polynomial choices can change time and memory abruptly.
Memory use at the reduced bound is also unmeasured; the 60 GiB observation
above applies to 20 trillion. There is no further residue available for a test
as of October 3. Verify elapsed time and memory on the next available residue;
the completed 20-trillion run remains the measured reference.

The original choice of B2=20 trillion, increased from 797,402,956,366, was an
extrapolation for twelve workers:

The completed 260-million and 850-million CPU runs above give 3.18 and 8.95
hours per 3,072 curves at B2=260 billion and 1.572 trillion, respectively, on
297–305-digit inputs. Fitting just these two points to `time ∝ B2^p` gives
`p ≈ 0.575`. Extending that empirical fit to the GPU target of 40.74 hours
suggests B2≈21.9 trillion. Rounding down to 20 trillion predicts **38.6 hours**.
For sensitivity, assuming a square-root relationship instead predicts
**31.9 hours** at the same B2. Neither relationship is a validated runtime law;
the fit crosses different B1 values and inputs, and automatic block/polynomial
choices can introduce jumps. Larger inputs or competing CPU work may also
take longer.

The original extrapolation was **roughly 32–40 hours per 3,072 curves with 12
workers**, not a guaranteed range. That scaled to about 24–30 hours for 2,304
curves or 56–70 hours for 5,376 curves. B2=20 trillion is about 25 times the old
entry; that does not imply 25 times the CPU runtime.

The initial 12-worker run of residue 76119 started at 12:13:41 on September 30
and exceeded the available RAM: WSL had 76.7 GiB total, about 20 GiB of system
swap in use, and only 1.7–2.9 GiB available during inspection. The user stopped
it at 12:20:03,
before any curve completed, and the client released the residue successfully.
It supplies memory evidence but no completed-curve timing.

The user restarted the same residue with **8 workers** at 12:21:15, retaining
B2=20 trillion and automatic `k`. At 12:27:49 the eight ECM processes used
59.8 GiB resident with zero process swap; 13.5 GiB remained available and recent
memory-pressure averages were zero. This is one snapshot, not a peak-memory
guarantee. The twelve-worker estimate does not describe the completed
eight-worker run, and worker-count scaling should not be assumed linear.
Memory snapshots are saved locally in
`data/benchmarks/b2_20260930/in_progress_user_work_76119.jsonl`.

The October 1 projection from each worker's latest 50-curve interval predicted
an October 2 finish near 11:50 EDT; the actual last finish was about 43 minutes
later. The original progress calculations remain saved locally in
`data/benchmarks/b2_20261001/in_progress_user_work_76119.json` for comparison.

When another residue is available, test the revised **B2=15 trillion** entry
on hardware with sufficient RAM using eight workers from `client/`:

```bash
python3 ecm_client.py --stage2-only --min-b1 2.9e9 --max-b1 2.9e9 \
  --b2-dictionary example_b2_dictionary.txt --workers 8 --work-count 1 \
  --exit-on-no-work --progress-interval 10
```

This uses automatic `k` and submits the completed work normally. Progress every
10 curves per worker should provide an early throughput estimate, before the
full assignment finishes.

The B1=7.6 billion and 25 billion entries retain the original B2 values with
`k` removed. They remain **uncalibrated**, including their smaller B2 values
relative to the provisional 2.9-billion entry. Their uneven progression is not a
measured throughput optimum.

## Using and refining the example

From `client/`, pass `--b2-dictionary example_b2_dictionary.txt --workers 12`
to a local two-stage or service stage-2-only run at the measured smaller bounds.
The completed 2.9-billion run used eight workers because of memory pressure,
as described above. The dictionary does not set the worker count. The separate local
`b2_dictionary.txt` is not changed by editing the example.

For service `--stage2-only` work, completing a residue without finding a factor
requires at least 75% of its curves and, for an explicit B2, **B2 >= 100 * B1**.
All entries in the example meet this threshold; B1 up to 11 million uses exactly
100 times B1. This is a service completion requirement, not an optimal timing
ratio. To use a different explicit `--b2`, omit the dictionary, since a matching
dictionary entry overrides it.

RSA-250 is a useful fixed input: its published factors are both 125-digit
primes, making a factor discovery in these short ECM tests negligibly unlikely.
See [Appendix B of the factorization paper](https://eprint.iacr.org/2020/697.pdf).
Separate RSA-250 cost probes, repeated three times per worker count, measured
**7.35–7.64x throughput with 12 workers versus one**, at B1/B2 pairs 3e6/1e8,
11e6/1e9, and 43e6/1e10. The 9800X3D has eight physical cores and 16 hardware
threads; assuming 12x throughput would underestimate CPU elapsed time here.

For further calibration, time GPU stage 1 without competing GPU work, save its
residues, then compare candidate B2 values with 12 concurrent CPU workers.
Compare equal curve counts on the same input, and confirm projections with a
full batch. Exclude factor discoveries and interrupted or contended runs.
Re-measure each B2 change: GMP-ECM's block and polynomial choices make runtime
change in steps. Input size and competing CPU load also affect the balance.

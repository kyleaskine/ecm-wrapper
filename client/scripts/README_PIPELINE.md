# Pipeline Batch Processing

## Overview

`run_batch_pipeline.py` implements concurrent GPU Stage 1 and CPU Stage 2 processing in a pipeline architecture. This maximizes hardware utilization by keeping both GPU and CPU busy simultaneously.

## Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐
│  GPU Thread │────▶│  CPU Thread │
│  Stage 1    │     │  Stage 2    │
└─────────────┘     └─────────────┘

Number 1: GPU S1 ────────────────▶ CPU S2
Number 2:         GPU S1 ─────────────────▶ CPU S2
Number 3:                 GPU S1 ──────────────────▶ CPU S2
```

While the GPU processes Stage 1 for number N, the CPU processes Stage 2 for number N-1. The threads synchronize via a bounded queue (maxsize=1) to prevent the GPU from getting too far ahead.

## Usage

### Basic Usage

```bash
# Process numbers with default settings (GPU stage 1, 1 curve = 3072 GPU curves, 4 CPU workers for stage 2)
python3 scripts/run_batch_pipeline.py --numbers-file data/numbers.txt --b1 160000000 --b2 17088933822400000

# Run 5 curves (5 × 3072 = 15,360 GPU curves per number)
python3 scripts/run_batch_pipeline.py --numbers-file data/numbers.txt --curves 5 --b1 160000000

# CPU-only mode (no GPU)
python3 scripts/run_batch_pipeline.py --numbers-file data/numbers.txt --no-gpu --b1 160000000

# Stage 1 only (skip stage 2)
python3 scripts/run_batch_pipeline.py --numbers-file data/numbers.txt --b1 160000000 --b2 0

# Custom GPU settings
python3 scripts/run_batch_pipeline.py --gpu-device 0 --gpu-curves 256 --b1 160000000

# More CPU workers for stage 2
python3 scripts/run_batch_pipeline.py --stage2-workers 8 --b1 160000000
```

### Options

- `--numbers-file FILE`: File containing numbers to factor (default: `data/numbers.txt`)
- `--b1 N`: B1 bound for stage 1 (default: 160000000)
- `--b2 N`: B2 bound for stage 2 (default: from config, use 0 to skip stage 2)
- `--curves N`: Number of curves per number (default: 1, which is 3072 curves on GPU)
- `--stage2-workers N`: Number of CPU workers for stage 2 (default: 4)
- `--gpu-device N`: GPU device ID to use
- `--gpu-curves N`: GPU curves per kernel launch
- `--no-gpu`: Use CPU for stage 1 instead of GPU
- `--project NAME`: Project name for result submission
- `--no-submit`: Do not submit results to server
- `--verbose, -v`: Verbose output
- `--continue-after-factor`: Continue stage 2 even if factor found
- `--progress-interval N`: Report progress every N curves in stage 2

## Performance Benefits

**Traditional Sequential Processing:**
- Time per number: Stage1_time + Stage2_time
- Total time: N × (Stage1_time + Stage2_time)

**Pipeline Processing:**
- Time per number: max(Stage1_time, Stage2_time)
- Total time: ≈ N × max(Stage1_time, Stage2_time)
- **Speedup: Up to 2x** if Stage 1 and Stage 2 take similar time

## Example

Process a batch of 100 numbers with GPU stage 1 and 4-worker CPU stage 2:

```bash
python3 scripts/run_batch_pipeline.py \
  --numbers-file data/numbers.txt \
  --b1 160000000 \
  --b2 17088933822400000 \
  --curves 100 \
  --stage2-workers 4 \
  --project MyProject \
  --verbose
```

## Monitoring

The script outputs progress logs from both threads:

```
[GPU Thread] [1/100] Starting stage 1 for 12345678...
[GPU Thread] [1/100] Stage 1 complete in 45.2s, passing to CPU thread
[CPU Thread] [1/100] Starting stage 2 for 12345678...
[CPU Thread] [1/100] Stage 2 complete in 50.1s, no factor
Pipeline Stats: 1/100 complete, 0 factors found, 1.5 min elapsed
```

## Technical Details

- **Thread synchronization**: `queue.Queue(maxsize=1)` provides bounded buffer
- **Residue files**: Temporary files in `/tmp/` cleaned up after stage 2
- **Factor handling**: Supports factors found in either stage 1 or stage 2
- **Early termination**: Stage 2 workers can terminate early when factor found (unless `--continue-after-factor`)
- **Result submission**: Automatic submission after each number completes (unless `--no-submit`)

## Comparison with run_batch.sh

**run_batch.sh** (Sequential):
```bash
for number in numbers; do
  run_stage1_and_stage2_sequentially(number)
done
```

**run_batch_pipeline.py** (Concurrent):
```python
GPU_thread:  run_stage1(N) → pass_to_queue
CPU_thread:  wait_for_queue → run_stage2(N-1)
```

The pipeline version keeps both GPU and CPU busy, approximately doubling throughput when stage times are balanced.

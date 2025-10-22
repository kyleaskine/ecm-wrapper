#!/usr/bin/env python3
"""
Pipeline batch processing for ECM factorization.

Runs GPU stage 1 and CPU stage 2 concurrently in a pipeline architecture:
- GPU thread: Processes stage 1 for number N, saves residues
- CPU thread: Processes stage 2 for number N-1 from residue queue

This maximizes hardware utilization by keeping both GPU and CPU busy.
"""

import sys
import argparse
import threading
import queue
import time
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import wrapper using importlib to handle hyphenated module name
import importlib.util
spec = importlib.util.spec_from_file_location("ecm_wrapper",
    str(Path(__file__).parent.parent / "ecm-wrapper.py"))
ecm_wrapper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ecm_wrapper_module)
ECMWrapper = ecm_wrapper_module.ECMWrapper


class PipelineStats:
    """Track pipeline statistics"""
    def __init__(self):
        self.lock = threading.Lock()
        self.total_numbers = 0
        self.stage1_completed = 0
        self.stage2_completed = 0
        self.factors_found = 0
        self.start_time = time.time()

    def increment_stage1(self):
        with self.lock:
            self.stage1_completed += 1

    def increment_stage2(self):
        with self.lock:
            self.stage2_completed += 1

    def increment_factors(self):
        with self.lock:
            self.factors_found += 1

    def get_summary(self) -> str:
        with self.lock:
            elapsed = time.time() - self.start_time
            return (f"Pipeline Stats: {self.stage2_completed}/{self.total_numbers} complete, "
                   f"{self.factors_found} factors found, "
                   f"{elapsed/60:.1f} min elapsed")


def gpu_worker(wrapper: ECMWrapper, numbers: list, b1: int, curves: int,
               residue_queue: queue.Queue, stats: PipelineStats,
               use_gpu: bool, gpu_device: Optional[int], gpu_curves: Optional[int],
               verbose: bool, logger: logging.Logger):
    """
    GPU worker thread: Process stage 1 for all numbers.

    Saves residues to temporary files and passes them to CPU thread via queue.
    """
    logger.info(f"[GPU Thread] Starting stage 1 processing for {len(numbers)} numbers")

    for i, number in enumerate(numbers, 1):
        try:
            logger.info(f"[GPU Thread] [{i}/{len(numbers)}] Starting stage 1 for {number[:30]}...")

            # Create temporary residue file
            residue_file = Path(f"/tmp/pipeline_residue_{i}_{int(time.time())}.txt")

            # Run stage 1
            stage1_start = time.time()
            stage1_success, stage1_factor, actual_curves, stage1_output, all_factors = wrapper._run_stage1(
                composite=number,
                b1=b1,
                curves=curves,
                residue_file=residue_file,
                use_gpu=use_gpu,
                verbose=verbose,
                gpu_device=gpu_device,
                gpu_curves=gpu_curves
            )
            stage1_time = time.time() - stage1_start

            if stage1_factor:
                logger.info(f"[GPU Thread] [{i}/{len(numbers)}] Factor found in stage 1: {stage1_factor}")
                stats.increment_factors()
                # Still pass to CPU thread for result submission

            if not stage1_success:
                logger.error(f"[GPU Thread] [{i}/{len(numbers)}] Stage 1 failed for {number[:30]}...")
                continue

            if not residue_file.exists() or residue_file.stat().st_size == 0:
                logger.error(f"[GPU Thread] [{i}/{len(numbers)}] No valid residue file generated")
                continue

            stats.increment_stage1()
            logger.info(f"[GPU Thread] [{i}/{len(numbers)}] Stage 1 complete in {stage1_time:.1f}s, "
                       f"passing to CPU thread ({stats.get_summary()})")

            # Pass to CPU thread (blocks if queue is full - natural backpressure)
            residue_queue.put({
                'number': number,
                'residue_file': residue_file,
                'b1': b1,
                'curves': actual_curves,
                'stage1_factor': stage1_factor,
                'all_factors': all_factors,
                'stage1_time': stage1_time,
                'index': i,
                'total': len(numbers)
            })

        except Exception as e:
            logger.error(f"[GPU Thread] [{i}/{len(numbers)}] Error processing {number[:30]}...: {e}")

    # Send sentinel to signal CPU thread to stop
    logger.info("[GPU Thread] All stage 1 work complete, signaling CPU thread")
    residue_queue.put(None)


def cpu_worker(wrapper: ECMWrapper, b1: int, b2: int, stage2_workers: int,
               residue_queue: queue.Queue, stats: PipelineStats,
               verbose: bool, project: Optional[str], no_submit: bool,
               continue_after_factor: bool, progress_interval: int,
               logger: logging.Logger):
    """
    CPU worker thread: Process stage 2 from residue queue.

    Processes residues from GPU thread and submits results.
    """
    logger.info(f"[CPU Thread] Starting stage 2 processing with {stage2_workers} workers")

    while True:
        # Get next residue from queue
        work_item = residue_queue.get()

        # Check for sentinel (end signal)
        if work_item is None:
            logger.info("[CPU Thread] Received stop signal, shutting down")
            residue_queue.task_done()
            break

        try:
            number = work_item['number']
            residue_file = work_item['residue_file']
            b1_actual = work_item['b1']
            curves = work_item['curves']
            stage1_factor = work_item['stage1_factor']
            all_factors = work_item['all_factors']
            stage1_time = work_item['stage1_time']
            idx = work_item['index']
            total = work_item['total']

            logger.info(f"[CPU Thread] [{idx}/{total}] Starting stage 2 for {number[:30]}...")

            # Skip stage 2 if factor found in stage 1 and B2 is 0
            stage2_time = 0.0
            if stage1_factor and b2 == 0:
                logger.info(f"[CPU Thread] [{idx}/{total}] Skipping stage 2 (B2=0, factor found in stage 1)")
                stage2_factor = None
                stage2_sigma = None
            elif b2 > 0:
                # Run stage 2
                stage2_start = time.time()
                early_termination = wrapper.config['programs']['gmp_ecm'].get('early_termination', True) and not continue_after_factor
                stage2_result = wrapper._run_stage2_multithread(
                    residue_file=residue_file,
                    b1=b1_actual,
                    b2=b2,
                    workers=stage2_workers,
                    verbose=verbose,
                    early_termination=early_termination,
                    progress_interval=progress_interval
                )
                stage2_time = time.time() - stage2_start

                # Extract factor and sigma
                # Note: stage2_result is (factor, sigma) tuple if factor found, None if no factor
                stage2_factor = None
                stage2_sigma = None
                if stage2_result:
                    stage2_factor, stage2_sigma = stage2_result
                    logger.info(f"[CPU Thread] [{idx}/{total}] Factor found in stage 2: {stage2_factor}")
                    stats.increment_factors()
                else:
                    logger.info(f"[CPU Thread] [{idx}/{total}] Stage 2 complete in {stage2_time:.1f}s, no factor")
            else:
                logger.info(f"[CPU Thread] [{idx}/{total}] Skipping stage 2 (B2=0)")
                stage2_factor = None
                stage2_sigma = None

            # Submit results (split failure is rare and already logged as error)
            if not no_submit:
                factor_found = stage1_factor or stage2_factor

                # Calculate total execution time (stage 1 + stage 2)
                total_time = stage1_time + stage2_time

                # Build results dict for submission
                results = {
                    'composite': number,
                    'b1': b1_actual,
                    'b2': b2,
                    'curves_requested': curves,
                    'curves_completed': curves,
                    'factor_found': factor_found,
                    'method': 'ecm',
                    'two_stage': True,
                    'stage2_workers': stage2_workers,
                    'execution_time': total_time,
                    'raw_output': ''
                }

                # Handle multiple factors from stage 1
                if all_factors:
                    results['factors_found'] = [f[0] for f in all_factors]

                wrapper.submit_result(results, project, 'gmp-ecm-ecm')
                logger.info(f"[CPU Thread] [{idx}/{total}] Submitted results (total time: {total_time:.1f}s)")

            stats.increment_stage2()
            logger.info(f"[CPU Thread] [{idx}/{total}] Complete ({stats.get_summary()})")

            # Cleanup residue file
            try:
                residue_file.unlink()
            except:
                pass

        except Exception as e:
            logger.error(f"[CPU Thread] Error processing work item: {e}")

        finally:
            residue_queue.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline batch ECM processing (GPU stage 1 + CPU stage 2 concurrently)"
    )
    parser.add_argument('--numbers-file', default='data/numbers.txt',
                       help='File containing numbers to factor (one per line)')
    parser.add_argument('--config', default='client.yaml',
                       help='Configuration file')
    parser.add_argument('--b1', type=int, default=160000000,
                       help='B1 bound for stage 1')
    parser.add_argument('--b2', type=int, default=None,
                       help='B2 bound for stage 2 (0 to skip stage 2)')
    parser.add_argument('--curves', type=int, default=1,
                       help='Number of curves per number (default: 1, which is 3072 curves on GPU)')
    parser.add_argument('--stage2-workers', type=int, default=4,
                       help='Number of CPU workers for stage 2')
    parser.add_argument('--gpu-device', type=int, default=None,
                       help='GPU device to use')
    parser.add_argument('--gpu-curves', type=int, default=None,
                       help='GPU curves per kernel launch')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Use CPU for stage 1 instead of GPU')
    parser.add_argument('--project', help='Project name for result submission')
    parser.add_argument('--no-submit', action='store_true',
                       help='Do not submit results to server')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--continue-after-factor', action='store_true',
                       help='Continue stage 2 even if factor found')
    parser.add_argument('--progress-interval', type=int, default=0,
                       help='Report progress every N curves in stage 2')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Load numbers
    numbers_file = Path(args.numbers_file)
    if not numbers_file.exists():
        logger.error(f"Numbers file not found: {numbers_file}")
        sys.exit(1)

    with open(numbers_file) as f:
        numbers = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(numbers)} numbers from {numbers_file}")

    # Initialize wrapper
    wrapper = ECMWrapper(args.config)

    # Resolve parameters
    b1 = args.b1
    b2 = args.b2 if args.b2 is not None else wrapper.config['programs']['gmp_ecm'].get('default_b2', 0)
    curves = args.curves  # Default is 1 (which is 3072 on GPU)
    use_gpu = not args.no_gpu

    logger.info(f"Pipeline configuration:")
    logger.info(f"  Stage 1: {'GPU' if use_gpu else 'CPU'} (B1={b1}, {curves} curves per number)")
    logger.info(f"  Stage 2: {args.stage2_workers} CPU workers (B2={b2})")
    logger.info(f"  Submit results: {not args.no_submit}")

    # Create stats tracker
    stats = PipelineStats()
    stats.total_numbers = len(numbers)

    # Create bounded queue (maxsize=1 keeps stages synchronized)
    residue_queue = queue.Queue(maxsize=1)

    # Start CPU thread first (it will block waiting for work)
    cpu_thread = threading.Thread(
        target=cpu_worker,
        args=(wrapper, b1, b2, args.stage2_workers, residue_queue, stats,
              args.verbose, args.project, args.no_submit,
              args.continue_after_factor, args.progress_interval, logger),
        name="CPU-Stage2"
    )
    cpu_thread.start()

    # Start GPU thread
    gpu_thread = threading.Thread(
        target=gpu_worker,
        args=(wrapper, numbers, b1, curves, residue_queue, stats,
              use_gpu, args.gpu_device, args.gpu_curves, args.verbose, logger),
        name="GPU-Stage1"
    )
    gpu_thread.start()

    # Wait for both threads to complete
    logger.info("Pipeline started, waiting for completion...")
    gpu_thread.join()
    cpu_thread.join()

    # Final stats
    elapsed = time.time() - stats.start_time
    logger.info("="*60)
    logger.info("Pipeline complete!")
    logger.info(f"  Numbers processed: {stats.stage2_completed}/{stats.total_numbers}")
    logger.info(f"  Factors found: {stats.factors_found}")
    logger.info(f"  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"  Avg time per number: {elapsed/max(stats.stage2_completed, 1):.1f} seconds")
    logger.info("="*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import subprocess
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from base_wrapper import BaseWrapper
from parsing_utils import parse_ecm_output_multiple, count_ecm_steps_completed, ECMPatterns
from residue_manager import ResidueFileManager
from result_processor import ResultProcessor
from stage2_executor import Stage2Executor
from ecm_worker_process import run_worker_ecm_process

class ECMWrapper(BaseWrapper):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.residue_manager = ResidueFileManager()

    def _log_and_store_factors(self, all_factors: List[Tuple[str, Optional[str]]],
                               results: Dict[str, Any], composite: str, b1: int,
                               b2: Optional[int], curves: int, method: str,
                               program: str) -> Optional[str]:
        """
        Deduplicate factors, log them, and store in results dictionary.

        This is now a thin wrapper around ResultProcessor for backward compatibility.

        Args:
            all_factors: List of (factor, sigma) tuples
            results: Results dictionary to update
            composite: Composite number being factored
            b1, b2, curves: ECM parameters
            method: Method name (ecm, pm1, pp1)
            program: Program name for logging

        Returns:
            First factor (for compatibility)
        """
        processor = ResultProcessor(self, composite, method, b1, b2, curves, program)
        return processor.log_and_store_factors(all_factors, results, quiet=False)

    def run_ecm(self, composite: str, b1: int, b2: Optional[int] = None,
                curves: int = 100, sigma: Optional[int] = None,
                param: Optional[int] = None,
                use_gpu: bool = False, gpu_device: Optional[int] = None,
                gpu_curves: Optional[int] = None, verbose: bool = False,
                method: str = "ecm", continue_after_factor: bool = False,
                quiet: bool = False) -> Dict[str, Any]:
        """Run GMP-ECM or P-1 and capture output"""
        ecm_path = self.config['programs']['gmp_ecm']['path']

        # Build command base (without -c parameter, which will be added per batch)
        cmd_base = [ecm_path]

        # Add method-specific parameters
        if method == "pm1":
            cmd_base.append('-pm1')
        elif method == "pp1":
            cmd_base.append('-pp1')
        # Default is ECM, no flag needed

        if use_gpu and method == "ecm":  # GPU only works with ECM
            cmd_base.append('-gpu')
            if gpu_device is not None:
                cmd_base.extend(['-gpudevice', str(gpu_device)])
            if gpu_curves is not None:
                cmd_base.extend(['-gpucurves', str(gpu_curves)])
        if verbose:
            cmd_base.append('-v')
        if param is not None and method == "ecm":  # Param only applies to ECM
            cmd_base.extend(['-param', str(param)])
        if sigma and method == "ecm":  # Sigma only applies to ECM
            cmd_base.extend(['-sigma', str(sigma)])

        # B1 and B2 will be added after -c parameter

        method_name = method.upper() if method != "ecm" else "ECM"
        self.logger.info(f"Running {method_name} on {len(composite)}-digit number with B1={b1}, curves={curves}")

        # Run ECM using optimized batch execution
        start_time = time.time()
        results = {
            'composite': composite,
            'b1': b1,
            'b2': b2,
            'curves_requested': curves,
            'curves_completed': 0,
            'factor_found': None,
            'raw_outputs': [],
            'method': method
        }

        # Use batch execution: run multiple curves in single subprocess call
        batch_size = min(curves, 50)  # Process in batches to reduce overhead
        curves_completed = 0

        while curves_completed < curves and not results['factor_found']:
            curves_this_batch = min(batch_size, curves - curves_completed)

            # Build proper command: ecm [options] -c curves B1 [B2]
            batch_cmd = cmd_base + ['-c', str(curves_this_batch), str(b1)]
            if b2 is not None:
                batch_cmd.append(str(b2))

            try:
                # Track progress in callback
                batch_curves_completed = 0

                def progress_callback(line, output_lines):
                    nonlocal batch_curves_completed
                    if "Step 1 took" in line:
                        batch_curves_completed += 1
                        total_completed = curves_completed + batch_curves_completed
                        if total_completed % 10 == 0:
                            self.logger.info(f"Completed {total_completed}/{curves} curves")

                # Stream subprocess output
                process, output_lines = self._stream_subprocess_output(
                    batch_cmd, composite, "ECM", progress_callback
                )

                stdout = '\n'.join(output_lines)
                results['raw_outputs'].append(stdout)

                # Parse output for factors using multiple factor parsing
                all_factors = parse_ecm_output_multiple(stdout)
                if all_factors:
                    # Use precise curve count from output parsing
                    actual_curves = count_ecm_steps_completed(stdout)
                    results['curves_completed'] = curves_completed + actual_curves

                    # Handle all factors found (skip logging if quiet mode)
                    if not quiet:
                        program_name = f"GMP-ECM ({method.upper()})" + (" with GPU" if use_gpu else "")
                        self._log_and_store_factors(all_factors, results, composite, b1, b2, curves, method, program_name)
                    else:
                        # Store factors without logging
                        if 'factors_found' not in results:
                            results['factors_found'] = []
                        results['factors_found'].extend([f[0] for f in all_factors])
                        if not results.get('factor_found'):
                            results['factor_found'] = all_factors[0][0]

                    self.logger.info(f"Factors found after {results['curves_completed']} curves")

                    if not continue_after_factor:
                        break
                    else:
                        self.logger.info("Continuing to process remaining curves due to --continue-after-factor flag")

                # Update curves completed for this batch
                curves_completed += curves_this_batch
                results['curves_completed'] = curves_completed

            except subprocess.SubprocessError as e:
                self.logger.error(f"Subprocess error in batch starting at curve {curves_completed + 1}: {e}")
                break
            except (OSError, IOError) as e:
                self.logger.error(f"I/O error in batch starting at curve {curves_completed + 1}: {e}")
                break
            except Exception as e:
                self.logger.exception(f"Unexpected error in batch starting at curve {curves_completed + 1}: {e}")
                break

        results['execution_time'] = time.time() - start_time
        results['raw_output'] = '\n'.join(results['raw_outputs'])

        # Final deduplication of factors found across all batches and full factorization
        if 'factors_found' in results and results['factors_found'] and not quiet:
            processor = ResultProcessor(self, composite, method, b1, b2, curves, f"GMP-ECM ({method.upper()})")
            processor.fully_factor_and_store(results['factors_found'], results, quiet=False)

        # Extract parametrization from raw output (look for "sigma=1:xxx" or "sigma=3:xxx")
        if 'parametrization' not in results:
            parametrization = 3  # Default to param 3
            raw_output = results.get('raw_output', '')
            # Look for sigma pattern in output
            sigma_match = ECMPatterns.SIGMA_COLON_FORMAT.search(raw_output)
            if sigma_match:
                sigma_str = sigma_match.group(1)
                if ':' in sigma_str:
                    parametrization = int(sigma_str.split(':')[0])
            results['parametrization'] = parametrization

        # Save raw output if configured
        if self.config['execution']['save_raw_output']:
            self.save_raw_output(results, f'gmp-ecm-{method}')

        return results

    def run_ecm_two_stage(self, composite: str, b1: int, b2: Optional[int] = None,
                         curves: int = 100, sigma: Optional[int] = None,
                         param: Optional[int] = None,
                         use_gpu: bool = True,
                         stage2_workers: int = 4, verbose: bool = False,
                         save_residues: Optional[str] = None,
                         resume_residues: Optional[str] = None,
                         gpu_device: Optional[int] = None,
                         gpu_curves: Optional[int] = None,
                         continue_after_factor: bool = False,
                         progress_interval: int = 0) -> Dict[str, Any]:
        """Run ECM using two-stage approach: GPU stage 1 + multi-threaded CPU stage 2"""

        # Note: B2 can be None to use GMP-ECM defaults

        stage1_desc = "GPU" if use_gpu else "CPU"
        self.logger.info(f"Running two-stage ECM: {stage1_desc} stage 1 ({curves} curves) + {stage2_workers} CPU workers for stage 2")

        start_time = time.time()
        results = {
            'composite': composite,
            'b1': b1,
            'b2': b2,
            'curves_requested': curves,
            'curves_completed': 0,
            'factor_found': None,
            'raw_outputs': [],
            'method': 'ecm',
            'two_stage': True,
            'stage2_workers': stage2_workers
        }

        # Handle residue file location
        if resume_residues:
            # Resume from existing residues - skip stage 1
            residue_file = Path(resume_residues)
            if not residue_file.exists():
                self.logger.error(f"Resume residue file not found: {residue_file}")
                results['execution_time'] = time.time() - start_time
                return results

            self.logger.info(f"Resuming from residue file: {residue_file}")
            stage1_factor = None  # No stage 1 run
            residue_info = self._parse_residue_file(residue_file)
            actual_curves = residue_info['curve_count']

        else:
            # Determine residue file location
            if save_residues:
                # Save to configured residue directory with specified filename
                residue_dir = Path(self.config['execution']['residue_dir'])
                residue_dir.mkdir(parents=True, exist_ok=True)
                residue_file = residue_dir / save_residues
                self.logger.info(f"Will save residues to: {residue_file}")
            else:
                # Use temporary directory for transient residue files
                import tempfile
                import hashlib
                composite_hash = hashlib.md5(composite.encode()).hexdigest()[:12]
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                residue_file = Path(tempfile.gettempdir()) / f"residue_{composite_hash}_{timestamp}.txt"
                self.logger.info(f"Using temporary residue file: {residue_file}")

            actual_curves = curves  # Initialize fallback
            try:
                # Stage 1: GPU or CPU execution with residue saving
                stage1_mode = "GPU" if use_gpu else "CPU"
                self.logger.info(f"Starting Stage 1 ({stage1_mode})")
                stage1_success, stage1_factor, actual_curves, stage1_output, all_stage1_factors = self._run_stage1(
                    composite, b1, curves, residue_file, use_gpu, verbose, gpu_device, gpu_curves, sigma, param
                )

                if stage1_factor:
                    # Stage 2 was never run, so set b2=0
                    results['b2'] = 0

                    # Log ALL unique factors found in Stage 1
                    if all_stage1_factors:
                        self._log_and_store_factors(all_stage1_factors, results, composite, b1, 0, curves, "ecm", "GMP-ECM (ECM)")
                    else:
                        # Fallback to single factor logging
                        self.log_factor_found(composite, stage1_factor, b1, 0, curves, method="ecm", sigma=None, program="GMP-ECM (ECM)")
                        results['factor_found'] = stage1_factor

                    results['curves_completed'] = actual_curves
                    results['execution_time'] = time.time() - start_time
                    results['raw_output'] = stage1_output
                    self.logger.info(f"Factor found in Stage 1: {stage1_factor}")
                    return results

                if not stage1_success:
                    self.logger.error("Stage 1 failed")
                    results['curves_completed'] = 0  # No valid results on failure
                    results['execution_time'] = time.time() - start_time
                    results['raw_output'] = stage1_output  # Include error output
                    return results

                if not residue_file.exists():
                    self.logger.error(f"No residue file generated at: {residue_file}")
                    results['curves_completed'] = 0  # No valid results
                    results['execution_time'] = time.time() - start_time
                    results['raw_output'] = stage1_output if 'stage1_output' in locals() else "No residue file generated"
                    return results

                # Check if residue file has content
                if residue_file.stat().st_size == 0:
                    self.logger.error(f"Residue file is empty: {residue_file}")
                    results['curves_completed'] = 0  # No valid results
                    results['execution_time'] = time.time() - start_time
                    results['raw_output'] = stage1_output if 'stage1_output' in locals() else "Empty residue file"
                    return results

                self.logger.info(f"Residue file created successfully: {residue_file} ({residue_file.stat().st_size} bytes)")

                if save_residues:
                    self.logger.info(f"Stage 1 residues saved permanently to: {residue_file}")
                else:
                    self.logger.debug(f"Stage 1 residues in temporary file (will be auto-deleted): {residue_file}")

            except subprocess.SubprocessError as e:
                self.logger.error(f"Stage 1 subprocess execution failed: {e}")
                results['curves_completed'] = 0
                results['execution_time'] = time.time() - start_time
                results['raw_output'] = f"Stage 1 subprocess failed: {e}"
                return results
            except (OSError, IOError) as e:
                self.logger.error(f"Stage 1 I/O error: {e}")
                results['curves_completed'] = 0
                results['execution_time'] = time.time() - start_time
                results['raw_output'] = f"Stage 1 I/O error: {e}"
                return results
            except Exception as e:
                self.logger.exception(f"Stage 1 unexpected error: {e}")
                results['curves_completed'] = 0
                results['execution_time'] = time.time() - start_time
                results['raw_output'] = f"Stage 1 unexpected error: {type(e).__name__}"
                return results

        # Stage 2: Multi-threaded CPU execution (skip if B2=0)
        stage2_factor = None
        stage2_sigma = None
        if b2 and b2 > 0:
            self.logger.info(f"Starting Stage 2 ({stage2_workers} workers) with B1={b1}, B2={b2}")
            early_termination = self.config['programs']['gmp_ecm'].get('early_termination', True) and not continue_after_factor
            if continue_after_factor:
                self.logger.info("Early termination disabled due to --continue-after-factor flag")
            stage2_result = self._run_stage2_multithread(
                residue_file, b1, b2, stage2_workers, verbose, early_termination, progress_interval
            )

            # Extract factor and sigma from stage 2 result
            if stage2_result:
                stage2_factor, stage2_sigma = stage2_result
        else:
            self.logger.info("Skipping Stage 2 (B2=0 - Stage 1 only mode)")

        factor_found = stage1_factor or stage2_factor
        if factor_found:
            # Only set factor_found if not already set by _log_and_store_factors
            if 'factor_found' not in results:
                results['factor_found'] = factor_found
            stage_found = "Stage 1" if stage1_factor else "Stage 2"
            sigma_used = stage2_sigma if stage2_factor else None
            self.logger.info(f"Factor found in {stage_found}: {factor_found}")
            self.log_factor_found(composite, factor_found, b1, b2, curves, method="ecm", sigma=sigma_used, program="GMP-ECM (ECM)")

        # Extract parametrization from stage 1 output
        parametrization = 3  # Default
        if stage1_output:
            sigma_match = ECMPatterns.SIGMA_COLON_FORMAT.search(stage1_output)
            if sigma_match:
                sigma_str = sigma_match.group(1)
                if ':' in sigma_str:
                    parametrization = int(sigma_str.split(':')[0])
        results['parametrization'] = parametrization

        results['curves_completed'] = actual_curves
        results['execution_time'] = time.time() - start_time

        return results

    def _run_stage1(self, composite: str, b1: int, curves: int,
                   residue_file: Path, use_gpu: bool, verbose: bool,
                   gpu_device: Optional[int] = None, gpu_curves: Optional[int] = None,
                   sigma: Optional[int] = None, param: Optional[int] = None) -> tuple[bool, Optional[str], int, str, List[tuple[str, Optional[str]]]]:
        """Run Stage 1 with GPU or CPU and save residues - returns (success, factor, actual_curves)"""
        ecm_path = self.config['programs']['gmp_ecm']['path']

        cmd = [ecm_path, '-save', str(residue_file)]
        if use_gpu:
            cmd.insert(1, '-gpu')
            if gpu_device is not None:
                cmd.extend(['-gpudevice', str(gpu_device)])
            if gpu_curves is not None:
                cmd.extend(['-gpucurves', str(gpu_curves)])
        if verbose:
            cmd.append('-v')
        if param is not None:
            cmd.extend(['-param', str(param)])
        if sigma is not None:
            cmd.extend(['-sigma', str(sigma)])
        cmd.extend(['-c', str(curves), str(b1), '0'])  # B2=0 for stage 1 only

        try:
            # Stream subprocess output
            process, output_lines = self._stream_subprocess_output(
                cmd, composite, "Stage1"
            )

            # Check for factors found in stage 1
            output = '\n'.join(output_lines)
            all_factors = parse_ecm_output_multiple(output)
            factor = all_factors[-1][0] if all_factors else None  # Use last factor for consistency

            # Extract actual curve count from GPU output
            actual_curves = curves  # fallback to requested
            curve_match = ECMPatterns.CURVE_COUNT.search(output)
            if curve_match:
                actual_curves = int(curve_match.group(1))
                self.logger.info(f"Stage 1 actually completed {actual_curves} curves")

            # Success if returncode is 0 (no factor) or if a factor was found (returncode 8)
            success = process.returncode == 0 or factor is not None
            return success, factor, actual_curves, output, all_factors

        except subprocess.SubprocessError as e:
            self.logger.error(f"Stage 1 subprocess error: {e}")
            return False, None, curves, "", []
        except (OSError, IOError) as e:
            self.logger.error(f"Stage 1 I/O error: {e}")
            return False, None, curves, "", []
        except Exception as e:
            self.logger.exception(f"Stage 1 unexpected error: {e}")
            return False, None, curves, "", []

    def _run_stage2_multithread(self, residue_file: Path, b1: int, b2: int,
                               workers: int, verbose: bool, early_termination: bool = True,
                               progress_interval: int = 0) -> Optional[Tuple[str, str]]:
        """
        Run Stage 2 with multiple CPU workers.

        This is now a thin wrapper around Stage2Executor.
        """
        executor = Stage2Executor(self, residue_file, b1, b2, workers, verbose)
        return executor.execute(early_termination, progress_interval)

    def _split_residue_file(self, residue_file: Path, num_chunks: int) -> List[Path]:
        """Split residue file into chunks for parallel processing"""
        # Create unique temporary directory for chunk files to avoid conflicts between concurrent jobs
        import tempfile
        chunk_dir = tempfile.mkdtemp(prefix="ecm_chunks_")
        self.logger.debug(f"Creating chunks in temporary directory: {chunk_dir}")

        # Use ResidueFileManager to split the file
        chunk_paths = self.residue_manager.split_into_chunks(
            str(residue_file), num_chunks, chunk_dir
        )

        # Convert string paths to Path objects
        return [Path(p) for p in chunk_paths]

    def run_ecm_multiprocess(self, composite: str, b1: int, b2: Optional[int] = None,
                            curves: int = 100, workers: int = 4, verbose: bool = False,
                            method: str = "ecm", continue_after_factor: bool = False) -> Dict[str, Any]:
        """Run ECM using multi-process approach: each worker runs full ECM cycles"""

        self.logger.info(f"Running multi-process ECM: {workers} workers, {curves} total curves")

        start_time = time.time()
        results = {
            'composite': composite,
            'b1': b1,
            'b2': b2,
            'curves_requested': curves,
            'curves_completed': 0,
            'factor_found': None,
            'raw_outputs': [],
            'method': method,
            'multiprocess': True,
            'workers': workers
        }

        # Distribute curves across workers
        curves_per_worker = curves // workers
        remaining_curves = curves % workers
        worker_assignments = []

        for worker_id in range(workers):
            worker_curves = curves_per_worker + (1 if worker_id < remaining_curves else 0)
            if worker_curves > 0:
                worker_assignments.append((worker_id + 1, worker_curves))

        self.logger.info(f"Curve distribution: {[f'Worker {w}: {c} curves' for w, c in worker_assignments]}")

        # Run workers in parallel
        factor_found = None
        total_curves_completed = 0

        # Use multiprocessing with a global function instead of ProcessPoolExecutor
        # to avoid pickling issues with instance methods
        import multiprocessing as mp

        # Create shared variables for early termination
        manager = mp.Manager()
        result_queue = manager.Queue()
        stop_event = manager.Event()

        # Start worker processes
        processes = []
        for worker_id, worker_curves in worker_assignments:
            p = mp.Process(
                target=run_worker_ecm_process,
                args=(worker_id, composite, b1, b2, worker_curves, verbose, method,
                      self.config['programs']['gmp_ecm']['path'], result_queue, stop_event)
            )
            p.start()
            processes.append(p)

        # Wait for results with improved synchronization
        factor_found = None
        factor_sigma = None
        all_sigma_values = []  # Collect all sigma values from all workers
        total_curves_completed = 0
        completed_workers = 0
        results_received = []

        # Use a shorter timeout and check processes more frequently
        while completed_workers < len(processes):
            got_result = False

            # Try to get results from queue with short timeout
            try:
                result = result_queue.get(timeout=0.5)
                results_received.append(result)
                total_curves_completed += result['curves_completed']

                # Collect sigma values from this worker
                if 'sigma_values' in result:
                    all_sigma_values.extend(result['sigma_values'])

                if result['factor_found'] and not factor_found:
                    factor_found = result['factor_found']
                    factor_sigma = result.get('sigma_found')
                    self.logger.info(f"Worker {result['worker_id']} found factor: {factor_found}")
                    if factor_sigma:
                        self.logger.info(f"Factor found with sigma: {factor_sigma}")
                    # Signal other workers to stop (unless continue_after_factor is enabled)
                    if not continue_after_factor:
                        stop_event.set()
                    else:
                        self.logger.info("Continuing all workers due to --continue-after-factor flag")

                got_result = True
            except:
                pass  # Timeout or empty queue

            # Check process status regardless of queue results
            active_processes = 0
            for i, p in enumerate(processes):
                if p.is_alive():
                    active_processes += 1
                elif p.exitcode is not None:
                    # Process finished - count it if we haven't already
                    if p.exitcode != 0:
                        self.logger.warning(f"Worker process {i+1} exited with code {p.exitcode}")

            completed_workers = len(processes) - active_processes

            # If no result was received and no processes are running, we might be done
            if not got_result and active_processes == 0:
                break

            # Brief sleep to prevent busy waiting
            if not got_result:
                time.sleep(0.1)

        # Final check for any remaining results in queue
        remaining_results = []
        try:
            while True:
                result = result_queue.get_nowait()
                remaining_results.append(result)
                total_curves_completed += result['curves_completed']

                # Collect sigma values from remaining results
                if 'sigma_values' in result:
                    all_sigma_values.extend(result['sigma_values'])

                if result['factor_found'] and not factor_found:
                    factor_found = result['factor_found']
                    factor_sigma = result.get('sigma_found')
                    self.logger.info(f"Worker {result['worker_id']} found factor: {factor_found}")
                    if factor_sigma:
                        self.logger.info(f"Factor found with sigma: {factor_sigma}")
        except:
            pass  # Queue is empty

        all_results = results_received + remaining_results
        self.logger.info(f"Collected results from {len(all_results)} worker(s)")

        # If we got fewer results than processes, some workers may have failed
        if len(all_results) < len(processes):
            missing_workers = len(processes) - len(all_results)
            self.logger.warning(f"{missing_workers} worker(s) completed without reporting results")

        # Clean up processes
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

        # Recalculate total curves from actual results
        actual_curves_completed = sum(result['curves_completed'] for result in all_results)

        # Remove duplicate sigma values and log summary
        unique_sigma_values = list(set(all_sigma_values))
        self.logger.info(f"Multiprocess run used {len(unique_sigma_values)} unique sigma values")

        # Extract parametrization from sigma values (format: "1:xxx" or "3:xxx")
        parametrization = None
        if unique_sigma_values:
            first_sigma = unique_sigma_values[0]
            if ':' in first_sigma:
                parametrization = int(first_sigma.split(':')[0])
            else:
                parametrization = 3  # Default

        # Fully factor any composite factors found
        if factor_found:
            processor = ResultProcessor(self, composite, method, b1, b2, curves, f"GMP-ECM ({method.upper()})")
            # Store factor and sigma first
            results['factor_found'] = factor_found
            results['factors_found'] = [factor_found]
            results['sigma'] = factor_sigma
            # Then fully factor it
            processor.fully_factor_and_store([factor_found], results, quiet=False)
        else:
            # No factors found
            if 'factor_found' not in results:
                results['factor_found'] = None

        # Set sigma and parametrization
        if 'sigma' not in results:
            results['sigma'] = factor_sigma  # Sigma that found the factor (if any)
        results['sigma_values'] = unique_sigma_values  # All sigma values used
        results['parametrization'] = parametrization
        results['curves_completed'] = actual_curves_completed
        results['execution_time'] = time.time() - start_time

        return results

    def run_stage2_only(self, residue_file: str, b1: int, b2: int,
                       stage2_workers: int = 4, verbose: bool = False,
                       continue_after_factor: bool = False,
                       progress_interval: int = 0) -> Dict[str, Any]:
        """Run Stage 2 only on existing residue file"""

        residue_path = Path(residue_file)
        if not residue_path.exists():
            self.logger.error(f"Residue file not found: {residue_path}")
            return {
                'residue_file': residue_file,
                'factor_found': None,
                'execution_time': 0,
                'error': 'Residue file not found'
            }

        # Extract composite number, curve count, and B1 from residue file
        residue_info = self._parse_residue_file(residue_path)
        composite = residue_info['composite']
        stage1_curves = residue_info['curve_count']
        stage1_b1 = residue_info['b1']

        self.logger.info(f"Running Stage 2 on residue file: {residue_path}")
        if composite != "unknown":
            self.logger.info(f"Composite: {composite[:20]}...{composite[-20:]} ({len(composite)} digits)")
        if stage1_curves > 0:
            self.logger.info(f"Stage 1 completed {stage1_curves} curves")
        # Use B1 from residue file instead of parameter
        actual_b1 = stage1_b1 if stage1_b1 > 0 else b1
        self.logger.info(f"Using {stage2_workers} CPU workers, B1={actual_b1}, B2={b2}")
        if stage1_b1 > 0 and stage1_b1 != b1:
            self.logger.info(f"Note: Using B1={actual_b1} from residue file (overriding parameter B1={b1})")

        start_time = time.time()

        # Run stage 2 multithread
        early_termination = self.config['programs']['gmp_ecm'].get('early_termination', True) and not continue_after_factor
        if continue_after_factor:
            self.logger.info("Early termination disabled due to --continue-after-factor flag")
        stage2_result = self._run_stage2_multithread(
            residue_path, actual_b1, b2, stage2_workers, verbose, early_termination, progress_interval
        )

        # Extract factor and sigma from result
        factor_found = None
        sigma_found = None
        if stage2_result:
            factor_found, sigma_found = stage2_result

        execution_time = time.time() - start_time

        results = {
            'residue_file': residue_file,
            'composite': composite,
            'b1': actual_b1,
            'b2': b2,
            'curves_requested': stage1_curves,  # Curves that were run in Stage 1
            'curves_completed': stage1_curves,  # All Stage 1 curves were completed
            'stage2_workers': stage2_workers,
            'factor_found': factor_found,
            'execution_time': execution_time,
            'method': 'ecm',
            'raw_output': ''  # Stage 2 only doesn't have single raw output
        }

        if factor_found:
            self.logger.info(f"Factor found in Stage 2: {factor_found}")
            self.log_factor_found(composite, factor_found, actual_b1, b2, stage1_curves, method="ecm", sigma=sigma_found, program="GMP-ECM (ECM-STAGE2)")
        else:
            self.logger.info("No factor found in Stage 2")

        return results

    def _parse_residue_file(self, residue_path: Path) -> Dict[str, Any]:
        """
        Parse ECM residue file and extract all metadata in a single pass.

        Returns:
            Dict with keys: composite, b1, curve_count
        """
        metadata = self.residue_manager.parse_metadata(str(residue_path))

        if metadata:
            composite, b1, curve_count = metadata
            return {
                'composite': composite,
                'b1': b1,
                'curve_count': curve_count
            }
        else:
            # Return default values on parse failure
            return {
                'composite': 'unknown',
                'b1': 0,
                'curve_count': 0
            }

    def _correlate_factor_to_sigma(self, factor: str, residue_path: Path) -> Optional[str]:
        """
        Try to determine which sigma value found the factor by testing each residue.
        This is a fallback when ECM output doesn't contain sigma information.
        """
        sigma = self.residue_manager.correlate_factor_to_sigma(factor, str(residue_path))

        if sigma:
            # Format as "3:sigma" (ECM format with parametrization 3)
            # If sigma already contains parametrization prefix, use as-is
            if ':' not in sigma:
                return f"3:{sigma}"
            return sigma

        return None




    def get_program_version(self, program: str) -> str:
        """Override base class method to get GMP-ECM version"""
        return self.get_ecm_version()

    def get_ecm_version(self) -> str:
        """Get GMP-ECM version"""
        try:
            result = subprocess.run(
                [self.config['programs']['gmp_ecm']['path'], '-h'],
                capture_output=True,
                text=True
            )
            from parsing_utils import extract_program_version
            return extract_program_version(result.stdout, 'ecm')
        except:
            pass
        return "unknown"

    def _trial_division(self, n: int, limit: int = 10**7) -> Tuple[List[int], int]:
        """
        Fast trial division to find small prime factors.

        Args:
            n: Number to factor
            limit: Trial division limit (default: 10^7)

        Returns:
            Tuple of (factors_found, cofactor)
        """
        factors = []
        cofactor = n

        # Trial division by 2
        while cofactor % 2 == 0:
            factors.append(2)
            cofactor //= 2

        # Trial division by 3
        while cofactor % 3 == 0:
            factors.append(3)
            cofactor //= 3

        # Trial division by 5
        while cofactor % 5 == 0:
            factors.append(5)
            cofactor //= 5

        # Trial division by odd numbers
        i = 7
        while i * i <= cofactor and i <= limit:
            while cofactor % i == 0:
                factors.append(i)
                cofactor //= i
            i += 2

        return factors, cofactor

    def _fully_factor_found_result(self, factor: str, max_ecm_attempts: int = 5, quiet: bool = False) -> List[str]:
        """
        Recursively factor a result from ECM until all prime factors found.
        Handles composite factors by using trial division + ECM with increasing B1.

        Args:
            factor: Factor found by ECM (may be composite)
            max_ecm_attempts: Maximum ECM attempts with increasing B1 (default: 5)

        Returns:
            List of prime factors (as strings)
        """
        factor_int = int(factor)

        # Trial division catches small factors quickly (2, 3, 5, 7, ... up to 10^7)
        small_primes, cofactor = self._trial_division(factor_int, limit=10**7)
        all_primes = [str(p) for p in small_primes]

        if cofactor == 1:
            return all_primes

        # Check if cofactor is prime using probabilistic test
        if self._is_probably_prime(cofactor):
            self.logger.info(f"Cofactor {cofactor} is prime")
            all_primes.append(str(cofactor))
            return all_primes

        # Cofactor is composite - use ECM with increasing B1
        digit_length = len(str(cofactor))
        self.logger.info(f"Cofactor remaining: C{digit_length}, using ECM to complete factorization")

        current_cofactor = cofactor
        for attempt in range(max_ecm_attempts):
            if current_cofactor == 1:
                break

            # Select B1 based on cofactor size
            cofactor_digits = len(str(current_cofactor))
            b1 = self._get_b1_for_digit_length(cofactor_digits)

            # Use more curves for smaller numbers (they're faster)
            curves = max(10, 50 - (cofactor_digits // 2))

            self.logger.info(f"ECM attempt {attempt+1}/{max_ecm_attempts} on C{cofactor_digits} with B1={b1}, {curves} curves")

            try:
                ecm_result = self.run_ecm(
                    composite=str(current_cofactor),
                    b1=b1,
                    curves=curves,
                    verbose=False,
                    quiet=quiet
                )

                found_factors = ecm_result.get('factors_found', [])
                if not found_factors and ecm_result.get('factor_found'):
                    found_factors = [ecm_result['factor_found']]

                if found_factors:
                    self.logger.info(f"ECM found {len(found_factors)} factor(s): {found_factors}")

                    # Recursively factor each found factor
                    for found_factor in found_factors:
                        sub_primes = self._fully_factor_found_result(found_factor, max_ecm_attempts, quiet=quiet)
                        all_primes.extend(sub_primes)

                        # Divide out from cofactor
                        for prime in sub_primes:
                            current_cofactor //= int(prime)

                    # Check if fully factored
                    if current_cofactor == 1:
                        break

                    # Check if remaining cofactor is prime
                    if self._is_probably_prime(current_cofactor):
                        self.logger.info(f"Remaining cofactor {current_cofactor} is prime")
                        all_primes.append(str(current_cofactor))
                        current_cofactor = 1
                        break
                else:
                    self.logger.info(f"No factor found in attempt {attempt+1}")

            except Exception as e:
                self.logger.error(f"ECM factorization error: {e}")
                break

        # If we still have a composite cofactor after all attempts, return it as-is
        if current_cofactor > 1:
            self.logger.warning(f"Could not fully factor C{len(str(current_cofactor))}: {current_cofactor}")
            all_primes.append(str(current_cofactor))

        return all_primes

    def _is_probably_prime(self, n: int, trials: int = 10) -> bool:
        """
        Miller-Rabin primality test.

        Args:
            n: Number to test
            trials: Number of trials (default: 10)

        Returns:
            True if probably prime, False if definitely composite
        """
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False

        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2

        # Witness loop
        import random
        for _ in range(trials):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)

            if x == 1 or x == n - 1:
                continue

            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False

        return True

    def _calculate_tlevel(self, curve_history: List[str]) -> float:
        """
        Call t-level binary to calculate current t-level.

        Args:
            curve_history: List of curve strings like "100@1000000,p=1"

        Returns:
            Current t-level as float
        """
        import re

        if not curve_history:
            return 0.0

        tlevel_path = self.config.get('programs', {}).get('t_level', {}).get('path', 'bin/t-level')

        # Join curve strings with semicolons
        curve_input = ";".join(curve_history)

        try:
            # Call t-level binary
            result = subprocess.run(
                [tlevel_path, '-q', curve_input],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse output: "t40.234"
            match = re.search(r't([\d.]+)', result.stdout)
            if match:
                return float(match.group(1))

            self.logger.warning(f"Failed to parse t-level from: {result.stdout}")
            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating t-level: {e}")
            return 0.0

    def _calculate_curves_for_target(self, current_tlevel: float, target_tlevel: float, b1: int) -> Optional[int]:
        """
        Calculate exact number of curves needed to reach target t-level from current t-level.

        Args:
            current_tlevel: Current t-level (e.g., 25.084)
            target_tlevel: Target t-level (e.g., 28.7)
            b1: B1 value to use

        Returns:
            Number of curves needed, or None if calculation failed
        """
        import re

        tlevel_path = self.config.get('programs', {}).get('t_level', {}).get('path', 'bin/t-level')

        try:
            # Call t-level binary: t-level -w <current> -t <target> -b <b1>
            result = subprocess.run(
                [tlevel_path, '-w', str(current_tlevel), '-t', str(target_tlevel), '-b', str(b1)],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse output like:
            # "Running the following will get you to t28.700:"
            # "262@25e4"
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if 'will get you to' in line:
                    # Check next line for the recommendation
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        match = re.search(r'(\d+)@', next_line)
                        if match:
                            return int(match.group(1))
                # Also check if format is on same line
                elif '@' in line and re.match(r'^\d+@', line.strip()):
                    match = re.search(r'(\d+)@', line)
                    if match:
                        return int(match.group(1))

            self.logger.warning(f"Failed to parse curve recommendation from t-level output: {result.stdout}")
            return None

        except Exception as e:
            self.logger.error(f"Error calculating curves for target: {e}")
            return None

    def _get_b1_for_digit_length(self, digits: int) -> int:
        """
        Select appropriate B1 value based on digit length.
        Uses GMP-ECM recommended parameters.

        Args:
            digits: Digit length of composite

        Returns:
            Recommended B1 value
        """
        # Based on GMP-ECM recommendations
        if digits < 30: return 11000
        elif digits < 40: return 50000
        elif digits < 50: return 250000
        elif digits < 60: return 1000000
        elif digits < 70: return 3000000
        elif digits < 80: return 11000000
        elif digits < 90: return 43000000
        elif digits < 100: return 110000000
        elif digits < 110: return 260000000
        elif digits < 120: return 850000000
        else: return 2900000000

    def _get_optimal_b1_for_tlevel(self, target_tlevel: float) -> Tuple[int, int]:
        """
        Get optimal B1 value and expected curves for reaching a specific t-level.
        Based on Zimmermann's table for GMP-ECM optimal plans.

        Args:
            target_tlevel: Target t-level in digits (e.g., 20, 25, 30, 35...)

        Returns:
            Tuple of (optimal_b1, expected_curves) for that t-level
        """
        # Zimmermann's optimal B1 values and expected curves for GMP-ECM 7
        # Format: (digits, optimal_b1, expected_curves_ecm7)
        # Source: https://members.loria.fr/PZimmermann/records/ecmnet.html
        OPTIMAL_B1_TABLE = [
            (20, 11000, 107),
            (25, 50000, 261),
            (30, 250000, 513),
            (35, 1000000, 1071),
            (40, 3000000, 2753),
            (45, 11000000, 5208),
            (50, 43000000, 8704),
            (55, 110000000, 20479),
            (60, 260000000, 47888),
            (65, 850000000, 78923),
            (70, 2900000000, 115153),
            (75, 7600000000, 211681),
            (80, 25000000000, 296479)
        ]

        # Find the closest entry (linear interpolation if between values)
        if target_tlevel <= OPTIMAL_B1_TABLE[0][0]:
            return (OPTIMAL_B1_TABLE[0][1], OPTIMAL_B1_TABLE[0][2])
        if target_tlevel >= OPTIMAL_B1_TABLE[-1][0]:
            return (OPTIMAL_B1_TABLE[-1][1], OPTIMAL_B1_TABLE[-1][2])

        # Find surrounding entries and interpolate
        for i in range(len(OPTIMAL_B1_TABLE) - 1):
            digits1, b1_1, curves1 = OPTIMAL_B1_TABLE[i]
            digits2, b1_2, curves2 = OPTIMAL_B1_TABLE[i + 1]

            if digits1 <= target_tlevel <= digits2:
                # Linear interpolation in log space for B1 (grows exponentially)
                # Linear interpolation in linear space for curves
                import math
                log_b1_1 = math.log(b1_1)
                log_b1_2 = math.log(b1_2)
                fraction = (target_tlevel - digits1) / (digits2 - digits1)
                log_b1 = log_b1_1 + fraction * (log_b1_2 - log_b1_1)
                optimal_b1 = int(math.exp(log_b1))
                expected_curves = int(curves1 + fraction * (curves2 - curves1))
                return (optimal_b1, expected_curves)

        # Fallback
        return (OPTIMAL_B1_TABLE[-1][1], OPTIMAL_B1_TABLE[-1][2])

    def run_ecm_with_tlevel(self, composite: str, target_tlevel: float,
                           batch_size: int = 100, workers: int = 1,
                           use_two_stage: bool = False, verbose: bool = False,
                           start_b1: Optional[int] = None, no_submit: bool = False,
                           project: Optional[str] = None) -> Dict[str, Any]:
        """
        Run ECM progressively until target t-level reached.
        Uses progressive approach: starts at t20 and increases by 5 digits each step.
        Fully factors any composite factors found.

        Args:
            composite: Number to factor
            target_tlevel: Target t-level in digits (e.g., 30.0, 40.0)
            batch_size: DEPRECATED - uses optimal curve counts from Zimmermann table
            workers: Number of parallel workers (default: 1)
            use_two_stage: Use two-stage GPU mode (default: False)
            verbose: Verbose output
            start_b1: DEPRECATED - uses optimal B1 from Zimmermann table
            no_submit: Skip API submission if True (default: False)
            project: Optional project name for API submission

        Returns:
            Results dict with:
            - success: bool
            - factors_found: List of all prime factors (or empty if none found)
            - final_cofactor: Remaining cofactor as string (or None if fully factored)
            - all_prime_factors: Alias for factors_found
            - curves_completed: Total curves run
            - execution_time: Time in seconds
        """
        current_composite = int(composite)
        original_digits = len(str(current_composite))
        all_prime_factors = []
        total_curves = 0
        curve_history = []  # Track all curves run for t-level calculation

        self.logger.info(f"Starting progressive ECM with target t{target_tlevel:.1f} on C{original_digits}")

        start_time = time.time()

        # Start at t0, work our way up in steps of 5 digits (20, 25, 30, 35...)
        current_t_level = 0.0
        step_targets = []
        t = 20.0
        while t <= target_tlevel:
            step_targets.append(t)
            t += 5.0
        # Add final target if it's not already a step
        if target_tlevel not in step_targets:
            step_targets.append(target_tlevel)

        self.logger.info(f"Progressive steps: {' → '.join([f't{t:.1f}' for t in step_targets])}")

        for step_target in step_targets:
            if current_composite == 1:
                break

            cofactor_digits = len(str(current_composite))

            # Get optimal B1 for this step from Zimmermann table
            optimal_b1, _ = self._get_optimal_b1_for_tlevel(step_target)

            # Calculate exact curves needed to reach this step from current position
            curves_needed = self._calculate_curves_for_target(current_t_level, step_target, optimal_b1)

            if curves_needed is None or curves_needed <= 0:
                self.logger.warning(f"Could not calculate curves for t{current_t_level:.3f} → t{step_target:.1f}, using Zimmermann estimate")
                # Fallback to Zimmermann table estimate
                _, curves_needed = self._get_optimal_b1_for_tlevel(step_target)

            self.logger.info(f"Progressive ECM: t{current_t_level:.3f} → t{step_target:.1f} on C{cofactor_digits}, B1={optimal_b1}, {curves_needed} curves")

            # Run ECM with exact number of curves at this B1
            if use_two_stage:
                batch_results = self.run_ecm_two_stage(
                    composite=str(current_composite),
                    b1=optimal_b1,
                    curves=curves_needed,
                    use_gpu=True,
                    verbose=verbose
                )
            elif workers > 1:
                batch_results = self.run_ecm_multiprocess(
                    composite=str(current_composite),
                    b1=optimal_b1,
                    curves=curves_needed,
                    workers=workers,
                    verbose=verbose
                )
            else:
                batch_results = self.run_ecm(
                    composite=str(current_composite),
                    b1=optimal_b1,
                    curves=curves_needed,
                    verbose=verbose
                )

            curves_completed = batch_results.get('curves_completed', 0)
            total_curves += curves_completed

            # Update curve history and calculate new t-level
            param = batch_results.get('parametrization', 1)
            curve_history.append(f"{curves_completed}@{optimal_b1},p={param}")
            current_t_level = self._calculate_tlevel(curve_history)
            self.logger.info(f"Reached t{current_t_level:.3f} after {curves_completed} curves")

            # Submit this step's results if submission enabled
            if not no_submit and curves_completed > 0:
                program_name = f'gmp-ecm-{batch_results.get("method", "ecm")}'
                self.logger.info(f"Submitting results for t{step_target:.1f} step (B1={optimal_b1}, {curves_completed} curves)")
                self.submit_result(batch_results, project, program_name)

            # Handle factors
            found_factors = batch_results.get('factors_found', [])
            if found_factors:
                self.logger.info(f"Found {len(found_factors)} factor(s): {found_factors}")
                all_prime_factors.extend(found_factors)

                # Divide out all found factors
                for factor in found_factors:
                    current_composite //= int(factor)

                new_digits = len(str(current_composite))
                self.logger.info(f"Cofactor reduced from C{cofactor_digits} to C{new_digits}")

                # Check if fully factored
                if current_composite == 1:
                    self.logger.info("Fully factored by progressive ECM")
                    break

                # Reset curve history for the new cofactor
                # The work we've done doesn't directly apply to the smaller cofactor
                curve_history = []
                current_t_level = 0.0
                self.logger.info(f"Starting fresh t-level progression on C{new_digits} cofactor")
            else:
                self.logger.info(f"No factors found at this step")

        execution_time = time.time() - start_time

        results = {
            'success': True,
            'factors_found': all_prime_factors,
            'all_prime_factors': all_prime_factors,  # Alias for compatibility
            'final_cofactor': str(current_composite) if current_composite > 1 else None,
            'current_tlevel': current_t_level,
            'target_tlevel': target_tlevel,
            'curves_completed': total_curves,
            'execution_time': execution_time,
            'method': 'ecm',
            'original_digits': original_digits,
            'final_cofactor_digits': len(str(current_composite)) if current_composite > 1 else 0
        }

        if all_prime_factors:
            self.logger.info(f"Progressive ECM found {len(all_prime_factors)} prime factor(s): {all_prime_factors}")

        if current_composite > 1:
            self.logger.info(f"Cofactor remaining: C{len(str(current_composite))} at t{current_t_level:.3f}/{target_tlevel:.1f}")
        else:
            self.logger.info(f"Fully factored at t{current_t_level:.3f}")

        return results


def main():
    from arg_parser import create_ecm_parser, validate_ecm_args, print_validation_errors
    from arg_parser import get_method_defaults, resolve_gpu_settings, resolve_worker_count, get_stage2_workers_default

    parser = create_ecm_parser()
    args = parser.parse_args()

    wrapper = ECMWrapper(args.config)

    # Validate arguments with config context
    errors = validate_ecm_args(args, wrapper.config)
    print_validation_errors(errors)

    # Use shared argument processing utilities
    b1_default, b2_default = get_method_defaults(wrapper.config, args.method)
    b1 = args.b1 or b1_default
    b2 = args.b2 if args.b2 is not None else b2_default
    curves = args.curves if args.curves is not None else wrapper.config['programs']['gmp_ecm']['default_curves']

    # Resolve GPU settings using shared utility
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)

    # Resolve worker count
    args.workers = resolve_worker_count(args)

    # Resolve stage2 workers from config if not explicitly set
    stage2_workers = args.stage2_workers if hasattr(args, 'stage2_workers') and args.stage2_workers != 4 else get_stage2_workers_default(wrapper.config)

    # Check for T-level mode first (highest priority)
    if hasattr(args, 'tlevel') and args.tlevel:
        # T-level mode: run ECM iteratively until target t-level reached
        results = wrapper.run_ecm_with_tlevel(
            composite=args.composite,
            target_tlevel=args.tlevel,
            batch_size=args.batch_size,
            workers=args.workers if args.multiprocess else 1,
            use_two_stage=args.two_stage,
            verbose=args.verbose,
            no_submit=args.no_submit,
            project=args.project
        )
    # Run ECM - choose mode based on arguments (validation already done by validate_ecm_args)
    elif args.resume_residues:
        # Resume from existing residues - run stage 2 only
        results = wrapper.run_stage2_only(
            residue_file=args.resume_residues,
            b1=b1,
            b2=b2,
            stage2_workers=stage2_workers,
            verbose=args.verbose,
            continue_after_factor=args.continue_after_factor,
            progress_interval=args.progress_interval
        )
    elif args.stage2_only:
        # Stage 2 only mode
        results = wrapper.run_stage2_only(
            residue_file=args.stage2_only,
            b1=b1,
            b2=b2,
            stage2_workers=stage2_workers,
            verbose=args.verbose,
            continue_after_factor=args.continue_after_factor,
            progress_interval=args.progress_interval
        )
    elif args.multiprocess:
        results = wrapper.run_ecm_multiprocess(
            composite=args.composite,
            b1=b1,
            b2=b2,
            curves=curves,
            workers=args.workers,
            verbose=args.verbose,
            continue_after_factor=args.continue_after_factor,
            method=args.method
        )
    elif args.two_stage and args.method == 'ecm':
        # Parse sigma if provided (convert "N" to integer, keep "3:N" as string)
        sigma = None
        if hasattr(args, 'sigma') and args.sigma:
            sigma = args.sigma if ':' in args.sigma else int(args.sigma)

        # Get param if provided, default to 3 for GPU
        param = args.param if hasattr(args, 'param') and args.param is not None else (3 if use_gpu else None)

        results = wrapper.run_ecm_two_stage(
            composite=args.composite,
            b1=b1,
            b2=b2,
            curves=curves,
            sigma=sigma,
            param=param,
            use_gpu=use_gpu,
            stage2_workers=stage2_workers,
            verbose=args.verbose,
            save_residues=args.save_residues,
            resume_residues=args.resume_residues,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            continue_after_factor=args.continue_after_factor,
            progress_interval=args.progress_interval
        )
    else:
        # Standard mode
        if args.two_stage:
            print("Warning: Two-stage mode only available for ECM method, falling back to standard mode")

        # Parse sigma if provided (convert "N" to integer, keep "3:N" as string)
        sigma = None
        if hasattr(args, 'sigma') and args.sigma:
            sigma = args.sigma if ':' in args.sigma else int(args.sigma)

        # Get param if provided, default to 3 for GPU
        param = args.param if hasattr(args, 'param') and args.param is not None else (3 if use_gpu else None)

        results = wrapper.run_ecm(
            composite=args.composite,
            b1=b1,
            b2=b2,
            curves=curves,
            sigma=sigma,
            param=param,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            verbose=args.verbose,
            method=args.method,
            continue_after_factor=args.continue_after_factor
        )

    # Submit results unless disabled or failed
    # Skip submission for t-level mode (each step already submitted)
    if not args.no_submit and not (hasattr(args, 'tlevel') and args.tlevel):
        # Only submit if we actually completed some curves (not a failure)
        if results.get('curves_completed', 0) > 0:
            program_name = f'gmp-ecm-{results.get("method", "ecm")}'
            success = wrapper.submit_result(results, args.project, program_name)
            sys.exit(0 if success else 1)
        else:
            wrapper.logger.warning("Skipping result submission due to failure (0 curves completed)")
            sys.exit(1)

if __name__ == '__main__':
    main()
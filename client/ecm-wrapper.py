#!/usr/bin/env python3
import subprocess
import argparse
import time
import sys
import threading
import tempfile
import shutil
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from base_wrapper import BaseWrapper
from parsing_utils import parse_ecm_output, parse_ecm_output_multiple, count_ecm_steps_completed, Timeouts, ECMPatterns

def _run_worker_ecm_process(worker_id: int, composite: str, b1: int, b2: Optional[int],
                           curves: int, verbose: bool, method: str, ecm_path: str,
                           result_queue, stop_event) -> None:
    """Global worker function for multiprocessing"""
    import subprocess
    import re
    
    # Import the shared parsing function for worker processes
    from parsing_utils import parse_ecm_output as parse_ecm_output_local
    
    # Build command for this worker
    cmd = [ecm_path]
    
    # Add method-specific parameters
    if method == "pm1":
        cmd.append('-pm1')
    elif method == "pp1":
        cmd.append('-pp1')
    
    if verbose:
        cmd.append('-v')
    
    # Run specified number of curves
    cmd.extend(['-c', str(curves), str(b1)])
    if b2 is not None:
        cmd.append(str(b2))
    
    try:
        print(f"Worker {worker_id} starting {curves} curves")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Send composite number
        process.stdin.write(composite)
        process.stdin.close()
        
        # Capture output
        output_lines = []
        curves_completed = 0
        factor_found = None
        sigma_found = None
        sigma_values = []  # Collect all sigma values used
        
        while True:
            # Check stop event
            if stop_event.is_set():
                process.terminate()
                break
                
            line = process.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            if line:
                print(f"Worker {worker_id}: {line}")
                output_lines.append(line)
                
                # Track progress and check for factors
                if "Step 1 took" in line:
                    curves_completed += 1
                
                # Collect sigma values from curve output
                # Match both formats: "sigma=1:xxxx" and "-sigma 3:xxxx"
                sigma_match = ECMPatterns.SIGMA_COLON_FORMAT.search(line) or ECMPatterns.SIGMA_DASH_FORMAT.search(line)
                if sigma_match:
                    sigma_val = sigma_match.group(1)
                    if sigma_val not in sigma_values:
                        sigma_values.append(sigma_val)
                
                # Check for factor
                if not factor_found:
                    factor, sigma = parse_ecm_output_local(line)
                    if factor:
                        factor_found = factor
                        sigma_found = sigma
                        break  # Stop immediately when factor found
        
        process.wait()
        
        # If no factor found during streaming, check full output
        if not factor_found and not stop_event.is_set():
            full_output = '\n'.join(output_lines)
            factor_found, sigma_found = parse_ecm_output_local(full_output)
            curves_completed = curves  # All curves completed
        
        result = {
            'worker_id': worker_id,
            'factor_found': factor_found,
            'sigma_found': sigma_found,  # Sigma of the curve that found the factor
            'sigma_values': sigma_values,  # All sigma values used by this worker
            'curves_completed': curves_completed if not stop_event.is_set() else curves_completed,
            'raw_output': '\n'.join(output_lines)
        }
        
        result_queue.put(result)
        
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        result_queue.put({
            'worker_id': worker_id,
            'factor_found': None,
            'sigma_found': None,
            'sigma_values': [],
            'curves_completed': 0,
            'raw_output': f"Worker failed: {e}"
        })

class ECMWrapper(BaseWrapper):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def _log_and_store_factors(self, all_factors: List[Tuple[str, Optional[str]]],
                               results: Dict[str, Any], composite: str, b1: int,
                               b2: Optional[int], curves: int, method: str,
                               program: str) -> Optional[str]:
        """
        Deduplicate factors, log them, and store in results dictionary.

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
        if not all_factors:
            return None

        # Deduplicate factors - same factor can be found by multiple curves
        unique_factors = {}
        for factor, sigma in all_factors:
            if factor not in unique_factors:
                unique_factors[factor] = sigma  # Keep first sigma found

        # Log each unique factor once
        for factor, sigma in unique_factors.items():
            self.log_factor_found(composite, factor, b1, b2, curves,
                                method=method, sigma=sigma, program=program)

        # Store all unique factors for API submission
        if 'factors_found' not in results:
            results['factors_found'] = []
        results['factors_found'].extend(unique_factors.keys())

        # Set the main factor for compatibility (use first factor found)
        main_factor = list(unique_factors.keys())[0]
        results['factor_found'] = main_factor

        self.logger.info(f"Factors found: {list(unique_factors.keys())}")

        return main_factor
    
    def run_ecm(self, composite: str, b1: int, b2: Optional[int] = None,
                curves: int = 100, sigma: Optional[int] = None,
                use_gpu: bool = False, gpu_device: Optional[int] = None,
                gpu_curves: Optional[int] = None, verbose: bool = False,
                method: str = "ecm", continue_after_factor: bool = False) -> Dict[str, Any]:
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

                    # Handle all factors found
                    program_name = f"GMP-ECM ({method.upper()})" + (" with GPU" if use_gpu else "")
                    self._log_and_store_factors(all_factors, results, composite, b1, b2, curves, method, program_name)

                    self.logger.info(f"Factors found after {results['curves_completed']} curves")

                    if not continue_after_factor:
                        break
                    else:
                        self.logger.info("Continuing to process remaining curves due to --continue-after-factor flag")

                # Update curves completed for this batch
                curves_completed += curves_this_batch
                results['curves_completed'] = curves_completed

            except Exception as e:
                self.logger.error(f"Error in batch starting at curve {curves_completed + 1}: {e}")
                break
        
        results['execution_time'] = time.time() - start_time
        results['raw_output'] = '\n'.join(results['raw_outputs'])

        # Final deduplication of factors found across all batches
        if 'factors_found' in results and results['factors_found']:
            results['factors_found'] = list(dict.fromkeys(results['factors_found']))  # Preserve order while deduplicating

        # Save raw output if configured
        if self.config['execution']['save_raw_output']:
            self.save_raw_output(results, f'gmp-ecm-{method}')

        return results
    
    def run_ecm_two_stage(self, composite: str, b1: int, b2: Optional[int] = None,
                         curves: int = 100, use_gpu: bool = True,
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
                residue_file = Path(save_residues)
                residue_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Use configured residue directory with auto-generated filename
                residue_dir = Path(self.config['execution']['residue_dir'])
                residue_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename: residue_<composite_hash>_<timestamp>.txt
                import hashlib
                composite_hash = hashlib.md5(composite.encode()).hexdigest()[:12]
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                residue_file = residue_dir / f"residue_{composite_hash}_{timestamp}.txt"
                self.logger.info(f"Using auto-generated residue file: {residue_file}")
            
            actual_curves = curves  # Initialize fallback
            try:
                # Stage 1: GPU or CPU execution with residue saving
                stage1_mode = "GPU" if use_gpu else "CPU"
                self.logger.info(f"Starting Stage 1 ({stage1_mode})")
                stage1_success, stage1_factor, actual_curves, stage1_output, all_stage1_factors = self._run_stage1(
                    composite, b1, curves, residue_file, use_gpu, verbose, gpu_device, gpu_curves
                )

                if stage1_factor:
                    # Log ALL unique factors found in Stage 1
                    if all_stage1_factors:
                        self._log_and_store_factors(all_stage1_factors, results, composite, b1, b2, curves, "ecm", "GMP-ECM (ECM)")
                    else:
                        # Fallback to single factor logging
                        self.log_factor_found(composite, stage1_factor, b1, b2, curves, method="ecm", sigma=None, program="GMP-ECM (ECM)")
                        results['factor_found'] = stage1_factor

                    results['curves_completed'] = actual_curves
                    results['execution_time'] = time.time() - start_time
                    results['raw_output'] = stage1_output
                    self.logger.info(f"Factor found in Stage 1: {stage1_factor}")
                    return results
                
                if not stage1_success:
                    self.logger.error("Stage 1 failed")
                    results['execution_time'] = time.time() - start_time
                    return results

                if not residue_file.exists():
                    self.logger.error(f"No residue file generated at: {residue_file}")
                    results['execution_time'] = time.time() - start_time
                    return results

                # Check if residue file has content
                if residue_file.stat().st_size == 0:
                    self.logger.error(f"Residue file is empty: {residue_file}")
                    results['execution_time'] = time.time() - start_time
                    return results

                self.logger.info(f"Residue file created successfully: {residue_file} ({residue_file.stat().st_size} bytes)")

                if save_residues:
                    self.logger.info(f"Stage 1 residues saved to: {residue_file}")

            except Exception as e:
                self.logger.error(f"Stage 1 execution failed: {e}")
                results['execution_time'] = time.time() - start_time
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
            results['factor_found'] = factor_found
            stage_found = "Stage 1" if stage1_factor else "Stage 2"
            sigma_used = stage2_sigma if stage2_factor else None
            self.logger.info(f"Factor found in {stage_found}: {factor_found}")
            self.log_factor_found(composite, factor_found, b1, b2, curves, method="ecm", sigma=sigma_used, program="GMP-ECM (ECM)")
        
        results['curves_completed'] = actual_curves
        results['execution_time'] = time.time() - start_time

        return results
    
    def _run_stage1(self, composite: str, b1: int, curves: int,
                   residue_file: Path, use_gpu: bool, verbose: bool,
                   gpu_device: Optional[int] = None, gpu_curves: Optional[int] = None) -> tuple[bool, Optional[str], int, str, List[tuple[str, Optional[str]]]]:
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
            
            return process.returncode == 0, factor, actual_curves, output, all_factors
            
        except Exception as e:
            self.logger.error(f"Stage 1 GPU execution failed: {e}")
            return False, None, curves, "", []
    
    def _run_stage2_multithread(self, residue_file: Path, b1: int, b2: int,
                               workers: int, verbose: bool, early_termination: bool = True,
                               progress_interval: int = 0) -> Optional[str]:
        """Run Stage 2 with multiple CPU workers"""

        # Extract B1 from residue file to ensure consistency
        residue_info = self._parse_residue_file(residue_file)
        actual_b1 = residue_info['b1']
        if actual_b1 > 0 and actual_b1 != b1:
            self.logger.info(f"Using B1={actual_b1} from residue file (overriding parameter B1={b1})")
        b1_to_use = actual_b1 if actual_b1 > 0 else b1

        # Split residue file into chunks for workers
        residue_chunks = self._split_residue_file(residue_file, workers)
        
        if not residue_chunks:
            self.logger.error("Failed to split residue file")
            return None
        
        factor_found = None
        factor_lock = threading.Lock()
        stop_event = threading.Event()
        running_processes = []
        process_lock = threading.Lock()
        
        def worker_stage2(chunk_file: Path, worker_id: int) -> Optional[tuple[str, str]]:
            """Worker function for Stage 2 processing"""
            ecm_path = self.config['programs']['gmp_ecm']['path']

            cmd = [ecm_path, '-resume', str(chunk_file)]
            if verbose:
                cmd.append('-v')
            cmd.extend([str(b1_to_use), str(b2)])

            # Count total lines in this worker's chunk for progress reporting
            total_lines = 0
            if verbose:
                try:
                    with open(chunk_file, 'r') as f:
                        total_lines = sum(1 for _ in f)
                except:
                    total_lines = 0

            try:
                self.logger.info(f"Worker {worker_id} starting Stage 2" +
                               (f" ({total_lines} curves)" if verbose and total_lines > 0 else ""))
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Register process for potential termination
                with process_lock:
                    running_processes.append(process)
                
                # Stream output for progress tracking if progress_interval is set
                if progress_interval > 0:
                    full_output = ""
                    last_progress_report = 0

                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break

                        full_output += line

                        # Check if we should terminate early
                        if early_termination and stop_event.is_set():
                            process.terminate()
                            self.logger.info(f"Worker {worker_id} terminating due to factor found elsewhere")
                            return None

                        # Check for curve completion and progress reporting
                        if "Step 2 took" in line:
                            curves_completed = full_output.count("Step 2 took")

                            # Report progress at intervals
                            if curves_completed - last_progress_report >= progress_interval:
                                if total_lines > 0:
                                    percentage = (curves_completed / total_lines) * 100
                                    self.logger.info(f"Worker {worker_id}: {curves_completed}/{total_lines} curves ({percentage:.1f}%)")
                                else:
                                    self.logger.info(f"Worker {worker_id}: {curves_completed} curves completed")
                                last_progress_report = curves_completed

                    process.wait()
                else:
                    # Original behavior - get all output at once
                    stdout, stderr = process.communicate()
                    full_output = stdout if stdout else ""

                    # Check if we should terminate early due to factor found elsewhere
                    if early_termination and stop_event.is_set():
                        self.logger.info(f"Worker {worker_id} terminating due to factor found elsewhere")
                        return None

                    # Count curve completions from output
                    curves_completed = full_output.count("Step 2 took")

                    # Progress reporting in verbose mode
                    if verbose and total_lines > 0:
                        percentage = (curves_completed / total_lines) * 100
                        self.logger.info(f"Worker {worker_id} progress: {curves_completed}/{total_lines} curves - {percentage:.1f}% complete")

                    # Check for factor (common to both paths)
                    factor, sigma_from_output = parse_ecm_output(full_output)
                    if factor:
                        with factor_lock:
                            nonlocal factor_found
                            if not factor_found:  # First factor wins
                                factor_found = (factor, sigma_from_output)
                                if early_termination:
                                    stop_event.set()  # Signal other workers to stop
                                self.logger.info(f"Worker {worker_id} found factor: {factor} (sigma: {sigma_from_output})")
                                # Kill other processes if early termination enabled
                                if early_termination:
                                    with process_lock:
                                        for p in running_processes:
                                            if p != process and p.poll() is None:
                                                p.terminate()
                        return (factor, sigma_from_output)

                # If no factor found, report completion
                self.logger.info(f"Worker {worker_id} completed (no factor)")
                return None

            except Exception as e:
                self.logger.error(f"Worker {worker_id} failed: {e}")
                return None
            finally:
                # Remove process from tracking
                with process_lock:
                    if process in running_processes:
                        running_processes.remove(process)
        
        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, chunk_file in enumerate(residue_chunks):
                future = executor.submit(worker_stage2, chunk_file, i+1)
                futures.append(future)
            
            # Wait for completion or first factor
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        factor_found = result  # This is now (factor, sigma) tuple
                        stop_event.set()  # Ensure all workers are signaled to stop
                        break
                except Exception as e:
                    self.logger.error(f"Worker thread error: {e}")
            
            # Ensure all remaining processes are terminated
            with process_lock:
                for process in running_processes:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
        
        # Cleanup temporary chunk files and directory
        chunk_dirs_to_cleanup = set()
        for chunk_file in residue_chunks:
            try:
                chunk_dirs_to_cleanup.add(chunk_file.parent)
                chunk_file.unlink()
            except:
                pass

        # Clean up temporary chunk directories
        for chunk_dir in chunk_dirs_to_cleanup:
            try:
                import shutil
                shutil.rmtree(chunk_dir)
                self.logger.debug(f"Cleaned up chunk directory: {chunk_dir}")
            except:
                pass
        
        return factor_found
    
    def _split_residue_file(self, residue_file: Path, num_chunks: int) -> List[Path]:
        """Split residue file into chunks for parallel processing"""
        try:
            with open(residue_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                return []

            # Create unique temporary directory for chunk files to avoid conflicts between concurrent jobs
            import tempfile
            import os
            chunk_dir = tempfile.mkdtemp(prefix="ecm_chunks_")
            self.logger.debug(f"Creating chunks in temporary directory: {chunk_dir}")

            # Each residue typically spans multiple lines, but we'll split by line count
            # This is a simple approach - GMP-ECM can handle partial residue files
            chunk_size = max(1, len(lines) // num_chunks)
            chunks = []

            for i in range(num_chunks):
                start_idx = i * chunk_size
                if i == num_chunks - 1:  # Last chunk gets remaining lines
                    end_idx = len(lines)
                else:
                    end_idx = (i + 1) * chunk_size

                if start_idx >= len(lines):
                    break

                # Use unique temporary directory with process ID for chunk files
                chunk_file = Path(chunk_dir) / f"residues_chunk_{os.getpid()}_{i+1}.txt"
                with open(chunk_file, 'w') as f:
                    f.writelines(lines[start_idx:end_idx])

                if chunk_file.stat().st_size > 0:  # Only add non-empty chunks
                    chunks.append(chunk_file)

            self.logger.info(f"Split residue file into {len(chunks)} chunks in {chunk_dir}")
            return chunks

        except Exception as e:
            self.logger.error(f"Failed to split residue file: {e}")
            return []
    
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
                target=_run_worker_ecm_process,
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
        
        results['factor_found'] = factor_found
        results['sigma'] = factor_sigma  # Sigma that found the factor (if any)
        results['sigma_values'] = unique_sigma_values  # All sigma values used
        results['curves_completed'] = actual_curves_completed
        results['execution_time'] = time.time() - start_time
        
        if factor_found:
            program_name = f"GMP-ECM ({method.upper()})"
            self.log_factor_found(composite, factor_found, b1, b2, curves, method=method, sigma=factor_sigma, program=program_name)
        
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
        result = {
            'composite': 'unknown',
            'b1': 0,
            'curve_count': 0
        }

        try:
            with open(residue_path, 'r') as f:
                first_line = f.readline().strip()

                # Extract from first line
                if first_line:
                    n_match = ECMPatterns.RESUME_N_PATTERN.search(first_line)
                    if n_match:
                        result['composite'] = n_match.group(1)

                    b1_match = ECMPatterns.RESUME_B1_PATTERN.search(first_line)
                    if b1_match:
                        result['b1'] = int(b1_match.group(1))

                # Count curves (first line already consumed)
                if first_line.startswith('METHOD=ECM'):
                    result['curve_count'] = 1

                for line in f:
                    if line.strip().startswith('METHOD=ECM'):
                        result['curve_count'] += 1

        except Exception as e:
            self.logger.warning(f"Could not parse residue file: {e}")

        return result

    def _correlate_factor_to_sigma(self, factor: str, residue_path: Path) -> Optional[str]:
        """
        Try to determine which sigma value found the factor by testing each residue.
        This is a fallback when ECM output doesn't contain sigma information.
        """
        try:
            factor_int = int(factor)
            
            with open(residue_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and line.startswith('METHOD=ECM'):
                        # Extract sigma from this residue line
                        sigma_match = ECMPatterns.RESUME_SIGMA_PATTERN.search(line)
                        if sigma_match:
                            sigma = sigma_match.group(1)

                            # Extract N from this line to verify it matches
                            n_match = ECMPatterns.RESUME_N_PATTERN.search(line)
                            if n_match:
                                n = int(n_match.group(1))
                                
                                # Check if this factor divides N
                                if n % factor_int == 0:
                                    self.logger.info(f"Factor {factor} correlates to sigma {sigma} (line {line_num})")
                                    return f"3:{sigma}"  # ECM format
                            
        except Exception as e:
            self.logger.warning(f"Could not correlate factor to sigma: {e}")
        
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
    
    # Run ECM - choose mode based on arguments (validation already done by validate_ecm_args)
    if args.resume_residues:
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
        results = wrapper.run_ecm_two_stage(
            composite=args.composite,
            b1=b1,
            b2=b2,
            curves=curves,
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
        
        results = wrapper.run_ecm(
            composite=args.composite,
            b1=b1,
            b2=b2,
            curves=curves,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            verbose=args.verbose,
            method=args.method,
            continue_after_factor=args.continue_after_factor
        )
    
    # Submit results unless disabled
    if not args.no_submit:
        program_name = f'gmp-ecm-{results.get("method", "ecm")}'
        success = wrapper.submit_result(results, args.project, program_name)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
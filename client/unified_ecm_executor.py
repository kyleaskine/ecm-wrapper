#!/usr/bin/env python3
"""
Unified ECM executor that consolidates all execution modes.
Replaces the 4 different ECM execution patterns with a single optimized implementation.
"""
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from execution_engine import UnifiedExecutionEngine, ExecutionParams, StreamingOutputHandler
from result_processor import ConsolidatedResultProcessor, ProcessingResult
from optimized_parsing import OptimizedPatterns, StreamingParser

@dataclass
class ECMExecutionParams:
    """Unified ECM execution parameters."""
    composite: str
    b1: int
    b2: Optional[int] = None
    curves: int = 100
    method: str = "ecm"
    sigma: Optional[int] = None

    # Execution mode options
    use_gpu: bool = False
    gpu_device: Optional[int] = None
    gpu_curves: Optional[int] = None

    # Multi-processing options
    workers: int = 1
    stage2_workers: int = 4

    # Two-stage options
    two_stage: bool = False
    save_residues: Optional[str] = None
    resume_residues: Optional[str] = None

    # Control options
    verbose: bool = False
    continue_after_factor: bool = False
    progress_interval: int = 0

class UnifiedECMExecutor:
    """
    High-performance unified ECM executor.
    Consolidates all ECM execution modes into a single optimized implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ecm_path = config['programs']['gmp_ecm']['path']
        self.execution_engine = UnifiedExecutionEngine(config)
        self.result_processor = ConsolidatedResultProcessor(config)

    def execute(self, params: ECMExecutionParams) -> ProcessingResult:
        """
        Unified ECM execution that automatically chooses optimal strategy.

        Args:
            params: ECM execution parameters

        Returns:
            ProcessingResult with complete execution data
        """
        # Choose execution strategy based on parameters
        if params.resume_residues:
            return self._execute_stage2_only(params)
        elif params.two_stage:
            return self._execute_two_stage(params)
        elif params.workers > 1:
            return self._execute_multiprocess(params)
        else:
            return self._execute_standard(params)

    def _execute_standard(self, params: ECMExecutionParams) -> ProcessingResult:
        """Standard ECM execution (replaces run_ecm)."""
        cmd = self._build_ecm_command(params)

        exec_params = ExecutionParams(
            cmd=cmd,
            input_data=params.composite,
            timeout=3600,
            stream_output=params.progress_interval > 0
        )

        # Use streaming if progress reporting requested
        stream_handler = None
        if params.progress_interval > 0:
            stream_handler = self._create_streaming_handler(params)

        exec_result = self.execution_engine.execute_single(exec_params, stream_handler)

        # Process results
        result = self.result_processor.process_execution_result(
            exec_result, params.composite, params.method, f"gmp-ecm-{params.method}",
            curves=params.curves, b1=params.b1, b2=params.b2
        )

        # Handle factor logging and output saving
        if result.factors_found:
            self.result_processor.log_factors(result)

        if self.config.get('execution', {}).get('save_raw_output', False):
            self.result_processor.save_raw_output(result)

        return result

    def _execute_multiprocess(self, params: ECMExecutionParams) -> ProcessingResult:
        """Multiprocess ECM execution (replaces run_ecm_multiprocess)."""
        # Use the optimized multiprocess execution from execution engine
        engine_result = self.execution_engine.execute_multiprocess_ecm(
            params.composite, params.curves, params.workers,
            params.b1, params.b2, self.ecm_path, params.method
        )

        # Convert to ProcessingResult format
        result = ProcessingResult(
            composite=params.composite,
            method=params.method,
            program=f"gmp-ecm-{params.method}",
            success=engine_result['success'],
            execution_time=engine_result['execution_time'],
            curves_requested=params.curves,
            curves_completed=engine_result['curves_completed'],
            b1=params.b1,
            b2=params.b2,
            raw_output='\n'.join(engine_result['raw_outputs'])
        )

        # Parse factors from aggregated output
        if engine_result['factor_found']:
            from optimized_parsing import parse_ecm_output_multiple
            from result_processor import FactorResult

            factors = parse_ecm_output_multiple(result.raw_output)
            result.factors_found = [
                FactorResult(factor=f[0], sigma=f[1], method=params.method)
                for f in factors
            ]
            result.primary_factor = result.factors_found[0].factor if result.factors_found else None

        if result.factors_found:
            self.result_processor.log_factors(result)

        return result

    def _execute_two_stage(self, params: ECMExecutionParams) -> ProcessingResult:
        """Two-stage ECM execution (replaces run_ecm_two_stage)."""
        import logging
        logger = logging.getLogger(__name__)

        stage1_desc = "GPU" if params.use_gpu else "CPU"

        # Show curve count description based on GPU settings
        if params.use_gpu and params.gpu_curves is not None:
            curve_desc = "auto-selected curves"  # GPU will choose optimal count
        else:
            curve_desc = f"{params.curves} curves"

        logger.info(f"Running two-stage ECM: {stage1_desc} stage 1 ({curve_desc}) + {params.stage2_workers} CPU workers for stage 2")

        # Handle residue file setup
        residue_file = self._setup_residue_file(params)

        # Stage 1: GPU/CPU execution
        logger.info(f"Starting Stage 1 ({stage1_desc})")
        stage1_result = self._execute_stage1(params, residue_file)

        if stage1_result.factors_found:
            logger.info(f"Factor found in Stage 1: {stage1_result.primary_factor}")
            return stage1_result

        # Stage 2: Multi-threaded CPU execution
        if params.b2 and params.b2 > 0:
            logger.info(f"Starting Stage 2 ({params.stage2_workers} workers) with B1={params.b1}, B2={params.b2}")
            stage2_result = self._execute_stage2_multithread(
                residue_file, params.b1, params.b2, params.stage2_workers,
                params.verbose, not params.continue_after_factor, params.progress_interval
            )

            if stage2_result:
                # stage2_result is now a list of FactorResult objects
                if isinstance(stage2_result, list):
                    logger.info(f"Factors found in Stage 2: {[f.factor for f in stage2_result]}")
                    stage1_result.factors_found = stage2_result
                    stage1_result.primary_factor = stage2_result[0].factor
                else:
                    # Backward compatibility for single factor
                    logger.info(f"Factor found in Stage 2: {stage2_result.factor}")
                    stage1_result.factors_found = [stage2_result]
                    stage1_result.primary_factor = stage2_result.factor
        else:
            logger.info("Skipping Stage 2 (B2=0 - Stage 1 only mode)")

        # Cleanup
        self._cleanup_residue_file(params, residue_file)

        if stage1_result.factors_found:
            self.result_processor.log_factors(stage1_result)
        else:
            logger.info("No factor found in two-stage ECM")

        return stage1_result

    def _execute_stage2_only(self, params: ECMExecutionParams) -> ProcessingResult:
        """Stage 2 only execution (replaces run_stage2_only)."""
        residue_file = Path(params.resume_residues)

        if not residue_file.exists():
            return ProcessingResult(
                composite=params.composite,
                method="ecm",
                program="gmp-ecm-ecm",
                success=False,
                execution_time=0,
                error=f"Residue file not found: {residue_file}"
            )

        # Extract metadata from residue file
        composite = self._extract_composite_from_residue_file(residue_file)
        stage1_curves = self._extract_curve_count_from_residue_file(residue_file)
        stage1_b1 = self._extract_b1_from_residue_file(residue_file)

        # Execute stage 2
        stage2_result = self._execute_stage2_multithread(
            residue_file, stage1_b1 or params.b1, params.b2, params.stage2_workers,
            params.verbose, not params.continue_after_factor, params.progress_interval
        )

        result = ProcessingResult(
            composite=composite if composite != "unknown" else params.composite,
            method="ecm",
            program="gmp-ecm-ecm",
            success=stage2_result is not None,
            execution_time=0,  # Will be set by stage 2
            curves_requested=stage1_curves,
            curves_completed=stage1_curves,
            b1=stage1_b1 or params.b1,
            b2=params.b2
        )

        if stage2_result:
            result.factors_found = [stage2_result]
            result.primary_factor = stage2_result.factor
            self.result_processor.log_factors(result)

        return result

    def _build_ecm_command(self, params: ECMExecutionParams) -> List[str]:
        """Build ECM command efficiently."""
        cmd = [self.ecm_path]

        # Method-specific flags
        if params.method == "pm1":
            cmd.append('-pm1')
        elif params.method == "pp1":
            cmd.append('-pp1')

        # GPU options
        if params.use_gpu and params.method == "ecm":
            cmd.append('-gpu')
            if params.gpu_device is not None:
                cmd.extend(['-gpudevice', str(params.gpu_device)])
            if params.gpu_curves is not None:
                cmd.extend(['-gpucurves', str(params.gpu_curves)])

        # Verbose mode
        if params.verbose:
            cmd.append('-v')

        # Sigma for ECM
        if params.sigma and params.method == "ecm":
            cmd.extend(['-sigma', str(params.sigma)])

        # Parameters
        cmd.append(str(params.b1))
        if params.b2 is not None:
            cmd.append(str(params.b2))

        return cmd

    def _create_streaming_handler(self, params: ECMExecutionParams) -> StreamingOutputHandler:
        """Create streaming handler for progress reporting."""
        parser = StreamingParser()

        def parse_callback(line: str):
            result = parser.parse_line_ecm(line)
            if result:
                factor, sigma = result
                # Factor found - return it to trigger early termination
                return {"factor": factor, "sigma": sigma}

            # Progress reporting
            if params.progress_interval > 0 and parser.curves_completed % params.progress_interval == 0:
                print(f"Completed {parser.curves_completed} curves")

            return None

        return StreamingOutputHandler(parse_callback)

    def _setup_residue_file(self, params: ECMExecutionParams) -> Path:
        """Set up residue file for two-stage execution."""
        if params.save_residues:
            residue_file = Path(params.save_residues)
            residue_file.parent.mkdir(parents=True, exist_ok=True)
            return residue_file
        else:
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            return Path(temp_dir) / "stage1_residues.txt"

    def _cleanup_residue_file(self, params: ECMExecutionParams, residue_file: Path):
        """Clean up temporary residue files."""
        if not params.save_residues and residue_file.exists():
            try:
                residue_file.unlink()
                if residue_file.parent.name.startswith('tmp'):
                    residue_file.parent.rmdir()
            except:
                pass

    def _execute_stage1(self, params: ECMExecutionParams, residue_file: Path) -> ProcessingResult:
        """Execute Stage 1 with residue saving."""
        import logging
        logger = logging.getLogger(__name__)

        cmd = [self.ecm_path, '-save', str(residue_file)]

        if params.use_gpu:
            cmd.insert(1, '-gpu')
            if params.gpu_device is not None:
                cmd.extend(['-gpudevice', str(params.gpu_device)])
            if params.gpu_curves is not None:
                cmd.extend(['-gpucurves', str(params.gpu_curves)])

        if params.verbose:
            cmd.append('-v')

        # For GPU mode, let GPU determine curve count automatically if gpu_curves is set
        # Otherwise use the specified curves parameter
        if params.use_gpu and params.gpu_curves is not None:
            # When using GPU with specified gpu_curves, don't use -c parameter
            # The GPU will automatically choose the optimal number of curves
            cmd.extend([str(params.b1), '0'])  # B2=0 for stage 1 only
        else:
            # Standard CPU mode or GPU mode without gpu_curves specified
            cmd.extend(['-c', str(params.curves), str(params.b1), '0'])  # B2=0 for stage 1 only

        logger.info(f"Stage 1 command: {' '.join(cmd)}")

        exec_params = ExecutionParams(
            cmd=cmd,
            input_data=params.composite,
            timeout=3600,
            stream_output=params.verbose  # Stream output in verbose mode
        )

        # Create streaming handler for verbose output
        stream_handler = None
        if params.verbose:
            def log_stage1_line(line: str):
                logger.info(f"Stage1: {line}")
                return None

            from execution_engine import StreamingOutputHandler
            stream_handler = StreamingOutputHandler(log_stage1_line)

        exec_result = self.execution_engine.execute_single(exec_params, stream_handler)

        # Extract actual curve count from GPU output if applicable
        actual_curves = params.curves  # fallback to requested
        if params.use_gpu and exec_result.stdout:
            from optimized_parsing import extract_gpu_curve_count
            gpu_curves = extract_gpu_curve_count(exec_result.stdout)
            if gpu_curves > 0:
                actual_curves = gpu_curves
                logger.info(f"Stage 1 actually completed {actual_curves} curves")

        result = self.result_processor.process_execution_result(
            exec_result, params.composite, "ecm", "gmp-ecm-ecm",
            curves=actual_curves, b1=params.b1, b2=0
        )

        # Update the result with correct curve counts
        result.curves_requested = actual_curves
        result.curves_completed = actual_curves

        # Check if factors were found regardless of return code
        if result.factors_found:
            logger.info(f"Stage 1 found factor(s): {[f.factor for f in result.factors_found]}")
        elif exec_result.success:
            logger.info(f"Stage 1 completed successfully. Residue file: {residue_file}")
            if residue_file.exists():
                logger.info(f"Residue file size: {residue_file.stat().st_size} bytes")
            else:
                logger.warning(f"No residue file generated at: {residue_file}")
        else:
            logger.error(f"Stage 1 failed: {exec_result.error}")

        return result

    def _execute_stage2_multithread(self, residue_file: Path, b1: int, b2: int,
                                   workers: int, verbose: bool, early_termination: bool,
                                   progress_interval: int):
        """Execute Stage 2 with multiple workers."""
        import logging
        logger = logging.getLogger(__name__)

        # Split residue file into chunks for workers
        residue_chunks = self._split_residue_file(residue_file, workers)

        if not residue_chunks:
            logger.error("Failed to split residue file")
            return None

        logger.info(f"Split residue file into {len(residue_chunks)} chunks for {workers} workers")

        # Prepare execution parameters for each worker
        worker_params = []
        for i, chunk_file in enumerate(residue_chunks):
            cmd = [self.ecm_path, '-resume', str(chunk_file)]
            if verbose:
                cmd.append('-v')
            cmd.extend([str(b1), str(b2)])

            worker_params.append(ExecutionParams(
                cmd=cmd,
                timeout=7200,  # 2 hours for stage 2
                stream_output=verbose and progress_interval > 0
            ))

        # Execute all workers in parallel
        results = self.execution_engine.execute_batch(worker_params, max_workers=workers)

        # Check results for factors - collect ALL factors found
        factors_found = []
        all_factors_set = set()  # To avoid duplicates

        for i, result in enumerate(results):
            if verbose:
                logger.info(f"Worker {i+1} completed in {result.execution_time:.1f}s")

            # Parse for factors regardless of return code (ECM exits non-zero when factor found)
            if result.stdout:
                from optimized_parsing import parse_ecm_output_multiple
                worker_factors = parse_ecm_output_multiple(result.stdout)

                if worker_factors:
                    for factor, sigma in worker_factors:
                        if factor not in all_factors_set:
                            logger.info(f"Worker {i+1} found factor: {factor}")
                            from result_processor import FactorResult
                            factors_found.append(FactorResult(factor=factor, sigma=sigma, method="ecm"))
                            all_factors_set.add(factor)
                        else:
                            logger.debug(f"Worker {i+1} found duplicate factor: {factor}")
                elif not result.success:
                    # Only report as failed if no factor AND actually failed
                    logger.warning(f"Worker {i+1} failed: {result.error}")
            elif not result.success:
                logger.warning(f"Worker {i+1} failed: {result.error}")

        # Return all factors found, or None if none
        if factors_found:
            if len(factors_found) > 1:
                logger.info(f"Stage 2 found {len(factors_found)} unique factors: {[f.factor for f in factors_found]}")
            return factors_found  # Return list of all factors
        else:
            return None

        # Cleanup chunk files
        for chunk_file in residue_chunks:
            try:
                chunk_file.unlink()
                # Clean up chunk directory if it's temporary
                if chunk_file.parent.name.startswith('ecm_chunks_'):
                    try:
                        chunk_file.parent.rmdir()
                    except:
                        pass
            except:
                pass

        return factor_found

    def _split_residue_file(self, residue_file: Path, num_chunks: int) -> List[Path]:
        """Split residue file into chunks for parallel processing."""
        import tempfile
        import os
        import logging

        try:
            with open(residue_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                return []

            # Create unique temporary directory for chunk files
            chunk_dir = tempfile.mkdtemp(prefix="ecm_chunks_")

            # Each residue typically spans multiple lines, but we'll split by line count
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

                # Create chunk file
                chunk_file = Path(chunk_dir) / f"residues_chunk_{os.getpid()}_{i+1}.txt"
                with open(chunk_file, 'w') as f:
                    f.writelines(lines[start_idx:end_idx])

                if chunk_file.stat().st_size > 0:  # Only add non-empty chunks
                    chunks.append(chunk_file)

            logging.getLogger(__name__).info(f"Split residue file into {len(chunks)} chunks in {chunk_dir}")
            return chunks

        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to split residue file: {e}")
            return []

    def _extract_composite_from_residue_file(self, residue_path: Path) -> str:
        """Extract composite from residue file."""
        try:
            with open(residue_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    import re
                    match = re.search(r'N=(\d+)', first_line)
                    if match:
                        return match.group(1)
        except:
            pass
        return "unknown"

    def _extract_curve_count_from_residue_file(self, residue_path: Path) -> int:
        """Extract curve count from residue file."""
        try:
            with open(residue_path, 'r') as f:
                count = 0
                for line in f:
                    if line.strip().startswith('METHOD=ECM'):
                        count += 1
                return count
        except:
            pass
        return 0

    def _extract_b1_from_residue_file(self, residue_path: Path) -> Optional[int]:
        """Extract B1 from residue file."""
        try:
            with open(residue_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    import re
                    match = re.search(r'B1=(\d+)', first_line)
                    if match:
                        return int(match.group(1))
        except:
            pass
        return None
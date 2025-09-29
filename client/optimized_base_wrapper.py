#!/usr/bin/env python3
"""
Optimized base wrapper class with minimal overhead for analytical workloads.
Reduces logging, I/O, and memory overhead while maintaining functionality.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

from execution_engine import UnifiedExecutionEngine
from result_processor import ConsolidatedResultProcessor
from unified_ecm_executor import UnifiedECMExecutor, ECMExecutionParams

class OptimizedBaseWrapper:
    """
    High-performance base wrapper optimized for minimal overhead.
    Consolidates functionality and eliminates redundant operations.
    """

    def __init__(self, config_path: str):
        """Initialize wrapper with optimized configuration loading."""
        self._validate_working_directory()
        self.config = self._load_config(config_path)
        self._setup_minimal_logging()

        # Initialize high-performance components
        self.client_id = self.config['client']['id']
        self.api_endpoint = self.config['api']['endpoint']
        self.execution_engine = UnifiedExecutionEngine(self.config)
        self.result_processor = ConsolidatedResultProcessor(self.config)

    @lru_cache(maxsize=1)
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and cache configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_working_directory(self):
        """Lightweight directory validation (only if needed)."""
        if not Path('client.yaml').exists():
            print("⚠️  Warning: Run from client/ directory for optimal operation")

    def _setup_minimal_logging(self):
        """Set up logging optimized for performance but with verbose support."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')

        # Always set up console logging for user feedback
        log_file = self.config.get('logging', {}).get('file')
        handlers = [logging.StreamHandler()]

        if log_file:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file_path))

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Override any existing config
        )

        self.logger = logging.getLogger(__name__)

    def submit_result(self, results, project: Optional[str] = None,
                     program: str = "unknown") -> bool:
        """
        High-performance result submission using consolidated processor.

        Args:
            results: Can be ProcessingResult or legacy dict format
            project: Project name
            program: Program name

        Returns:
            True if submission successful
        """
        # Handle legacy dict format for backward compatibility
        if isinstance(results, dict):
            from result_processor import ProcessingResult, FactorResult

            processing_result = ProcessingResult(
                composite=results['composite'],
                method=results.get('method', 'ecm'),
                program=program,
                success=results.get('success', True),
                execution_time=results.get('execution_time', 0),
                curves_requested=results.get('curves_requested', 0),
                curves_completed=results.get('curves_completed', 0),
                b1=results.get('b1'),
                b2=results.get('b2'),
                raw_output=results.get('raw_output', '')
            )

            # Handle factors
            if results.get('factor_found'):
                processing_result.factors_found = [FactorResult(
                    factor=results['factor_found'],
                    sigma=results.get('sigma'),
                    method=processing_result.method
                )]
                processing_result.primary_factor = results['factor_found']
            elif results.get('factors_found'):
                processing_result.factors_found = [
                    FactorResult(factor=f, method=processing_result.method)
                    for f in results['factors_found']
                ]
                processing_result.primary_factor = results['factors_found'][0]

            results = processing_result

        # Use consolidated result processor
        return self.result_processor.submit_results(results, project)

    @lru_cache(maxsize=32)
    def get_program_version(self, program: str) -> str:
        """Get program version with caching."""
        # This would be implemented by subclasses
        return "unknown"

class OptimizedECMWrapper(OptimizedBaseWrapper):
    """
    High-performance ECM wrapper using unified execution.
    Replaces all the complex ECM execution modes with a single optimized path.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.ecm_executor = UnifiedECMExecutor(self.config)

    def run_ecm(self, composite: str, b1: int, b2: Optional[int] = None,
                curves: int = 100, sigma: Optional[int] = None,
                use_gpu: bool = False, gpu_device: Optional[int] = None,
                gpu_curves: Optional[int] = None, verbose: bool = False,
                method: str = "ecm", continue_after_factor: bool = False) -> Dict[str, Any]:
        """
        Unified ECM execution (replaces multiple run_ecm_* methods).
        Automatically chooses optimal execution strategy.
        """
        params = ECMExecutionParams(
            composite=composite,
            b1=b1,
            b2=b2,
            curves=curves,
            method=method,
            sigma=sigma,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            verbose=verbose,
            continue_after_factor=continue_after_factor
        )

        result = self.ecm_executor.execute(params)

        # Convert back to legacy dict format for compatibility
        return self._convert_to_legacy_format(result)

    def run_ecm_multiprocess(self, composite: str, b1: int, b2: Optional[int] = None,
                            curves: int = 100, workers: int = 4, verbose: bool = False,
                            method: str = "ecm", continue_after_factor: bool = False) -> Dict[str, Any]:
        """Multiprocess ECM execution."""
        params = ECMExecutionParams(
            composite=composite,
            b1=b1,
            b2=b2,
            curves=curves,
            method=method,
            workers=workers,
            verbose=verbose,
            continue_after_factor=continue_after_factor
        )

        result = self.ecm_executor.execute(params)
        return self._convert_to_legacy_format(result)

    def run_ecm_two_stage(self, composite: str, b1: int, b2: Optional[int] = None,
                         curves: int = 100, use_gpu: bool = True,
                         stage2_workers: int = 4, verbose: bool = False,
                         save_residues: Optional[str] = None,
                         resume_residues: Optional[str] = None,
                         gpu_device: Optional[int] = None,
                         gpu_curves: Optional[int] = None,
                         continue_after_factor: bool = False,
                         progress_interval: int = 0) -> Dict[str, Any]:
        """Two-stage ECM execution."""
        params = ECMExecutionParams(
            composite=composite,
            b1=b1,
            b2=b2,
            curves=curves,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            stage2_workers=stage2_workers,
            verbose=verbose,
            two_stage=True,
            save_residues=save_residues,
            resume_residues=resume_residues,
            continue_after_factor=continue_after_factor,
            progress_interval=progress_interval
        )

        result = self.ecm_executor.execute(params)
        return self._convert_to_legacy_format(result)

    def run_stage2_only(self, residue_file: str, b1: int, b2: int,
                       stage2_workers: int = 4, verbose: bool = False,
                       continue_after_factor: bool = False,
                       progress_interval: int = 0) -> Dict[str, Any]:
        """Stage 2 only execution."""
        params = ECMExecutionParams(
            composite="",  # Will be extracted from residue file
            b1=b1,
            b2=b2,
            stage2_workers=stage2_workers,
            verbose=verbose,
            continue_after_factor=continue_after_factor,
            progress_interval=progress_interval,
            resume_residues=residue_file
        )

        result = self.ecm_executor.execute(params)
        return self._convert_to_legacy_format(result)

    def _convert_to_legacy_format(self, result) -> Dict[str, Any]:
        """Convert ProcessingResult to legacy dict format for compatibility."""
        return {
            'composite': result.composite,
            'method': result.method,
            'success': result.success,
            'execution_time': result.execution_time,
            'curves_requested': result.curves_requested,
            'curves_completed': result.curves_completed,
            'b1': result.b1,
            'b2': result.b2,
            'factor_found': result.primary_factor,
            'factors_found': [f.factor for f in result.factors_found],
            'raw_output': result.raw_output,
            'error': result.error
        }

    @lru_cache(maxsize=1)
    def get_program_version(self, program: str) -> str:
        """Get GMP-ECM version with caching."""
        try:
            import subprocess
            result = subprocess.run(
                [self.config['programs']['gmp_ecm']['path'], '-h'],
                capture_output=True,
                text=True,
                timeout=5
            )
            from optimized_parsing import extract_program_version
            return extract_program_version(result.stdout, 'ecm')
        except:
            return "unknown"

class OptimizedYAFUWrapper(OptimizedBaseWrapper):
    """
    High-performance YAFU wrapper using unified infrastructure.
    Simplified and optimized for minimal overhead.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)

    def run_yafu_ecm(self, composite: str, b1: int, b2: Optional[int] = None,
                     curves: int = 100, method: str = "ecm"):
        """Optimized YAFU ECM execution."""
        from execution_engine import ExecutionParams
        from optimized_parsing import parse_yafu_ecm_output, Timeouts

        # Build command
        yafu_path = self.config['programs']['yafu']['path']
        method_input = f"{method}({composite})"
        cmd = [yafu_path, method_input]

        # Add method-specific parameters
        if method == "pm1":
            cmd.extend(['-B1pm1', str(b1)])
            if b2:
                cmd.extend(['-B2pm1', str(b2)])
        elif method == "pp1":
            cmd.extend(['-B1pp1', str(b1)])
            if b2:
                cmd.extend(['-B2pp1', str(b2)])
        else:  # ecm
            cmd.extend(['-B1ecm', str(b1)])
            if b2:
                cmd.extend(['-B2ecm', str(b2)])
            cmd.extend(['-curves', str(curves)])

        # Execute
        exec_params = ExecutionParams(cmd=cmd, timeout=Timeouts.YAFU_ECM)
        exec_result = self.execution_engine.execute_single(exec_params)

        # Process results
        result = self.result_processor.process_execution_result(
            exec_result, composite, method, f"yafu-{method}",
            curves=curves, b1=b1, b2=b2
        )

        return self._convert_to_legacy_format(result)

    def run_yafu_auto(self, composite: str, method: Optional[str] = None):
        """Optimized YAFU auto execution."""
        from execution_engine import ExecutionParams
        from optimized_parsing import parse_yafu_auto_factors, Timeouts

        # Build command
        yafu_path = self.config['programs']['yafu']['path']
        auto_input = f"factor({composite})"
        cmd = [yafu_path, auto_input]

        if method:
            cmd.extend(['-method', method])

        # Execute
        exec_params = ExecutionParams(cmd=cmd, timeout=Timeouts.YAFU_AUTO)
        exec_result = self.execution_engine.execute_single(exec_params)

        # Process results
        result = self.result_processor.process_execution_result(
            exec_result, composite, method or 'auto', f"yafu-{method or 'auto'}"
        )

        return self._convert_to_legacy_format(result)

    def _convert_to_legacy_format(self, result):
        """Convert ProcessingResult to legacy dict format."""
        return {
            'composite': result.composite,
            'method': result.method,
            'success': result.success,
            'execution_time': result.execution_time,
            'factor_found': result.primary_factor,
            'factors_found': [f.factor for f in result.factors_found],
            'raw_output': result.raw_output
        }

    @lru_cache(maxsize=1)
    def get_program_version(self, program: str) -> str:
        """Get YAFU version with caching."""
        try:
            import subprocess
            result = subprocess.run(
                [self.config['programs']['yafu']['path'], '-h'],
                capture_output=True,
                text=True,
                timeout=5
            )
            from optimized_parsing import extract_program_version
            return extract_program_version(result.stdout, 'yafu')
        except:
            return "unknown"
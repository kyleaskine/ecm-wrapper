"""
Configuration classes for ECM wrapper operations.

This module defines dataclasses that encapsulate ECM execution parameters,
reducing function argument counts and improving code maintainability.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TypedDict
from pathlib import Path


class ECMConfigValidation:
    """Mixin providing shared validation methods for ECM configs."""

    def _validate_composite(self, composite: str) -> None:
        """Validate composite is set."""
        if not composite:
            raise ValueError("composite is required and cannot be empty")

    def _validate_b1(self, b1: int) -> None:
        """Validate B1 bound."""
        if b1 <= 0:
            raise ValueError(f"B1 must be positive, got {b1}")

    def _validate_method(self, method: str) -> None:
        """Validate method selection."""
        if method not in ['ecm', 'pm1', 'pp1']:
            raise ValueError(f"Method must be 'ecm', 'pm1', or 'pp1', got {method}")

    def _validate_parametrization(self, param: int) -> None:
        """Validate parametrization."""
        if param not in [0, 1, 2, 3]:
            raise ValueError(f"Parametrization must be 0-3, got {param}")


class ECMPrimitiveResult(TypedDict):
    """Return type of _execute_ecm_primitive() and its helper wrappers."""
    success: bool
    factors: List[str]
    sigmas: List[Optional[str]]
    curves_completed: int
    raw_output: str
    parametrization: int
    exit_code: int
    interrupted: bool


@dataclass
class ECMConfig(ECMConfigValidation):
    """Configuration for standard ECM execution."""

    composite: str
    b1: int
    b2: Optional[int] = None
    curves: int = 1
    sigma: Optional[int] = None
    parametrization: int = 1  # Default to 1 (CPU/Montgomery), use 3 for GPU
    threads: int = 1
    verbose: bool = False
    save_residues: Optional[str] = None

    # GPU support
    use_gpu: bool = False
    gpu_device: Optional[int] = None
    gpu_curves: Optional[int] = None

    # Method and progress
    method: str = "ecm"  # 'ecm', 'pm1', 'pp1'
    progress_interval: int = 0
    maxmem: Optional[int] = None  # Maximum memory in MB for stage 2 (-maxmem)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_composite(self.composite)
        self._validate_b1(self.b1)
        self._validate_parametrization(self.parametrization)
        self._validate_method(self.method)
        if self.curves <= 0:
            raise ValueError(f"Curves must be positive, got {self.curves}")

        # Auto-set parametrization for GPU mode if not explicitly set
        if self.use_gpu and self.parametrization == 1:
            self.parametrization = 3  # GPU uses twisted Edwards curves


@dataclass
class TwoStageConfig(ECMConfigValidation):
    """Configuration for two-stage ECM pipeline (GPU + CPU)."""

    composite: str
    b1: int
    b2: Optional[int] = None
    stage1_curves: int = 100
    stage2_curves_per_residue: int = 1000
    stage1_device: str = "GPU"
    stage2_device: str = "CPU"
    stage1_parametrization: int = 3
    stage2_parametrization: int = 1
    threads: int = 1
    verbose: bool = False
    save_residues: Optional[str] = None
    resume_file: Optional[str] = None

    # GPU support
    gpu_device: Optional[int] = None
    gpu_curves: Optional[int] = None

    # Execution control
    continue_after_factor: bool = False
    progress_interval: int = 0  # Show progress every N curves (0 = disabled)
    pin_threads: bool = False  # Pin each stage 2 worker to its own CPU core (Linux only)

    # API submission
    project: Optional[str] = None
    no_submit: bool = False

    def __post_init__(self):
        """Validate configuration."""
        self._validate_composite(self.composite)
        self._validate_b1(self.b1)
        if self.stage1_curves <= 0:
            raise ValueError(f"Stage1 curves must be positive")
        if self.stage2_curves_per_residue <= 0:
            raise ValueError(f"Stage2 curves must be positive")


@dataclass
class MultiprocessConfig(ECMConfigValidation):
    """Configuration for multiprocess ECM execution."""

    composite: str
    b1: int
    b2: Optional[int] = None
    total_curves: int = 1000
    curves_per_process: int = 100
    num_processes: Optional[int] = None  # None = auto-detect CPU count
    parametrization: int = 1  # Default to 1 (CPU/Montgomery)
    method: str = "ecm"  # 'ecm', 'pm1', 'pp1'
    verbose: bool = False

    # Execution control
    continue_after_factor: bool = False
    progress_interval: int = 0  # Show progress every N curves (0 = disabled)
    pin_threads: bool = False  # Pin each worker process to its own CPU core (Linux only)

    def __post_init__(self):
        """Validate and auto-configure."""
        self._validate_composite(self.composite)
        self._validate_b1(self.b1)
        self._validate_method(self.method)
        if self.total_curves <= 0:
            raise ValueError(f"Total curves must be positive")
        if self.curves_per_process <= 0:
            raise ValueError(f"Curves per process must be positive")

        # Auto-detect number of processes if not specified
        if self.num_processes is None:
            import multiprocessing
            self.num_processes = multiprocessing.cpu_count()


@dataclass
class TLevelConfig(ECMConfigValidation):
    """Configuration for t-level targeting ECM."""

    composite: str
    target_t_level: float
    start_t_level: float = 0.0  # Starting t-level (for continuing after factor found)
    b1_strategy: str = "optimal"  # 'optimal', 'conservative', 'aggressive'
    parametrization: int = 1  # Default to 1 (CPU/Montgomery), use 3 for GPU/two-stage
    threads: int = 1
    verbose: bool = False

    # Execution modes
    workers: int = 1  # Alias for threads (multiprocess support)
    use_two_stage: bool = False
    progress_interval: int = 0  # Show progress every N curves (0 = disabled)
    max_batch_curves: Optional[int] = None  # Max curves per GPU batch (enables chunking for pipelined mode)
    b2_multiplier: float = 500.0  # B2 = B1 * multiplier for two-stage mode (default 500)
    b2_dictionary: Optional[Dict[int, int]] = None  # B1 → B2 lookup (overrides b2_multiplier)

    # GPU support
    gpu_device: Optional[int] = None
    gpu_curves: Optional[int] = None

    # Thread pinning
    pin_threads: bool = False  # Pin each CPU worker to its own core (Linux only)

    # API submission
    project: Optional[str] = None
    no_submit: bool = False
    auto_adjust_target: bool = False  # Adjust target when factors found
    work_id: Optional[str] = None  # For auto-work mode batch submissions (server work ID)

    def __post_init__(self):
        """Validate configuration."""
        self._validate_composite(self.composite)
        self._validate_parametrization(self.parametrization)
        if self.target_t_level <= 0:
            raise ValueError(f"Target t-level must be positive")
        if self.b1_strategy not in ['optimal', 'conservative', 'aggressive']:
            raise ValueError(f"Unknown B1 strategy: {self.b1_strategy}")

        # Sync workers and threads (they're the same thing)
        if self.workers != 1:
            self.threads = self.workers
        elif self.threads != 1:
            self.workers = self.threads

        # Two-stage mode runs stage 1 on GPU (twisted Edwards), so the
        # effective parametrization is 3. If the caller didn't explicitly
        # override the default (1 = Montgomery/CPU), bump it.
        if self.use_two_stage and self.parametrization == 1:
            self.parametrization = 3

    def get_b2_for_b1(self, b1: int) -> int:
        """Resolve B2 for a given B1: dictionary lookup, then multiplier fallback."""
        if self.b2_dictionary and b1 in self.b2_dictionary:
            return self.b2_dictionary[b1]
        return int(b1 * self.b2_multiplier)


@dataclass
class FactorResult:
    """Result from ECM factorization attempt."""

    factors: List[str] = field(default_factory=list)
    sigmas: List[Optional[str]] = field(default_factory=list)
    curves_run: int = 0
    execution_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    raw_output: Optional[str] = None

    # Execution metadata (optional, for debugging)
    parametrization: Optional[int] = None
    exit_code: Optional[int] = None
    interrupted: bool = False

    # T-level tracking (for progressive ECM)
    t_level_achieved: float = 0.0  # T-level reached during execution
    curve_summary: List[Dict[str, Any]] = field(default_factory=list)  # Detailed curve execution summary

    # True if a run that submits internally (t-level mode submits per B1 batch)
    # had at least one submission fail and get queued. The curves are done, so
    # the caller must not treat this as an execution error - but it must not
    # report success either, or the work assignment gets completed with no
    # progress recorded and the composite is handed straight back out.
    submission_failed: bool = False

    def add_factor(self, factor: str, sigma: Optional[str] = None):
        """Add a discovered factor with its sigma."""
        self.factors.append(factor)
        self.sigmas.append(sigma)
        self.success = True

    def print_curve_summary(self, show_parametrization: bool = False) -> None:
        """
        Print a summary table of curves executed during t-level runs.
        Groups identical B1/B2/param combinations together.

        Args:
            show_parametrization: If True, include parametrization column
        """
        if not self.curve_summary:
            return

        # Group curves by (B1, B2, param)
        grouped: dict = {}
        for step in self.curve_summary:
            b1 = step['b1']
            b2 = step.get('b2')
            param = step.get('parametrization', 1)
            curves = step['curves']

            key = (b1, b2, param)
            if key in grouped:
                grouped[key] += curves
            else:
                grouped[key] = curves

        print("\nCurve Execution Summary:")
        print("=" * 80)

        # Calculate column widths
        if show_parametrization:
            header = f"{'B1':<15} {'B2':<15} {'Curves':<10} {'Param':<6}"
            print(header)
            print("-" * 80)
        else:
            header = f"{'B1':<15} {'B2':<15} {'Curves':<10}"
            print(header)
            print("-" * 80)

        total_curves = 0
        # Sort by B1 for consistent display
        for (b1, b2, param), curves in sorted(grouped.items(), key=lambda x: x[0][0]):
            total_curves += curves

            # Format B2 display
            if b2 is None:
                b2_str = "GMP default"
            elif b2 == 0:
                b2_str = "0 (stage1)"
            else:
                b2_str = f"{b2:,}"

            # Format B1 display
            b1_str = f"{b1:,}"

            if show_parametrization:
                param_str = f"p={param}"
                print(f"{b1_str:<15} {b2_str:<15} {curves:<10} {param_str:<6}")
            else:
                print(f"{b1_str:<15} {b2_str:<15} {curves:<10}")

        print("-" * 80)
        print(f"{'Total:':<15} {'':<15} {total_curves:<10}")
        if self.t_level_achieved > 0:
            print(f"\nT-level achieved: {self.t_level_achieved:.2f}")
        print("=" * 80)

    @property
    def factor_sigma_pairs(self) -> List[tuple]:
        """Get list of (factor, sigma) tuples."""
        return list(zip(self.factors, self.sigmas))

    @classmethod
    def from_primitive_result(cls, prim_result: ECMPrimitiveResult, execution_time: float = 0.0) -> 'FactorResult':
        """
        Create FactorResult from _execute_ecm_primitive() dict output.

        Args:
            prim_result: Dict from _execute_ecm_primitive with keys:
                - success, factors, sigmas, curves_completed, raw_output,
                - parametrization, exit_code, interrupted
            execution_time: Execution time in seconds

        Returns:
            FactorResult instance
        """
        return cls(
            factors=prim_result.get('factors', []),
            sigmas=prim_result.get('sigmas', []),
            curves_run=prim_result.get('curves_completed', 0),
            execution_time=execution_time,
            success=prim_result.get('success', False),
            raw_output=prim_result.get('raw_output'),
            parametrization=prim_result.get('parametrization'),
            exit_code=prim_result.get('exit_code'),
            interrupted=prim_result.get('interrupted', False)
        )

    def to_dict(self, composite: Optional[str] = None, method: str = 'ecm') -> Dict[str, Any]:
        """
        Convert FactorResult to dictionary format (for backward compatibility).

        Args:
            composite: Optional composite number to include in result
            method: Method used ('ecm', 'pm1', 'pp1')

        Returns:
            Dictionary with v1-compatible format
        """
        result: Dict[str, Any] = {
            'success': self.success,
            'factors_found': self.factors,
            'curves_completed': self.curves_run,
            'execution_time': self.execution_time,
            'raw_output': self.raw_output,
            'method': method
        }
        if composite:
            result['composite'] = composite

        # Include factor_sigmas mapping for API submission
        if self.factors and self.sigmas:
            result['factor_sigmas'] = {
                factor: sigma for factor, sigma in zip(self.factors, self.sigmas)
                if sigma is not None
            }
            # Also include first sigma as main sigma for backward compat
            if self.sigmas[0] is not None:
                result['sigma'] = self.sigmas[0]

        return result


@dataclass
class ExecutionBatch:
    """
    Parameters for a batch of ECM work.

    Defines the interface between planning (what to run) and execution (how to run it).
    Used by CompositeExecutionEngine as the common input format.
    """

    composite: str
    b1: int
    b2: Optional[int] = None
    curves: int = 1
    method: str = "ecm"
    verbose: bool = False
    progress_interval: int = 0


@dataclass
class BatchResult:
    """
    Result from executing a batch of ECM work.

    Engine's output format. Can be converted to FactorResult for API compatibility.
    """

    factors: List[str] = field(default_factory=list)
    sigmas: List[Optional[str]] = field(default_factory=list)
    curves_run: int = 0
    execution_time: float = 0.0
    success: bool = False
    raw_output: Optional[str] = None
    interrupted: bool = False
    submission_failed: bool = False  # A per-batch submission failed and was queued

    def to_factor_result(self) -> FactorResult:
        """Convert to FactorResult for API compatibility."""
        result = FactorResult()
        for factor, sigma in zip(self.factors, self.sigmas):
            result.add_factor(factor, sigma)
        result.curves_run = self.curves_run
        result.execution_time = self.execution_time
        result.success = self.success
        result.raw_output = self.raw_output
        result.interrupted = self.interrupted
        result.submission_failed = self.submission_failed
        return result

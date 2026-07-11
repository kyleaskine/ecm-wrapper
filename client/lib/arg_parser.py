#!/usr/bin/env python3
"""
Shared argument parsing logic for ECM and YAFU wrappers.
"""
import argparse
import sys
import multiprocessing
from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .typed_config import AppConfig
    from .work_args import WorkArgs

# Helpers below accept either an argparse.Namespace (manual mode, ecm_wrapper.py)
# or a typed WorkArgs (auto-work mode). Both expose the same attribute names
# for the fields these helpers read.
ArgsLike = Union[argparse.Namespace, "WorkArgs"]


def parse_int_with_scientific(value: str) -> int:
    """
    Parse integer from string, supporting scientific notation.

    Examples:
        "1000000" -> 1000000
        "1e6" -> 1000000
        "26e7" -> 260000000
        "4e11" -> 400000000000
        "-1" -> -1 (special: GMP-ECM default for B2)

    Args:
        value: String representation of number

    Returns:
        Integer value

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed
    """
    try:
        # Convert through float to handle scientific notation, then to int
        result = int(float(value))
        # Allow -1 as a special sentinel value (GMP-ECM default for B2)
        if result < -1:
            raise argparse.ArgumentTypeError(f"Value must be -1 or positive: {value}")
        return result
    except (ValueError, OverflowError) as e:
        raise argparse.ArgumentTypeError(f"Invalid integer or scientific notation: {value}") from e


def load_b2_dictionary(filepath: str) -> tuple[Dict[int, int], Dict[int, int]]:
    """
    Load a B1 → B2 mapping (and optional B1 → k mapping) from a dictionary file.

    File format: one entry per line, whitespace-separated columns:
        B1  B2  [k]  [# comment]
    Lines starting with #, ', or -- are comments. Supports scientific notation
    in B1 and B2. The k column is optional; when present and positive, it is
    returned in the second dict. Comment markers in the k column slot are
    treated as "no k value".

    Returns:
        Tuple of (b2_dict, k_dict). b2_dict maps every parsed B1 to its B2;
        k_dict maps only those B1 values that had a positive k.
    """
    b2_dict: Dict[int, int] = {}
    k_dict: Dict[int, int] = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith("'") or line.startswith('--'):
                    continue
                entries = line.split()
                if len(entries) < 2:
                    print(f"Warning: Skipping malformed B2 dictionary line: {line}")
                    continue
                try:
                    key = int(float(entries[0]))
                    value = int(float(entries[1]))
                except ValueError:
                    print(f"Warning: Skipping invalid B2 dictionary entry: {line}")
                    continue

                b2_dict[key] = value

                # Optional 3rd column: k value (skip if it's a comment marker)
                if len(entries) >= 3:
                    k_entry = entries[2]
                    if not (k_entry.startswith('#') or k_entry.startswith("'") or k_entry.startswith('--')):
                        try:
                            k = int(k_entry)
                            if k > 0:
                                k_dict[key] = k
                        except ValueError:
                            pass  # Silent skip — k is optional
    except OSError as e:
        print(f"Warning: Could not load B2 dictionary '{filepath}': {e}")
    return b2_dict, k_dict


def _add_core_ecm_params(parser: argparse.ArgumentParser) -> None:
    """Add core ECM parameters shared by ECM and client parsers: B1, B2, method, etc.

    Note: --curves is NOT included here because the ECM parser needs the -c alias
    while the client parser does not. Each parser adds --curves separately.
    """
    parser.add_argument('--b1', type=parse_int_with_scientific,
                       help='B1 bound (supports scientific notation, e.g., 26e7)')
    parser.add_argument('--b2', type=parse_int_with_scientific,
                       help='B2 bound (supports scientific notation, e.g., 4e11). Use -1 for GMP-ECM default')
    parser.add_argument('--b2-multiplier', type=float,
                       help='Dynamic B2 calculation: B2 = B1 * multiplier (e.g., 1000 for B2=1000*B1). Overridden by explicit --b2')
    parser.add_argument('--b2-dictionary', type=str, default=None,
                       help='File which maps B1 to B2 values (one entry per line, separated by space)')
    parser.add_argument('--max-batch', type=int,
                       help='Max curves per GPU batch in two-stage t-level mode (enables chunking for earlier factor discovery)')
    parser.add_argument('--method', choices=['ecm', 'pm1', 'pp1'], default='ecm',
                       help='Factorization method (ECM, P-1, P+1)')


def _add_gpu_options(parser: argparse.ArgumentParser) -> None:
    """Add GPU-related options."""
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--gpu-device', type=int, help='GPU device number')
    parser.add_argument('--gpu-curves', type=int, help='Number of curves per GPU batch')


def _add_sigma_and_param(parser: argparse.ArgumentParser) -> None:
    """Add sigma/parametrization options."""
    parser.add_argument('--sigma', type=str, help='Sigma value (integer or parametrization:value)')
    parser.add_argument('--param', type=int, choices=[0, 1, 2, 3], help='ECM parametrization (0-3)')


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    """Add execution mode options: multiprocess, two-stage, workers."""
    parser.add_argument('--multiprocess', action='store_true',
                       help='Use multi-process mode: parallel full ECM cycles (CPU-optimized)')
    parser.add_argument('--two-stage', action='store_true',
                       help='Use two-stage mode: GPU stage 1 + multi-threaded CPU stage 2')
    parser.add_argument('--workers', type=int,
                       help='Number of parallel workers (processes for multiprocess, threads for stage2)')
    parser.add_argument('--pin-threads', action='store_true',
                       help='Pin each worker to its own CPU core (Linux only). Respects existing '
                            'affinity restrictions; errors if workers exceed available CPU slots.')


def _add_work_filter_options(parser: argparse.ArgumentParser) -> None:
    """Add server work filtering: work-count, min/max-digits, priority, work-type."""
    parser.add_argument('--work-count', type=int,
                       help='Number of work assignments to complete before exiting (default: unlimited)')
    parser.add_argument('--exit-on-no-work', action='store_true',
                       help='Exit immediately when no work is available instead of waiting 30 seconds')
    parser.add_argument('--min-digits', type=int, help='Minimum composite digit length')
    parser.add_argument('--max-digits', type=int, help='Maximum composite digit length')
    parser.add_argument('--priority', type=int, help='Minimum priority level')
    parser.add_argument('--work-type', choices=['standard', 'progressive'], default='standard',
                       help='Work assignment strategy: standard (smallest first) or progressive (least ECM done first)')


def _add_behavior_options(parser: argparse.ArgumentParser) -> None:
    """Add behavior options: verbose, progress-interval, continue-after-factor, maxmem."""
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--progress-interval', type=int, default=0,
                       help='Show progress updates every N completed curves '
                            '(0 = disabled; --two-stage/--stage2-only default to 50 '
                            'unless -v is set without --progress-interval)')
    parser.add_argument('--continue-after-factor', action='store_true',
                       help='Continue processing all curves even after finding a factor')
    parser.add_argument('--maxmem', type=int,
                       help='Maximum memory in MB for GMP-ECM stage 2 (-maxmem flag)')


def create_ecm_parser() -> argparse.ArgumentParser:
    """Create argument parser for ECM wrapper (local/manual factorization)."""
    parser = argparse.ArgumentParser(description='ECM Wrapper Client')

    # Configuration
    parser.add_argument('--config', default='client.yaml', help='Config file path')

    # Core parameters
    parser.add_argument('--composite', '-n', help='Number to factor (not required in --auto-work mode)')
    _add_core_ecm_params(parser)
    parser.add_argument('--curves', '-c', type=int, help='Number of curves')

    # ECM-specific t-level (nargs='?' with sentinel for auto-calc)
    parser.add_argument('--tlevel', '-t', type=float, nargs='?', const=-1.0,
                       help='Target t-level. If specified without a value, auto-calculates as 4/13 of digit length and runs progressively until factored.')
    parser.add_argument('--start-tlevel', type=float, help='Starting t-level (for resuming, requires --tlevel)')
    parser.add_argument('--project', '-p', help='Project name')
    parser.add_argument('--submit', action='store_true', help='Submit results to API')

    # Auto-work mode
    parser.add_argument('--auto-work', action='store_true',
                       help='Continuously request and process work assignments from server (uses server t-levels unless --b1/--b2 or --tlevel specified)')
    _add_work_filter_options(parser)

    # Decoupled two-stage mode (stage 1 and stage 2 run separately)
    parser.add_argument('--stage1-only', action='store_true',
                       help='Run stage 1 only, submit results and upload residue file to server (GPU producer mode)')
    parser.add_argument('--upload', action='store_true',
                       help='Upload residue file to server after stage 1 (for --stage1-only mode)')

    _add_gpu_options(parser)
    _add_sigma_and_param(parser)
    _add_execution_options(parser)
    _add_behavior_options(parser)

    # Residue file handling
    parser.add_argument('--save-residues', type=str, help='Save stage 1 residues with specified filename in configured residue_dir')
    parser.add_argument('--stage2-only', type=str, help='Run stage 2 only on residue file path')

    return parser


def create_client_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for ecm_client.py (server-coordinated modes).

    This parser is for server-coordinated work where composites and t-levels
    come from the server. For local/manual factorization, use create_ecm_parser().
    """
    parser = argparse.ArgumentParser(
        description='ECM Client - Server-coordinated factorization work',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-work with server defaults
  python3 ecm_client.py

  # Target a specific composite (server provides t-level info)
  python3 ecm_client.py --composite "123456789..."

  # Target composite with multiprocess
  python3 ecm_client.py --composite "123456789..." --multiprocess --workers 8

  # Process 10 work items with client-specified B1/B2
  python3 ecm_client.py --work-count 10 --b1 50000 --b2 5000000 --curves 100

  # Stage 1 only - upload residues to server
  python3 ecm_client.py --stage1-only --b1 110000000 --curves 3000

  # Stage 2 only - download and process residues
  python3 ecm_client.py --stage2-only --b2 11000000000000 --workers 8
"""
    )

    # Composite targeting
    parser.add_argument('--composite', type=str,
                       help='Target a specific composite (queries server for t-level status)')

    # Work filtering
    _add_work_filter_options(parser)
    parser.add_argument('--min-target-tlevel', type=float,
                       help='Minimum target t-level (filter work by difficulty)')
    parser.add_argument('--max-target-tlevel', type=float,
                       help='Maximum target t-level (filter work by difficulty)')

    # Client-specific t-level (simple float, no auto-calc sentinel)
    parser.add_argument('--tlevel', type=float,
                       help='Target t-level (overrides server t-level)')

    # Shared ECM params, execution modes
    _add_core_ecm_params(parser)
    parser.add_argument('--curves', type=int, help='Curves per batch')
    _add_execution_options(parser)

    # Work mode selection (mutually exclusive)
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument('--adaptive', action='store_true',
                            help='Adaptive CPU mode: prioritize stage 2 residues, fall back to ECM (default if no mode specified)')
    stage_group.add_argument('--standard', action='store_true',
                            help='Standard auto-work mode: t-level or B1/B2 based execution')
    stage_group.add_argument('--pm1', action='store_true',
                            help='Run P-1 factorization (1 curve per composite)')
    stage_group.add_argument('--pp1', action='store_true',
                            help='Run P+1 factorization (3 curves per composite)')
    stage_group.add_argument('--p1', action='store_true',
                            help='Run P-1 (1 curve) + P+1 (3 curves) per composite')
    stage_group.add_argument('--stage1-only', action='store_true',
                            help='Stage 1 only (GPU producer): upload residue to server')
    stage_group.add_argument('--stage2-only', action='store_true',
                            help='Stage 2 only: download residue from server')

    # P+1 options
    parser.add_argument('--pp1-curves', type=int, default=3,
                       help='Number of P+1 curves per composite (default: 3)')

    # Stage 2 filtering (for --stage2-only mode)
    parser.add_argument('--min-b1', type=parse_int_with_scientific,
                       help='Minimum B1 filter for --stage2-only (supports scientific notation, e.g., 11e6)')
    parser.add_argument('--max-b1', type=parse_int_with_scientific,
                       help='Maximum B1 filter for --stage2-only (supports scientific notation, e.g., 26e7)')

    _add_gpu_options(parser)
    _add_sigma_and_param(parser)
    _add_behavior_options(parser)

    # API settings
    parser.add_argument('--project', type=str,
                       help='Project name for submissions')
    parser.add_argument('--no-submit', action='store_true',
                       help='Skip result submission to server')

    # Hidden: for backward compatibility, auto-work is implied
    parser.add_argument('--auto-work', action='store_true', dest='auto_work_explicit',
                       help=argparse.SUPPRESS)

    return parser


def create_yafu_parser() -> argparse.ArgumentParser:
    """Create argument parser for YAFU wrapper."""
    parser = argparse.ArgumentParser(description='YAFU Wrapper Client')

    # Configuration
    parser.add_argument('--config', default='client.yaml', help='Config file path')
    parser.add_argument('--composite', '-n', required=True, help='Number to factor')

    # Mode selection
    parser.add_argument('--mode', choices=['ecm', 'pm1', 'pp1', 'auto', 'siqs', 'nfs'],
                       default='ecm', help='Factorization mode')

    # ECM parameters
    parser.add_argument('--b1', type=parse_int_with_scientific, help='B1 bound for ECM (supports scientific notation, e.g., 26e7)')
    parser.add_argument('--b2', type=parse_int_with_scientific, help='B2 bound for ECM (supports scientific notation, e.g., 4e11)')
    parser.add_argument('--curves', '-c', type=int, default=100, help='Number of curves for ECM')

    # General parameters
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose YAFU output (stream in real-time)')

    return parser


def validate_ecm_args(args: argparse.Namespace, config: Optional['AppConfig'] = None) -> Dict[str, str]:
    """
    Validate ECM arguments and return any validation errors.

    Args:
        args: Parsed command line arguments
        config: Typed AppConfig (optional, for B2 validation)

    Returns:
        Dictionary mapping argument names to error messages
    """
    errors = {}

    # Decoupled two-stage mode validation
    if hasattr(args, 'stage1_only') and args.stage1_only:
        if hasattr(args, 'tlevel') and args.tlevel is not None:
            errors['tlevel'] = "--stage1-only not compatible with --tlevel. Use --b1/--curves instead."
        if args.b2 is not None and args.b2 != 0:
            errors['b2'] = "--stage1-only runs stage 1 only. B2 should be 0 or omitted."

        # Auto-work mode: Only B1 is required (curves will default to config)
        if hasattr(args, 'auto_work') and args.auto_work:
            if args.b1 is None:
                errors['b1'] = "--stage1-only with --auto-work requires --b1 to be specified"
        # Manual mode: composite required, B1/curves use config defaults if not specified
        else:
            if not args.composite:
                errors['composite'] = "--stage1-only without --auto-work requires --composite to be specified"

    # Auto-work mode validation (check first, before other modes)
    if hasattr(args, 'auto_work') and args.auto_work:
        has_b1_b2 = args.b1 is not None and args.b2 is not None
        has_tlevel = hasattr(args, 'tlevel') and args.tlevel is not None

        # Parameters are now optional - can use server's t-level data
        # Three modes: server t-level (default), client B1/B2, or client t-level

        # Two-stage only compatible with B1/B2 mode (not t-level mode)
        if args.two_stage:
            if has_tlevel:
                errors['two_stage'] = "Two-stage mode not compatible with --tlevel. Use --b1/--b2 instead."
            elif not has_b1_b2:
                errors['two_stage'] = "Two-stage mode requires --b1 and --b2 to be specified"
            # Warn if using two-stage with curves > 1 (GPU batches automatically)
            if args.curves and args.curves > 1:
                errors['curves'] = "Two-stage mode: GPU batches curves automatically. Use --curves 1 or omit."

        # Multiprocess is allowed (works with t-level mode)
        # stage2-only not supported in auto-work
        if args.stage2_only:
            errors['stage2_only'] = "Auto-work mode not compatible with --stage2-only"

        # Composite should not be specified in auto-work mode
        if args.composite:
            errors['composite'] = "Auto-work mode gets composites from server. Do not specify --composite."

        # Return early to avoid conflicting validations
        return errors

    # Filter options only valid in auto-work mode
    if hasattr(args, 'work_count') and args.work_count is not None and not args.auto_work:
        errors['work_count'] = "--work-count only valid in --auto-work mode"
    if hasattr(args, 'min_target_tlevel') and args.min_target_tlevel is not None and not args.auto_work:
        errors['min_target_tlevel'] = "--min-target-tlevel only valid in --auto-work mode"
    if hasattr(args, 'max_target_tlevel') and args.max_target_tlevel is not None and not args.auto_work:
        errors['max_target_tlevel'] = "--max-target-tlevel only valid in --auto-work mode"
    if hasattr(args, 'priority') and args.priority is not None and not args.auto_work:
        errors['priority'] = "--priority only valid in --auto-work mode"

    # T-level mode validation
    if hasattr(args, 'tlevel') and args.tlevel is not None:
        if args.curves:
            errors['curves'] = "Cannot specify both --tlevel and --curves. Choose one."

        # Validate start-tlevel (only meaningful when explicit t-level given)
        if hasattr(args, 'start_tlevel') and args.start_tlevel is not None:
            if args.start_tlevel < 0:
                errors['start_tlevel'] = "--start-tlevel must be non-negative"
            # Only check start < target when explicit t-level given (not auto mode)
            elif args.tlevel > 0 and args.start_tlevel >= args.tlevel:
                errors['start_tlevel'] = f"--start-tlevel ({args.start_tlevel}) must be less than --tlevel ({args.tlevel})"

    # Validate start-tlevel requires tlevel
    if hasattr(args, 'start_tlevel') and args.start_tlevel is not None:
        if not hasattr(args, 'tlevel') or args.tlevel is None:
            errors['start_tlevel'] = "--start-tlevel requires --tlevel to be specified"
        if not args.composite:
            errors['composite'] = "T-level mode requires composite number. Use --composite argument."
        if args.b1:
            errors['b1'] = "T-level mode automatically selects B1. Remove --b1 argument."
        if args.stage2_only:
            errors['mode'] = "T-level mode not compatible with --stage2-only mode."

    # Mode compatibility checks
    if args.multiprocess and args.two_stage:
        errors['mode'] = "Cannot use both --multiprocess and --two-stage. Choose one mode."


    # Stage 2 only mode validation
    if args.stage2_only:
        if args.composite:
            errors['composite'] = "Stage 2 only mode - composite number not required"
        if not args.b2:
            errors['b2'] = "Stage 2 only mode requires B2 bound. Use --b2 argument."

    # Two-stage mode validation
    elif args.two_stage and args.method == 'ecm':
        if not args.composite:
            errors['composite'] = "Two-stage mode requires composite number. Use --composite argument."
        # Two-stage mode requires a B2 source for Stage 2 coordination.
        # Valid B2 sources: --b2, --b2-multiplier, --b2-dictionary, --tlevel
        # (calculates B2 automatically), or config default.
        has_b2_source = (args.b2 is not None
                         or getattr(args, 'b2_multiplier', None) is not None
                         or getattr(args, 'b2_dictionary', None) is not None
                         or (hasattr(args, 'tlevel') and args.tlevel is not None))
        if not has_b2_source and config:
            _, b2_default = get_method_defaults(config, args.method)
            if not b2_default:
                errors['b2'] = "Two-stage mode requires B2 bound. Use --b2, --b2-multiplier, --b2-dictionary, or --tlevel."
        elif not has_b2_source and not config:
            errors['b2'] = "Two-stage mode requires B2 bound. Use --b2, --b2-multiplier, --b2-dictionary, or --tlevel."

    # Multiprocess mode validation
    elif args.multiprocess:
        if not args.composite:
            errors['composite'] = "Multiprocess mode requires composite number. Use --composite argument."
        if args.save_residues:
            errors['residues'] = "--save-residues not applicable in multiprocess mode."


    # Standard mode validation
    else:
        if not args.composite:
            errors['composite'] = "Standard mode requires composite number. Use --composite argument."
        if args.two_stage and args.method != 'ecm':
            errors['method'] = "Two-stage mode only available for ECM method."
        if args.save_residues:
            errors['residues'] = "Save residues option only available in two-stage mode."

    # GPU validation
    if args.gpu and args.no_gpu:
        errors['gpu'] = "Cannot specify both --gpu and --no-gpu"

    return errors


def get_workers_default(config: 'AppConfig') -> int:
    """
    Get default workers value from typed config.

    Used for both multiprocess workers and stage2 threads.
    """
    return config.programs.gmp_ecm.workers or 4


def get_max_batch_default(config: 'AppConfig') -> Optional[int]:
    """Get default max_batch value from typed config."""
    return config.programs.gmp_ecm.max_batch


def get_method_defaults(config: 'AppConfig', method: str) -> tuple[int, Optional[int]]:
    """
    Get default B1 and B2 values for the specified method.

    Args:
        config: Typed AppConfig
        method: Method name ('ecm', 'pm1', 'pp1')

    Returns:
        Tuple of (b1_default, b2_default)
    """
    gmp = config.programs.gmp_ecm

    if method == 'pm1':
        # pm1_b1 is non-Optional in dataclass with a default; treat 0 as "fall back to default_b1"
        b1_default = gmp.pm1_b1 or gmp.default_b1
        b2_default = gmp.pm1_b2 if gmp.pm1_b2 else gmp.default_b2
    elif method == 'pp1':
        b1_default = gmp.pp1_b1 or gmp.default_b1
        b2_default = gmp.pp1_b2 if gmp.pp1_b2 else gmp.default_b2
    else:  # ecm
        b1_default = gmp.default_b1
        b2_default = gmp.default_b2

    return b1_default, b2_default


def resolve_gpu_settings(args: ArgsLike, config: 'AppConfig') -> tuple[bool, Optional[int], Optional[int]]:
    """
    Resolve GPU settings from arguments and typed config.

    Returns:
        Tuple of (use_gpu, gpu_device, gpu_curves)
    """
    gmp = config.programs.gmp_ecm

    # GPU settings: command line overrides config defaults
    if args.no_gpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = gmp.gpu_enabled

    gpu_device = args.gpu_device if args.gpu_device is not None else gmp.gpu_device
    gpu_curves = args.gpu_curves if args.gpu_curves is not None else gmp.gpu_curves

    return use_gpu, gpu_device, gpu_curves


def resolve_stage2_progress_interval(args: ArgsLike) -> int:
    """Effective progress_interval for stage 2 output (--two-stage / --stage2-only).

    Stage 2 emits a "Step X took Yms" line per curve, which can be thousands per
    composite. Default to a periodic summary so the console stays readable:

    - Explicit --progress-interval N (N > 0): use N.
    - -v without --progress-interval: 0 (stream every line).
    - Neither: 50 (concise default).
    """
    pi = getattr(args, 'progress_interval', None) or 0
    if pi > 0:
        return pi
    if getattr(args, 'verbose', False):
        return 0
    return 50


def resolve_pin_threads(args: ArgsLike) -> bool:
    """Return True if --pin-threads was requested. Validates platform support."""
    if not args.pin_threads:
        return False
    from .thread_pinning import is_supported
    if not is_supported():
        import sys as _sys
        print(f"Error: --pin-threads is not supported on this platform ({_sys.platform}). "
              f"Thread pinning requires Linux (os.sched_setaffinity).")
        _sys.exit(1)
    return True


def resolve_worker_count(args: ArgsLike, config: Optional['AppConfig'] = None) -> int:
    """Resolve number of workers for multiprocess/two-stage mode.

    Priority: command-line --workers > config programs.gmp_ecm.workers > CPU count.
    """
    workers = args.workers or 0
    if workers > 0:
        return workers
    if config is not None:
        config_workers = get_workers_default(config)
        if config_workers > 0:
            return config_workers
    return multiprocessing.cpu_count()


def print_validation_errors(errors: Dict[str, str]) -> None:
    """Print validation errors and exit."""
    if errors:
        print("Argument validation errors:")
        for field, message in errors.items():
            print(f"  {field}: {message}")
        sys.exit(1)

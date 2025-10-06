#!/usr/bin/env python3
"""
Shared argument parsing logic for ECM and YAFU wrappers.
"""
import argparse
import sys
import multiprocessing
from typing import Dict, Any, Optional


def create_ecm_parser() -> argparse.ArgumentParser:
    """Create argument parser for ECM wrapper."""
    parser = argparse.ArgumentParser(description='ECM Wrapper Client')
    
    # Configuration
    parser.add_argument('--config', default='client.yaml', help='Config file path')
    
    # Core parameters
    parser.add_argument('--composite', '-n', help='Number to factor')
    parser.add_argument('--b1', type=int, help='B1 bound (overrides config)')
    parser.add_argument('--b2', type=int, help='B2 bound')
    parser.add_argument('--curves', '-c', type=int, help='Number of curves')
    parser.add_argument('--project', '-p', help='Project name')
    parser.add_argument('--no-submit', action='store_true', help='Do not submit to API')
    
    # GPU options
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration (CGBN)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--gpu-device', type=int, help='GPU device number to use')
    parser.add_argument('--gpu-curves', type=int, help='Number of curves to compute in parallel on GPU')
    
    # Method and verbosity
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose ECM output')
    parser.add_argument('--progress-interval', type=int, default=0,
                       help='Show progress updates every N completed curves (0 = disabled)')
    parser.add_argument('--method', choices=['ecm', 'pm1', 'pp1'], default='ecm',
                       help='Factorization method (ECM, P-1, P+1)')
    
    # Advanced modes
    parser.add_argument('--two-stage', action='store_true', 
                       help='Use two-stage mode: GPU stage 1 + multi-threaded CPU stage 2')
    parser.add_argument('--stage2-workers', type=int, default=4, 
                       help='Number of CPU workers for stage 2 (default: 4)')
    parser.add_argument('--multiprocess', action='store_true', 
                       help='Use multi-process mode: parallel full ECM cycles (CPU-optimized)')
    parser.add_argument('--workers', type=int, default=0, 
                       help='Number of worker processes (default: CPU count)')
    
    # Residue file handling
    parser.add_argument('--save-residues', type=str, help='Save stage 1 residues with specified filename in configured residue_dir')
    parser.add_argument('--resume-residues', type=str, help='Resume from existing residue file (skip stage 1)')
    parser.add_argument('--stage2-only', type=str, help='Run stage 2 only on residue file path')

    # Factor handling
    parser.add_argument('--continue-after-factor', action='store_true',
                       help='Continue processing all curves even after finding a factor')


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
    parser.add_argument('--b1', type=int, help='B1 bound for ECM')
    parser.add_argument('--b2', type=int, help='B2 bound for ECM')
    parser.add_argument('--curves', '-c', type=int, default=100, help='Number of curves for ECM')
    
    # General parameters
    parser.add_argument('--project', '-p', help='Project name')
    parser.add_argument('--no-submit', action='store_true', help='Do not submit to API')
    
    return parser


def validate_ecm_args(args: argparse.Namespace, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Validate ECM arguments and return any validation errors.

    Args:
        args: Parsed command line arguments
        config: Configuration dictionary (optional, for B2 validation)

    Returns:
        Dictionary mapping argument names to error messages
    """
    errors = {}
    
    # Mode compatibility checks
    if args.multiprocess and args.two_stage:
        errors['mode'] = "Cannot use both --multiprocess and --two-stage. Choose one mode."

    
    # Resume residues mode validation (check first, overrides other modes)
    if args.resume_residues:
        if args.composite:
            errors['composite'] = "Resume residues mode - composite number not required (extracted from residue file)"
        if not args.b2:
            errors['b2'] = "Resume residues mode requires B2 bound. Use --b2 argument."

    # Stage 2 only mode validation
    elif args.stage2_only:
        if args.composite:
            errors['composite'] = "Stage 2 only mode - composite number not required"
        if not args.b2:
            errors['b2'] = "Stage 2 only mode requires B2 bound. Use --b2 argument."

    # Two-stage mode validation
    elif args.two_stage and args.method == 'ecm':
        if not args.composite:
            errors['composite'] = "Two-stage mode requires composite number. Use --composite argument."
        # Two-stage mode requires explicit B2 for Stage 2 coordination
        # Exception: B2=0 is allowed when saving residues (Stage 1 only)
        if args.b2 is None and config:
            _, b2_default = get_method_defaults(config, args.method)
            if not b2_default:
                errors['b2'] = "Two-stage mode requires B2 bound. Use --b2 argument or set default_b2 in config."
        elif args.b2 is None and not config:
            errors['b2'] = "Two-stage mode requires B2 bound. Use --b2 argument."
    
    # Multiprocess mode validation
    elif args.multiprocess:
        if not args.composite:
            errors['composite'] = "Multiprocess mode requires composite number. Use --composite argument."
        if args.save_residues or args.resume_residues:
            errors['residues'] = "Residue options not applicable in multiprocess mode."

    
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


def get_stage2_workers_default(config: Dict[str, Any]) -> int:
    """
    Get default stage2_workers value from config.

    Args:
        config: Configuration dictionary

    Returns:
        Default number of stage2 workers
    """
    if config and 'programs' in config and 'gmp_ecm' in config['programs']:
        return config['programs']['gmp_ecm'].get('stage2_workers', 4)
    return 4


def get_method_defaults(config: Dict[str, Any], method: str) -> tuple[int, Optional[int]]:
    """
    Get default B1 and B2 values for the specified method.

    Args:
        config: Configuration dictionary
        method: Method name ('ecm', 'pm1', 'pp1')

    Returns:
        Tuple of (b1_default, b2_default)
    """
    gmp_config = config['programs']['gmp_ecm']
    
    if method == 'pm1':
        b1_default = gmp_config.get('pm1_b1', gmp_config['default_b1'])
        b2_default = gmp_config.get('pm1_b2', gmp_config.get('default_b2'))
    elif method == 'pp1':
        b1_default = gmp_config.get('pp1_b1', gmp_config['default_b1'])
        b2_default = gmp_config.get('pp1_b2', gmp_config.get('default_b2'))
    else:  # ecm
        b1_default = gmp_config['default_b1']
        b2_default = gmp_config.get('default_b2')
    
    return b1_default, b2_default


def resolve_gpu_settings(args: argparse.Namespace, config: Dict[str, Any]) -> tuple[bool, Optional[int], Optional[int]]:
    """
    Resolve GPU settings from arguments and configuration.
    
    Returns:
        Tuple of (use_gpu, gpu_device, gpu_curves)
    """
    # GPU settings: command line overrides config defaults
    if args.no_gpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = config['programs']['gmp_ecm'].get('gpu_enabled', False)
    
    gpu_device = (args.gpu_device if args.gpu_device is not None 
                  else config['programs']['gmp_ecm'].get('gpu_device'))
    gpu_curves = (args.gpu_curves if args.gpu_curves is not None 
                  else config['programs']['gmp_ecm'].get('gpu_curves'))
    
    return use_gpu, gpu_device, gpu_curves


def resolve_worker_count(args: argparse.Namespace) -> int:
    """Resolve number of workers for multiprocess mode."""
    if args.multiprocess and args.workers <= 0:
        return multiprocessing.cpu_count()
    return args.workers


def print_validation_errors(errors: Dict[str, str]) -> None:
    """Print validation errors and exit."""
    if errors:
        print("Argument validation errors:")
        for field, message in errors.items():
            print(f"  {field}: {message}")
        sys.exit(1)
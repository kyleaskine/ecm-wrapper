#!/usr/bin/env python3
"""
Optimized ECM wrapper with same command-line interface.
Drop-in replacement for ecm-wrapper.py with better performance.
"""
import sys
from optimized_base_wrapper import OptimizedECMWrapper

def main():
    from arg_parser import create_ecm_parser, validate_ecm_args, print_validation_errors
    from arg_parser import get_method_defaults, resolve_gpu_settings, resolve_worker_count, get_stage2_workers_default

    parser = create_ecm_parser()
    args = parser.parse_args()

    # Use optimized wrapper instead of legacy ECMWrapper
    wrapper = OptimizedECMWrapper(args.config)

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
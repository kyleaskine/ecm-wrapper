"""
Extracted mode handlers for ecm_wrapper.py.

Each function corresponds to one execution mode from main(). They accept
a common set of parameters via ResolvedParams so the branching logic can
be unit-tested with mocked wrappers.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

from .arg_parser import resolve_stage2_progress_interval
from .ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult
from .ecm_math import calculate_target_tlevel, is_probably_prime
from .results_builder import results_for_stage1
from .stage1_helpers import submit_stage1_complete_workflow
from .user_output import UserOutput


@dataclass
class ResolvedParams:
    """Bundle of values derived from args + config, shared across all modes."""
    b1: int
    method: str
    use_gpu: bool
    gpu_device: Optional[int]
    gpu_curves: Optional[int]
    workers: int
    max_batch: Optional[int]
    b2_dictionary: Optional[Dict[int, int]]


def run_stage2_only_mode(wrapper, args, output: UserOutput, params: ResolvedParams) -> Tuple[FactorResult, str]:
    """Stage 2 Only Mode - load residue from local file.

    Returns (result, composite) since the composite is extracted from the residue file.
    """
    import sys

    if not args.b2:
        output.error("--stage2-only requires --b2")
        sys.exit(1)

    residue_path = Path(args.stage2_only)
    if not residue_path.exists():
        output.error(f"Residue file not found: {args.stage2_only}")
        sys.exit(1)

    # Parse residue file for B1 and composite
    residue_info = wrapper._parse_residue_file(residue_path)
    b1 = residue_info.get('b1', 0)
    composite_from_residue = residue_info.get('composite', 'unknown')

    output.mode_header("Stage 2 Only Mode", {
        "Residue": args.stage2_only,
        "Composite": composite_from_residue[:40] + "..." if len(composite_from_residue) > 40 else composite_from_residue,
        "B2": args.b2,
        "Workers": params.workers
    })

    if b1 == 0:
        output.warning("Could not parse B1 from residue file, using 0")

    # Use engine's two-stage pipeline with resume (installs 3-level Ctrl+C handler)
    batch_result = wrapper._engine.run_two_stage(
        composite=composite_from_residue,
        b1=b1,
        b2=args.b2,
        stage1_curves=0,
        stage2_workers=params.workers,
        verbose=args.verbose or False,
        progress_interval=resolve_stage2_progress_interval(args),
        resume_file=residue_path,
    )

    result = batch_result.to_factor_result()
    if batch_result.interrupted:
        output.info(f"Graceful shutdown completed - processed {batch_result.curves_run} curves")

    return result, composite_from_residue


def run_stage1_only_mode(wrapper, args, output: UserOutput, params: ResolvedParams) -> FactorResult:
    """Stage 1 Only Mode - save residue to local file (optionally upload with --upload)."""
    import sys

    if not args.b1:
        output.error("--stage1-only requires --b1")
        sys.exit(1)

    residue_dir = Path("data/residues")
    residue_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.save_residues or str(residue_dir / f"residue_{hash(args.composite) % 100000}_{int(time.time())}.txt")

    output.mode_header("Stage 1 Only Mode", {
        "Save to": save_path,
        "Composite": args.composite,
        "B1": params.b1,
        "Curves": args.curves or 1
    })
    if args.upload:
        output.item("Upload", "Will upload residue to server after completion")

    # Create config for stage 1 execution
    param = args.param or (3 if params.use_gpu else 1)
    config = ECMConfig(
        composite=args.composite,
        b1=params.b1,
        b2=0,  # Stage 1 only
        curves=args.curves or 1,
        sigma=args.sigma,
        parametrization=param,
        threads=1,
        verbose=args.verbose or False,
        save_residues=save_path,
        use_gpu=params.use_gpu,
        gpu_device=params.gpu_device,
        gpu_curves=params.gpu_curves,
        method=params.method,
        progress_interval=args.progress_interval or 0
    )

    result = wrapper.run_ecm_v2(config)

    if result.success or result.curves_run > 0:
        output.section("Stage 1 complete:")
        output.item("Residue saved", save_path)
        output.item("Curves run", result.curves_run)
        if result.factors:
            output.item("Factors found", result.factors)

        # Upload residue to server if --upload flag is set
        if args.upload and args.submit:
            residue_path = Path(save_path)
            if residue_path.exists():
                # Get client_id from config
                client_id = wrapper.typed_config.client.username

                # Build factor info for upload
                factor_found = result.factors[0] if result.factors else None
                all_factors = [(f, result.sigmas[i] if i < len(result.sigmas) else None)
                               for i, f in enumerate(result.factors)]

                # Build results dict for submission
                results = (results_for_stage1(args.composite, params.b1, result.curves_run, param)
                    .with_curves(result.curves_run, result.curves_run)
                    .with_factors(all_factors)
                    .with_execution_time(result.execution_time)
                    .add_raw_output(result.raw_output or "")
                    .build())

                # Use the consolidated workflow to submit + upload
                attempt_id = submit_stage1_complete_workflow(
                    wrapper=wrapper,
                    results=results,
                    residue_file=residue_path,
                    work_id=None,  # No work assignment for manual mode
                    project=args.project,
                    client_id=client_id,
                    factor_found=factor_found,
                    cleanup_residue=False  # Keep local copy
                )

                if attempt_id:
                    output.item("Residue uploaded", f"attempt_id: {attempt_id}")
                else:
                    output.warning("Failed to upload residue to server")
            else:
                output.warning(f"Residue file not found at {save_path}")

        # Mark that we've already submitted if --upload was used
        if args.upload:
            args.submit = False  # Prevent double submission

    return result


def run_tlevel_mode(wrapper, args, output: UserOutput, params: ResolvedParams) -> FactorResult:
    """T-level Mode (including progressive factorization when --tlevel given without value)."""
    # Determine if we're in progressive mode (auto t-level calculation)
    is_progressive = args.tlevel < 0  # -1.0 sentinel means auto-calculate

    # Current state for progressive factorization
    current_composite = args.composite
    current_t_level = args.start_tlevel or 0.0
    all_factors: list[str] = []
    all_curve_summaries: list[dict] = []  # Aggregate curve summary across all iterations
    total_execution_time = 0.0
    total_curves_run = 0
    final_t_level = 0.0

    # Progressive factorization loop
    while True:
        digit_length = len(current_composite)

        # Calculate or use explicit target t-level
        if is_progressive:
            target_t_level = calculate_target_tlevel(digit_length)
        else:
            target_t_level = args.tlevel

        # Skip if we've already exceeded the target
        if current_t_level >= target_t_level:
            output.info(f"Already at t{current_t_level:.2f} >= target t{target_t_level:.1f}")
            break

        mode_name = "Progressive T-level Mode" if is_progressive else "T-level Mode"
        output.mode_header(mode_name, {
            "Target": f"t{target_t_level:.1f}",
            "Current": f"t{current_t_level:.2f}" if current_t_level > 0 else "t0",
            "Composite": f"C{digit_length} ({current_composite[:20]}...)" if len(current_composite) > 25 else f"C{digit_length}"
        })

        tlevel_progress_interval = (
            resolve_stage2_progress_interval(args)
            if args.two_stage
            else (args.progress_interval or 0)
        )
        config = TLevelConfig(
            composite=current_composite,
            target_t_level=target_t_level,
            start_t_level=current_t_level,
            b1_strategy='optimal',
            parametrization=args.param or (3 if args.two_stage else 1),
            threads=args.workers or 1,
            verbose=args.verbose or False,
            workers=args.workers or 1,
            use_two_stage=args.two_stage or False,
            progress_interval=tlevel_progress_interval,
            max_batch_curves=params.max_batch,
            b2_multiplier=getattr(args, 'b2_multiplier', None) or 500.0,
            b2_dictionary=params.b2_dictionary,
            project=args.project,
            no_submit=not args.submit,
            gpu_device=params.gpu_device,
            gpu_curves=params.gpu_curves
        )

        result = wrapper.run_tlevel_v2(config)

        # Aggregate statistics from this iteration
        if result.curve_summary:
            all_curve_summaries.extend(result.curve_summary)
        total_execution_time += result.execution_time
        total_curves_run += result.curves_run
        final_t_level = result.t_level_achieved if result.t_level_achieved else final_t_level

        # Collect factors
        if result.factors:
            all_factors.extend(result.factors)
            output.success(f"Found {len(result.factors)} factor(s): {', '.join(result.factors[:3])}{'...' if len(result.factors) > 3 else ''}")

            # Update composite by dividing out factors
            composite_int = int(current_composite)
            for factor in result.factors:
                factor_int = int(factor)
                while composite_int % factor_int == 0:
                    composite_int //= factor_int

            # Check if fully factored
            if composite_int == 1:
                output.success("Fully factored!")
                break

            # Check if remaining cofactor is prime
            if is_probably_prime(composite_int):
                output.success(f"Cofactor C{len(str(composite_int))} is prime - factorization complete!")
                all_factors.append(str(composite_int))
                break

            # Continue with cofactor in progressive mode
            if is_progressive:
                current_composite = str(composite_int)
                # T-level achieved carries over to cofactor
                current_t_level = result.t_level_achieved if result.t_level_achieved else 0.0
                output.info(f"Continuing with cofactor C{len(current_composite)} from t{current_t_level:.2f}")
            else:
                # Explicit t-level mode: stop after finding factors
                break
        else:
            # No factors found
            if result.interrupted:
                achieved = result.t_level_achieved if result.t_level_achieved else current_t_level
                output.warning(f"Interrupted at t{achieved:.2f} (target was t{target_t_level:.1f})")
                break
            if is_progressive:
                output.info(f"Reached t{target_t_level:.1f} with no factor found")
            break

        # Check for interrupt
        if result.interrupted:
            output.warning("Interrupted by user")
            break

    # Build final aggregate result with all collected data
    result = FactorResult()
    for f in all_factors:
        result.add_factor(f, None)
    result.success = len(all_factors) > 0
    result.curves_run = total_curves_run
    result.execution_time = total_execution_time
    result.curve_summary = all_curve_summaries
    result.t_level_achieved = final_t_level

    # Print all factors found at the end
    if all_factors:
        output.section("All Factors Found")
        for i, factor in enumerate(all_factors, 1):
            output.item(f"Factor {i}", f"{factor} ({len(factor)} digits)")

    return result


def run_multiprocess_mode(wrapper, args, output: UserOutput, params: ResolvedParams) -> FactorResult:
    """Multiprocess Mode - parallel CPU workers."""
    output.mode_header("Multiprocess Mode", {
        "Workers": args.workers or "auto",
        "Composite": args.composite,
        "B1": params.b1,
        "B2": args.b2 or "default"
    })

    config = MultiprocessConfig(
        composite=args.composite,
        b1=params.b1,
        b2=args.b2,
        total_curves=args.curves or 1000,
        curves_per_process=100,
        num_processes=args.workers,
        parametrization=args.param or 1,
        method=args.method or 'ecm',
        verbose=args.verbose or False,
        continue_after_factor=False,
        progress_interval=args.progress_interval or 0
    )

    return wrapper.run_multiprocess_v2(config)


def run_two_stage_mode(wrapper, args, output: UserOutput, params: ResolvedParams) -> FactorResult:
    """Two-stage Mode - GPU stage 1 + CPU stage 2 pipeline."""
    import sys

    # Resolve B2: explicit --b2, then dictionary lookup, then multiplier
    # (same precedence as t-level mode's get_b2_for_b1)
    if args.b2 is not None:
        two_stage_b2 = args.b2
    elif params.b2_dictionary and params.b1 in params.b2_dictionary:
        two_stage_b2 = params.b2_dictionary[params.b1]
    elif getattr(args, 'b2_multiplier', None) is not None:
        two_stage_b2 = int(params.b1 * args.b2_multiplier)
    elif params.b2_dictionary is not None:
        # Dictionary was the only B2 source but has no entry for this B1
        output.error(f"B1 {params.b1} not found in B2 dictionary. "
                     f"Add an entry for it or use --b2/--b2-multiplier.")
        sys.exit(1)
    else:
        two_stage_b2 = None  # Use GMP-ECM default

    output.mode_header("Two-stage Mode", {
        "Pipeline": "GPU stage 1 + CPU stage 2",
        "Composite": args.composite,
        "B1": params.b1,
        "B2": two_stage_b2 if two_stage_b2 is not None else "default"
    })

    config = TwoStageConfig(
        composite=args.composite,
        b1=params.b1,
        b2=two_stage_b2,
        stage1_curves=args.curves or 100,  # Use --curves for stage 1
        stage2_curves_per_residue=1000,     # Default for stage 2
        stage1_device="GPU",  # Two-stage always uses GPU for stage 1
        stage2_device="CPU",
        stage1_parametrization=args.param or 3,
        stage2_parametrization=1,
        threads=params.workers,
        verbose=args.verbose or False,
        save_residues=args.save_residues,
        gpu_device=params.gpu_device,
        gpu_curves=params.gpu_curves,
        continue_after_factor=False,
        progress_interval=resolve_stage2_progress_interval(args),
        project=args.project,
        no_submit=not args.submit
    )

    return wrapper.run_two_stage_v2(config)


def run_standard_mode(wrapper, args, output: UserOutput, params: ResolvedParams) -> FactorResult:
    """Standard Mode - basic ECM factorization."""
    output.mode_header("Standard ECM Mode", {
        "Composite": args.composite,
        "B1": params.b1,
        "B2": args.b2 if args.b2 is not None else "default",
        "Curves": args.curves or 1
    })

    config = ECMConfig(
        composite=args.composite,
        b1=params.b1,
        b2=args.b2,
        curves=args.curves or 1,
        sigma=args.sigma,
        parametrization=args.param or (3 if params.use_gpu else 1),
        threads=1,
        verbose=args.verbose or False,
        save_residues=args.save_residues,
        use_gpu=params.use_gpu,
        gpu_device=params.gpu_device,
        gpu_curves=params.gpu_curves,
        method=params.method,
        progress_interval=args.progress_interval or 0
    )

    return wrapper.run_ecm_v2(config)


def submit_ecm_result(wrapper, args, output: UserOutput, params: ResolvedParams,
                      result: FactorResult, composite: str) -> None:
    """Submit results to API if appropriate.

    T-level mode handles its own submissions internally,
    so we skip post-execution submission for that mode to avoid double submission.
    """
    mode_handles_own_submission = args.tlevel is not None
    if not (result and args.submit and not mode_handles_own_submission):
        return

    results_dict = result.to_dict(composite, params.method)

    # Add ECM parameters that aren't in FactorResult
    results_dict['b1'] = params.b1
    # Stage1-only mode should always submit b2=0, not None
    results_dict['b2'] = 0 if getattr(args, 'stage1_only', False) else args.b2
    results_dict['curves_requested'] = args.curves
    results_dict['parametrization'] = args.param or (3 if getattr(args, 'gpu', False) else 1)

    # Add project if specified
    if args.project:
        results_dict['project'] = args.project

    # Submit via API
    if result.success and result.factors:
        output.info(f"\nSubmitting {len(result.factors)} factor(s) to API...")
        wrapper.submit_result(results_dict, args.project, f"gmp-ecm-{params.method}")
    elif result.curves_run > 0:
        output.info(f"\nSubmitting {result.curves_run} curves (no factors) to API...")
        wrapper.submit_result(results_dict, args.project, f"gmp-ecm-{params.method}")

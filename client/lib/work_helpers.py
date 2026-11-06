#!/usr/bin/env python3
"""
Helper functions for ECM work assignment display and management.

These utilities eliminate duplicated work handling code in auto-work mode.
"""
from typing import Optional, Dict, Any, Tuple
import sys
import time

from .ecm_math import MIN_BATCH_B1, b1_for_current_tlevel
from .user_output import UserOutput
from .work_args import WorkArgs


def print_work_header(work_id: Optional[str], composite: str, digit_length: int,
                     params: Dict[str, Any], output: Optional[UserOutput] = None) -> None:
    """
    Print formatted work assignment header.

    Args:
        work_id: Work assignment ID (may be None for residue work)
        composite: Composite number being factored
        digit_length: Number of digits in composite
        params: Dictionary of parameters to display (e.g., {'B1': 50000, 'curves': 100})
        output: Optional UserOutput instance (creates one if not provided)
    """
    if output is None:
        output = UserOutput()

    output.blank()
    output.separator()
    work_id_display = work_id if work_id else "(no work_id)"
    output.info(f"Processing work assignment {work_id_display}")

    # Truncate composite for display
    composite_display = composite[:50] + "..." if len(composite) > 50 else composite
    output.info(f"Composite: {composite_display} ({digit_length} digits)")

    # Format parameters
    if params:
        param_strs = [f"{k}={v}" for k, v in params.items()]
        output.info(f"Parameters: {', '.join(param_strs)}")

    output.separator()
    output.blank()


def print_work_status(stage_name: str, completed_count: int,
                     work_count_limit: Optional[int] = None,
                     output: Optional[UserOutput] = None) -> bool:
    """
    Print work completion status and check if limit reached.

    Args:
        stage_name: Name of the stage/mode (e.g., "Stage 1", "Stage 2 work")
        completed_count: Number of work assignments completed
        work_count_limit: Optional limit on number of assignments
        output: Optional UserOutput instance (creates one if not provided)

    Returns:
        True if work count limit has been reached, False otherwise
    """
    if output is None:
        output = UserOutput()

    output.blank()

    # Display completion status
    if work_count_limit:
        output.info(f"{stage_name} complete ({completed_count}/{work_count_limit})")
    else:
        output.info(f"{stage_name} complete (total: {completed_count})")

    output.separator()
    output.blank()

    # Check if limit reached
    if work_count_limit and completed_count >= work_count_limit:
        output.info(f"Reached work count limit ({work_count_limit}), exiting...")
        return True

    return False


_no_work_count = 0


def _handle_no_work(logger, label: str, exit_on_no_work: bool = False) -> None:
    """Handle repeated 'no work available' with escalating warnings."""
    global _no_work_count
    _no_work_count += 1

    if exit_on_no_work:
        logger.info(f"No {label} work available. Exiting (--exit-on-no-work).")
        sys.exit(0)

    if _no_work_count >= 5 and _no_work_count % 5 == 0:
        logger.warning(
            f"No {label} work available {_no_work_count} times in a row. "
            f"If this persists, check for stale work assignments on the server "
            f"(admin dashboard -> outstanding work) or restart with a fresh client_id."
        )

    logger.info(f"No {label} work available, waiting 30 seconds before retry...")
    time.sleep(30)


def _reset_no_work_count() -> None:
    """Reset counter when work is successfully received."""
    global _no_work_count
    _no_work_count = 0


def request_ecm_work(api_client, client_id: str, args: WorkArgs,
                    logger) -> Optional[Dict[str, Any]]:
    """
    Request ECM work from server with automatic retry on failure.

    Args:
        api_client: API client instance
        client_id: Client identifier
        args: Typed WorkArgs containing filter parameters
        logger: Logger instance for info messages

    Returns:
        Work assignment dictionary or None if no work available after retry
    """
    work = api_client.get_ecm_work(
        client_id=client_id,
        min_target_tlevel=args.min_target_tlevel,
        max_target_tlevel=args.max_target_tlevel,
        # If client specifies --tlevel, filter out composites already past that level
        max_current_tlevel=args.tlevel,
        # --one-batch picks its B1 as the stage 1 producer does: the server
        # sizes the claim from that B1
        stage1_only=args.stage1_only or args.one_batch,
        requested_b1=args.b1,
        priority=args.priority,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        work_type=args.work_type,
        project=args.project
    )

    if not work:
        _handle_no_work(logger, "ECM", exit_on_no_work=args.exit_on_no_work)
    else:
        _reset_no_work_count()

    return work


def request_p1_work(api_client, client_id: str, args: WorkArgs,
                    logger) -> Optional[Dict[str, Any]]:
    """
    Request P-1/P+1 work from server with automatic retry on failure.

    Calls the /p1-work endpoint which only assigns composites that haven't
    had PM1/PP1 done at the required B1 level.

    Args:
        api_client: API client instance
        client_id: Client identifier
        args: Typed WorkArgs containing filter parameters and method flags
        logger: Logger instance for info messages

    Returns:
        Work assignment dictionary or None if no work available after retry
    """
    if args.pm1:
        method = 'pm1'
    elif args.pp1:
        method = 'pp1'
    else:
        method = 'p1'

    work = api_client.get_p1_work(
        client_id=client_id,
        method=method,
        min_target_tlevel=args.min_target_tlevel,
        max_target_tlevel=args.max_target_tlevel,
        priority=args.priority,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        work_type=args.work_type,
        project=args.project
    )

    if not work:
        _handle_no_work(logger, "P1", exit_on_no_work=args.exit_on_no_work)
    else:
        _reset_no_work_count()

    return work


def one_gpu_batch_params(args: WorkArgs, curves_per_batch: int,
                         current_t: Optional[float], logger,
                         label: str) -> Tuple[int, int]:
    """
    (B1, curves) for a work unit that is one GPU batch (--stage1-only,
    --one-batch).

    B1 is --b1, else the optimal B1 for the next t-level above current_t
    (t54.7 -> the B1 for t55); either way at least MIN_BATCH_B1, so fast GPU
    runs don't flood the server with tiny submissions. Curves is --curves,
    else curves_per_batch: GMP-ECM rounds a GPU request up to its natural
    batch size, so a request no larger than that runs exactly one batch.
    """
    current_t = current_t or 0.0
    if args.b1 is not None:
        b1 = args.b1
        if b1 < MIN_BATCH_B1:
            logger.warning(f"B1={b1} below minimum {MIN_BATCH_B1} for {label}, using minimum")
            b1 = MIN_BATCH_B1
    else:
        b1 = b1_for_current_tlevel(current_t)
        if b1 < MIN_BATCH_B1:
            logger.info(f"{label}: B1={b1} (t{current_t:.1f}) below minimum, "
                        f"using B1={MIN_BATCH_B1}")
            b1 = MIN_BATCH_B1
    curves = args.curves if args.curves is not None else curves_per_batch
    if curves <= 0:
        raise ValueError(f"--curves must be positive for {label}, got {curves}")
    logger.info(f"{label}: t{current_t:.1f} using B1={b1}, curves={curves} (one batch)")
    return b1, curves

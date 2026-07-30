#!/usr/bin/env python3
"""
Work mode strategy pattern for ECM auto-work execution.

Each mode implements the same abstract interface:
- request_work() - Get work assignment from server
- execute_work() - Run ECM on the assignment
- submit_results() - Submit results to server
- complete_work() - Mark work as complete
- cleanup_on_failure() - Mode-specific cleanup

The base class provides the work loop template that all modes share.

This package was split out of the original `lib/work_modes.py` so each mode
lives in its own module. Public symbols are re-exported here so existing
imports (`from lib.work_modes import WorkLoopContext, get_work_mode`) keep
working unchanged.
"""

from .base import WorkMode, WorkLoopContext, MAX_CONSECUTIVE_FAILURES, SubmissionFailedError
from .stage1_producer import Stage1ProducerMode
from .stage2_consumer import Stage2ConsumerMode
from .p1_sweep import P1WorkMode
from .standard import StandardAutoWorkMode
from .composite_target import CompositeTargetMode
from .adaptive import AdaptiveCPUMode


def get_work_mode(ctx: WorkLoopContext) -> WorkMode:
    """
    Factory function to create the appropriate WorkMode based on args.

    Priority order:
    1. Explicit mode flags (--composite, --pm1, --stage1-only, --stage2-only)
    2. Adaptive CPU mode (--adaptive or interactive selection)
    3. Standard auto-work mode (legacy default with explicit --standard flag,
       or when B1/tlevel args are provided)

    Args:
        ctx: Work loop context with wrapper, client_id, and args

    Returns:
        Appropriate WorkMode subclass instance
    """
    args = ctx.args

    if args.composite:
        return CompositeTargetMode(ctx)
    elif args.pm1 or args.pp1 or args.p1:
        return P1WorkMode(ctx)
    elif args.stage1_only:
        return Stage1ProducerMode(ctx)
    elif args.stage2_only:
        return Stage2ConsumerMode(ctx)
    elif args.adaptive:
        return AdaptiveCPUMode(ctx)
    else:
        return StandardAutoWorkMode(ctx)


__all__ = [
    'WorkMode',
    'WorkLoopContext',
    'MAX_CONSECUTIVE_FAILURES',
    'SubmissionFailedError',
    'Stage1ProducerMode',
    'Stage2ConsumerMode',
    'P1WorkMode',
    'StandardAutoWorkMode',
    'CompositeTargetMode',
    'AdaptiveCPUMode',
    'get_work_mode',
]

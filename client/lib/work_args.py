#!/usr/bin/env python3
"""
Typed arguments for the auto-work loop.

`WorkArgs` mirrors the flags exposed by `create_client_parser()` in
`lib.arg_parser`, plus a couple of fields that downstream code mutates after
parsing (`auto_work`). All fields have defaults so direct attribute access is
always safe — this replaces the long chain of
`getattr(self.args, 'flag', default)` calls that used to live in
`lib/work_modes/*.py`.

Construct one via `WorkArgs.from_namespace(parser.parse_args())` at the entry
point and pass it to `WorkLoopContext`.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkArgs:
    """Typed args for the auto-work loop.

    Field defaults match the argparse defaults from `create_client_parser()`,
    so a `WorkArgs()` instance behaves identically to running `ecm_client.py`
    with no flags.
    """

    # Composite targeting
    composite: Optional[str] = None

    # Core ECM parameters
    b1: Optional[int] = None
    b2: Optional[int] = None
    b2_multiplier: Optional[float] = None
    b2_dictionary: Optional[str] = None
    max_batch: Optional[int] = None
    method: str = "ecm"
    curves: Optional[int] = None

    # T-level targeting
    tlevel: Optional[float] = None
    # --one-batch: one GPU batch per assignment instead of a t-level climb
    one_batch: bool = False
    # `--start-tlevel` is only on `create_ecm_parser`; kept here so
    # `StandardAutoWorkMode` can read it defensively without `hasattr`.
    start_tlevel: Optional[float] = None

    # Execution mode
    multiprocess: bool = False
    two_stage: bool = False
    workers: Optional[int] = None
    pin_threads: bool = False

    # Server work filtering
    work_count: Optional[int] = None
    exit_on_no_work: bool = False
    min_digits: Optional[int] = None
    max_digits: Optional[int] = None
    priority: Optional[int] = None
    work_type: str = "standard"
    min_target_tlevel: Optional[float] = None
    max_target_tlevel: Optional[float] = None

    # Behavior
    verbose: bool = False
    progress_interval: int = 0
    continue_after_factor: bool = False
    maxmem: Optional[int] = None

    # Stage 2 filters
    min_b1: Optional[int] = None
    max_b1: Optional[int] = None

    # P-1/P+1
    pm1: bool = False
    pp1: bool = False
    p1: bool = False
    pp1_curves: int = 3

    # Mode flags (mutually exclusive in the parser, but we store all)
    stage1_only: bool = False
    stage2_only: bool = False
    adaptive: bool = False
    standard: bool = False

    # GPU
    gpu: bool = False
    no_gpu: bool = False
    gpu_device: Optional[int] = None
    gpu_curves: Optional[int] = None

    # Sigma / parametrization
    sigma: Optional[str] = None
    param: Optional[int] = None  # 0..3

    # API
    project: Optional[str] = None
    no_submit: bool = False

    # Hidden / runtime-set
    auto_work: bool = True  # ecm_client.py always operates in auto-work mode
    auto_work_explicit: bool = False  # set by `--auto-work` (suppressed flag)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "WorkArgs":
        """Build a `WorkArgs` from an `argparse.Namespace`.

        Copies every field the dataclass knows about that's also present on the
        Namespace. Unknown Namespace attributes are silently ignored — this
        keeps `WorkArgs` decoupled from any single parser definition. Fields
        absent from the Namespace fall back to the dataclass default.
        """
        kwargs = {}
        for f in dataclasses.fields(cls):
            if hasattr(ns, f.name):
                kwargs[f.name] = getattr(ns, f.name)
        return cls(**kwargs)

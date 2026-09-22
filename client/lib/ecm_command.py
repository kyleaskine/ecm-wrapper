#!/usr/bin/env python3
"""
Shared GMP-ECM command-line builder.

Single source of truth for constructing GMP-ECM command lines.
All call sites (ecm_executor, ecm_worker_process, stage2_executor)
delegate to build_ecm_command() to ensure consistent flag ordering
and B2 handling.

This is a standalone function (no class) so it remains pickleable
for multiprocessing workers.
"""
from pathlib import Path
from typing import List, Optional, Union


def build_ecm_command(
    ecm_path: str,
    b1: int,
    *,
    b2: Optional[int] = None,
    k: Optional[int] = None,
    curves: Optional[int] = None,
    method: str = "ecm",
    use_gpu: bool = False,
    gpu_device: Optional[int] = None,
    gpu_curves: Optional[int] = None,
    residue_save: Optional[Path] = None,
    residue_load: Optional[Path] = None,
    verbose: bool = False,
    param: Optional[int] = None,
    sigma: Optional[Union[str, int]] = None,
    one: bool = False,
    maxmem: Optional[int] = None,
    b1done: Optional[int] = None,
) -> List[str]:
    """
    Build a GMP-ECM command line with correct flag ordering.

    Flag ordering (per GMP-ECM requirements):
        method -> GPU -> residue ops -> verbose -> param -> sigma -> -one -> curves -> B1 -> B2

    B2 rules:
        None or -1  -> omit (GMP-ECM uses its default)
        0           -> include as "0" (stage 1 only)
        >0          -> include

    Args:
        ecm_path: Path to GMP-ECM binary
        b1: B1 bound
        b2: B2 bound (None/-1 = omit, 0 = stage1 only, >0 = explicit)
        k: number of segments in stage 2; automatic if None
        curves: Number of curves (-c flag); omitted if None
        method: "ecm", "pm1", or "pp1"
        use_gpu: Enable GPU acceleration (ECM only)
        gpu_device: GPU device number (-gpudevice)
        gpu_curves: Curves per GPU batch (-gpucurves)
        residue_save: Path for -save flag
        residue_load: Path for -resume flag
        verbose: Enable -v flag. GPU ECM always gets -v regardless, because
            GMP-ECM gates its GPU stage 1 progress/ETA output behind
            OUTPUT_VERBOSE (cgbn_stage1.cu) - without -v a multi-hour batch
            prints nothing at all. GMP-ECM self-throttles those lines
            ("5 early + 5 per 10s + 5 per 100s + every 1000s"), so this costs
            roughly a dozen lines per invocation however long it runs.
        param: Parametrization value (-param, ECM only)
        sigma: Sigma value (-sigma, ECM only)
        one: Stop after first factor (-one)
        maxmem: Maximum memory in MB for stage 2 (-maxmem)
        b1done: Stage 1 already complete up to this bound. When set,
            B1 is formatted as "b1done-b1" so GMP-ECM skips stage 1.
            Use b1done=b1 for pure stage 2 runs (e.g. resuming P95 residues).

    Returns:
        List of command-line arguments suitable for subprocess.
    """
    cmd: List[str] = [ecm_path]

    # 1. Method flags
    if method == "pm1":
        cmd.append("-pm1")
    elif method == "pp1":
        cmd.append("-pp1")

    # 2. GPU flags (ECM only)
    if use_gpu and method == "ecm":
        cmd.append("-gpu")
        if gpu_device is not None:
            cmd.extend(["-gpudevice", str(gpu_device)])
        if gpu_curves is not None:
            cmd.extend(["-gpucurves", str(gpu_curves)])

    # 3. Residue operations
    if residue_save:
        cmd.extend(["-save", str(residue_save)])
    if residue_load:
        cmd.extend(["-resume", str(residue_load)])

    # 4. Verbose. GPU stage 1 progress is OUTPUT_VERBOSE-gated upstream, and
    # GMP-ECM already rate-limits it, so always ask for it on GPU runs.
    if verbose or (use_gpu and method == "ecm"):
        cmd.append("-v")

    # 5. Parametrization (ECM only)
    if param is not None and method == "ecm":
        cmd.extend(["-param", str(param)])

    # 6. Sigma (ECM only)
    if sigma and method == "ecm":
        cmd.extend(["-sigma", str(sigma)])

    # 7. -one flag
    if one:
        cmd.append("-one")

    # 8. Curves
    if curves is not None:
        cmd.extend(["-c", str(curves)])

    # 9. optional k (number of segments in stage 2)
    if k is not None:
        cmd.extend(["-k", str(k)])

    # 10. optional maxmem (memory limit for stage 2 in MB)
    if maxmem is not None:
        cmd.extend(["-maxmem", str(maxmem)])

    # 11. B1 (with optional B1done prefix for stage 2 resume)
    if b1done is not None:
        cmd.append(f"{b1done}-{b1}")
    else:
        cmd.append(str(b1))

    # 12. B2: omit when None or -1
    if b2 is not None and b2 != -1:
        cmd.append(str(b2))

    return cmd

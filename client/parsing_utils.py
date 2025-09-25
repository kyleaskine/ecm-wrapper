#!/usr/bin/env python3
"""
Optimized parsing utilities and constants for ECM and YAFU output parsing.
"""
import re
import logging
from typing import Optional, Tuple, List, Dict, Any

# Configure logger for parsing operations
logger = logging.getLogger(__name__)


# Timeout constants (seconds)
class Timeouts:
    ECM_DEFAULT = 3600  # 1 hour
    YAFU_ECM = 7200     # 2 hours  
    YAFU_AUTO = 14400   # 4 hours for NFS


# Compiled regex patterns for performance
class ECMPatterns:
    """Compiled regex patterns for ECM output parsing."""

    # GPU format: "GPU: factor 12345 found in Step 1 with curve 0 (-sigma 3:2126921240)"
    GPU_FACTOR = re.compile(r'GPU: factor (\d+) found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?')

    # Standard format: "Factor found in step 1: 67280421310721"
    STANDARD_FACTOR = re.compile(r'Factor found in step \d+: (\d+)')

    # Prime factor lines: "Found prime factor of 14 digits: 59460190057621"
    PRIME_FACTOR = re.compile(r'Found prime factor of \d+ digits: (\d+)')

    # Sigma parameter extraction
    SIGMA_PARAM = re.compile(r'-sigma (3:\d+|\d+)')

    # Step completion tracking
    STEP_COMPLETED = re.compile(r'Step 1 took')

    # Curve completion tracking - "Step 2 took xxxms" indicates curve completion
    CURVE_COMPLETED = re.compile(r'Step 2 took (\d+)ms')

    # Alternative curve completion patterns
    CURVE_COMPLETED_ALT = re.compile(r'ECM: Step 2 took (\d+)ms')


class YAFUPatterns:
    """Compiled regex patterns for YAFU output parsing."""
    
    # Direct factor announcements
    FACTOR_FOUND = re.compile(r'found factor[:\s]+(\d+)', re.IGNORECASE)
    
    # P/Q notation: "P39 = 123456789012345678901234567890123456789"
    PQ_NOTATION = re.compile(r'[PQ]\d+\s*=\s*(\d+)')
    
    # Curves completion tracking
    CURVES_COMPLETED = re.compile(r'completed?\s+(\d+)\s+curves?', re.IGNORECASE)
    
    # Curve progress: "curve X of Y"
    CURVE_PROGRESS = re.compile(r'curve\s+(\d+)\s+of\s+(\d+)', re.IGNORECASE)
    
    # Factor section markers
    FACTOR_SECTION_START = re.compile(r'\*{3,}factors? found\*{3,}', re.IGNORECASE)
    
    # Factor lines in auto mode: "C123 = 456..." 
    AUTO_FACTOR = re.compile(r'[PCQ]\d+\s*=\s*(\d+)')
    
    # Simple number lines (fallback)
    SIMPLE_NUMBER = re.compile(r'^\s*(\d+)\s*$')


def parse_ecm_output_multiple(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse GMP-ECM output for multiple prime factors only.
    Uses ECM's own "Found prime factor" identification to avoid composite factors.

    Returns:
        List of (factor, sigma) tuples. Empty list if no factors found.
    """
    factors = []

    # Look for explicit prime factor announcements first
    prime_matches = ECMPatterns.PRIME_FACTOR.findall(output)
    if prime_matches:
        logger.debug(f"Found {len(prime_matches)} prime factors via PRIME_FACTOR pattern")
        for factor in prime_matches:
            # For prime factors, we need to find the corresponding sigma
            # Look for the GPU line that mentions this factor
            gpu_match = re.search(rf'GPU: factor {factor} found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?', output)
            sigma = f"3:{gpu_match.group(1)}" if gpu_match and gpu_match.group(1) else None
            factors.append((factor, sigma))
        return factors

    # Fallback to GPU format if no prime factor announcements
    gpu_matches = ECMPatterns.GPU_FACTOR.findall(output)
    if gpu_matches:
        logger.debug(f"Found {len(gpu_matches)} factors via GPU_FACTOR pattern")
    for match in gpu_matches:
        factor = match[0]
        sigma = f"3:{match[1]}" if match[1] else None
        factors.append((factor, sigma))

    # If no GPU factors, try standard format
    if not factors:
        standard_matches = ECMPatterns.STANDARD_FACTOR.findall(output)
        if standard_matches:
            logger.debug(f"Found {len(standard_matches)} factors via STANDARD_FACTOR pattern")
        for factor in standard_matches:
            # Look for sigma in the output (simplified approach)
            sigma_match = ECMPatterns.SIGMA_PARAM.search(output)
            sigma = sigma_match.group(1) if sigma_match else None
            factors.append((factor, sigma))

    if factors:
        logger.info(f"Parsed {len(factors)} factors from ECM output")

    return factors


def parse_ecm_output(output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse GMP-ECM output for factors and sigma.
    Optimized version using compiled patterns.

    Returns:
        Tuple of (factor, sigma) where both can be None
    """
    # Try GPU format first (most specific)
    match = ECMPatterns.GPU_FACTOR.search(output)
    if match:
        factor = match.group(1)
        sigma = match.group(2)  # May be None
        return factor, sigma

    # Try standard format
    match = ECMPatterns.STANDARD_FACTOR.search(output)
    if match:
        factor = match.group(1)
        # Look for the sigma from the "Using" line that's closest before the factor announcement
        factor_position = match.start()
        lines = output[:factor_position].split('\n')

        # Search backwards for the most recent "Using ... sigma=" line
        sigma = None
        for line in reversed(lines):
            sigma_match = re.search(r'Using.*?sigma=(?:3:)?(\d+)', line)
            if sigma_match:
                sigma = f"3:{sigma_match.group(1)}"
                break

        # If no Using line found, fallback to parameter format
        if not sigma:
            sigma_match = ECMPatterns.SIGMA_PARAM.search(output)
            sigma = sigma_match.group(1) if sigma_match else None
        return factor, sigma

    return None, None


def parse_yafu_ecm_output(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse YAFU ECM output for factors and curves completed.
    Returns list of (factor, sigma) tuples.
    """
    lines = output.split('\n')
    factors = []

    for line in lines:
        # Look for factor findings
        match = YAFUPatterns.FACTOR_FOUND.search(line)
        if match:
            factor = match.group(1)
            if factor not in [f[0] for f in factors]:
                logger.debug(f"Found factor via FACTOR_FOUND pattern: {factor}")
                factors.append((factor, None))  # YAFU doesn't report sigma
            continue

        # Check P/Q format
        match = YAFUPatterns.PQ_NOTATION.search(line)
        if match:
            factor = match.group(1)
            if factor not in [f[0] for f in factors]:
                logger.debug(f"Found factor via PQ_NOTATION pattern: {factor}")
                factors.append((factor, None))
            continue

    if factors:
        logger.info(f"Parsed {len(factors)} factors from YAFU ECM output")

    return factors


def parse_yafu_auto_factors(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse factors from YAFU automatic factorization.
    Returns list of (factor, sigma) tuples.
    """
    lines = output.split('\n')
    in_factor_section = False
    factors = []

    for line in lines:
        # Check for factor section start
        if YAFUPatterns.FACTOR_SECTION_START.search(line):
            logger.debug("Entered YAFU factor section")
            in_factor_section = True
            continue

        if in_factor_section:
            # Parse factor lines
            match = YAFUPatterns.AUTO_FACTOR.search(line)
            if match:
                factor = match.group(1)
                if factor not in [f[0] for f in factors]:
                    logger.debug(f"Found factor via AUTO_FACTOR pattern: {factor}")
                    factors.append((factor, None))
                continue

            # Handle simple number lines
            match = YAFUPatterns.SIMPLE_NUMBER.search(line)
            if match:
                factor = match.group(1)
                if factor not in [f[0] for f in factors]:
                    logger.debug(f"Found factor via SIMPLE_NUMBER pattern: {factor}")
                    factors.append((factor, None))

    if factors:
        logger.info(f"Parsed {len(factors)} factors from YAFU auto output")

    return factors


def count_ecm_steps_completed(output: str) -> int:
    """Count completed ECM steps from output."""
    return len(ECMPatterns.STEP_COMPLETED.findall(output))


def count_ecm_curves_completed(output: str) -> int:
    """
    Count completed ECM curves by looking for 'Step 2 took xxxms' messages.
    Each curve completes both Step 1 and Step 2, so Step 2 completion = curve completion.
    """
    # Try both patterns to handle different log formats
    curves = len(ECMPatterns.CURVE_COMPLETED.findall(output))
    if curves == 0:
        curves = len(ECMPatterns.CURVE_COMPLETED_ALT.findall(output))
    return curves


def get_ecm_curve_times(output: str) -> List[int]:
    """
    Extract Step 2 completion times in milliseconds.
    Returns list of times, useful for performance analysis.
    """
    times = ECMPatterns.CURVE_COMPLETED.findall(output)
    if not times:
        times = ECMPatterns.CURVE_COMPLETED_ALT.findall(output)
    return [int(time_str) for time_str in times]


def get_ecm_progress_estimate(output: str, target_curves: Optional[int] = None) -> Dict[str, Any]:
    """
    Get ECM progress estimation based on completed curves.

    Args:
        output: ECM program output
        target_curves: Expected total curves (if known)

    Returns:
        Dictionary with curve progress and timing info
    """
    completed_curves = count_ecm_curves_completed(output)
    curve_times = get_ecm_curve_times(output)

    progress = {
        'curves_completed': completed_curves,
        'steps_completed': count_ecm_steps_completed(output),
        'has_factors': bool(ECMPatterns.PRIME_FACTOR.search(output) or
                          ECMPatterns.GPU_FACTOR.search(output) or
                          ECMPatterns.STANDARD_FACTOR.search(output))
    }

    if curve_times:
        progress['avg_curve_time_ms'] = sum(curve_times) // len(curve_times)
        progress['fastest_curve_ms'] = min(curve_times)
        progress['slowest_curve_ms'] = max(curve_times)

    if target_curves and completed_curves > 0:
        progress['target_curves'] = target_curves
        progress['progress_percent'] = round((completed_curves / target_curves) * 100, 1)

        if curve_times and completed_curves < target_curves:
            remaining = target_curves - completed_curves
            avg_time_ms = sum(curve_times) // len(curve_times)
            estimated_remaining_ms = remaining * avg_time_ms
            progress['estimated_remaining_sec'] = estimated_remaining_ms // 1000

    return progress


def get_yafu_curves_progress(output: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract YAFU curve progress without overhead.
    Returns (current_curve, total_curves) or (None, None).
    """
    # Look for most recent "curve X of Y" pattern
    matches = YAFUPatterns.CURVE_PROGRESS.findall(output)
    if matches:
        current, total = matches[-1]  # Get the last/most recent progress
        return int(current), int(total)
    return None, None


def get_yafu_curves_completed(output: str) -> Optional[int]:
    """
    Extract number of completed curves from YAFU output.
    Returns the highest number found or None.
    """
    matches = YAFUPatterns.CURVES_COMPLETED.findall(output)
    if matches:
        return max(int(match) for match in matches)
    return None


def get_progress_summary(output: str, program_type: str, target_curves: Optional[int] = None) -> Dict[str, Any]:
    """
    Get lightweight progress summary without parsing full output.

    Args:
        output: Program output string
        program_type: 'ecm' or 'yafu'
        target_curves: Expected total curves (if known)

    Returns:
        Dictionary with progress metrics
    """
    summary = {'program': program_type, 'has_factors': False}

    if program_type == 'ecm':
        # Use improved ECM curve counting
        curves_completed = count_ecm_curves_completed(output)
        steps_completed = count_ecm_steps_completed(output)

        summary['curves_completed'] = curves_completed
        summary['steps_completed'] = steps_completed

        # Quick factor check without full parsing
        summary['has_factors'] = bool(ECMPatterns.PRIME_FACTOR.search(output) or
                                    ECMPatterns.GPU_FACTOR.search(output) or
                                    ECMPatterns.STANDARD_FACTOR.search(output))

        # Add progress percentage if target known
        if target_curves and curves_completed > 0:
            summary['target_curves'] = target_curves
            summary['progress_percent'] = round((curves_completed / target_curves) * 100, 1)

        # Quick timing info
        curve_times = get_ecm_curve_times(output)
        if curve_times:
            summary['avg_curve_time_ms'] = sum(curve_times) // len(curve_times)

    elif program_type == 'yafu':
        current, total = get_yafu_curves_progress(output)
        completed = get_yafu_curves_completed(output)

        summary['current_curve'] = current
        summary['total_curves'] = total
        summary['curves_completed'] = completed

        if current and total:
            summary['progress_percent'] = round((current / total) * 100, 1)

        # Quick factor check
        summary['has_factors'] = bool(YAFUPatterns.FACTOR_FOUND.search(output) or
                                    YAFUPatterns.PQ_NOTATION.search(output) or
                                    YAFUPatterns.FACTOR_SECTION_START.search(output))

    return summary


def extract_program_version(output: str, program_type: str) -> str:
    """
    Extract program version from help output.
    
    Args:
        output: Program help/version output
        program_type: Either 'ecm' or 'yafu'
    
    Returns:
        Version string or 'unknown'
    """
    if program_type == 'ecm':
        match = re.search(r'GMP-ECM (\d+\.\d+(?:\.\d+)?)', output)
        return match.group(1) if match else "unknown"
    elif program_type == 'yafu':
        match = re.search(r'YAFU Version (\d+\.\d+(?:\.\d+)?)', output)
        return match.group(1) if match else "unknown"
    else:
        return "unknown"
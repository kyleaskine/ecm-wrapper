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
    YAFU_ECM = None     # No timeout (for aliquot sequences)
    YAFU_AUTO = None    # No timeout (for aliquot sequences)
    CADO_NFS = None     # No timeout (for aliquot sequences)


# Compiled regex patterns for performance
class ECMPatterns:
    """Compiled regex patterns for ECM output parsing."""

    # Pattern 1: Prime factor announcements (most reliable - highest priority)
    # Example: "Found prime factor of 14 digits: 59460190057621"
    PRIME_FACTOR = re.compile(r'Found prime factor of \d+ digits: (\d+)', re.IGNORECASE)

    # Pattern 2: GPU format with sigma (very specific)
    # Example: "GPU: factor 12345 found in Step 1 with curve 0 (-sigma 3:2126921240)"
    GPU_FACTOR = re.compile(r'GPU: factor (\d+) found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?', re.IGNORECASE)

    # Pattern 3: Standard format (with optional asterisk prefix for stage 2)
    # Example: "Factor found in step 1: 67280421310721"
    # Example: "********** Factor found in step 2: 154848006894803752593902015592419621459239"
    STANDARD_FACTOR = re.compile(r'\**\s*Factor found in step \d+: (\d+)', re.IGNORECASE)

    # Sigma parameter extraction
    SIGMA_PARAM = re.compile(r'-sigma (3:\d+|\d+)')

    # Step completion tracking
    STEP_COMPLETED = re.compile(r'Step 1 took')

    # Curve completion tracking - "Step 2 took xxxms" indicates curve completion
    CURVE_COMPLETED = re.compile(r'Step 2 took (\d+)ms')

    # Alternative curve completion patterns
    CURVE_COMPLETED_ALT = re.compile(r'ECM: Step 2 took (\d+)ms')

    # Sigma extraction patterns (used in ecm-wrapper.py line-by-line parsing)
    SIGMA_COLON_FORMAT = re.compile(r'sigma=([1-3]:\d+)')
    SIGMA_DASH_FORMAT = re.compile(r'-sigma (3:\d+|\d+)')

    # Curve count extraction from GPU output
    CURVE_COUNT = re.compile(r'\((\d+) curves\)')

    # Resume file parsing patterns
    RESUME_N_PATTERN = re.compile(r'N=(\d+)')
    RESUME_B1_PATTERN = re.compile(r'B1=(\d+)')
    RESUME_SIGMA_PATTERN = re.compile(r'SIGMA=(\d+)')

    # Version detection
    GMP_ECM_VERSION = re.compile(r'GMP-ECM (\d+\.\d+(?:\.\d+)?)')

    # Dynamic GPU factor search pattern generator
    @staticmethod
    def gpu_factor_search_pattern(factor):
        """Generate compiled pattern for specific factor GPU search."""
        return re.compile(rf'GPU: factor {re.escape(factor)} found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?')


class YAFUPatterns:
    """Compiled regex patterns for YAFU output parsing."""

    # Direct factor announcements - handle both "found factor" and "found prpN factor"
    FACTOR_FOUND = re.compile(r'found\s+(?:prp\d+\s+)?factor\s*[=:]\s*(\d+)', re.IGNORECASE)

    # P/Q notation: "P39 = 123456789012345678901234567890123456789"
    PQ_NOTATION = re.compile(r'[PQ]\d+\s*=\s*(\d+)')

    # Curves completion tracking
    CURVES_COMPLETED = re.compile(r'completed?\s+(\d+)\s+curves?', re.IGNORECASE)

    # Curve progress: "curve X of Y"
    CURVE_PROGRESS = re.compile(r'curve\s+(\d+)\s+of\s+(\d+)', re.IGNORECASE)

    # Factor section markers
    FACTOR_SECTION_START = re.compile(r'\*{3,}factors? found\*{3,}', re.IGNORECASE)

    # Factor lines in auto mode: "P15 = 856395168938929"
    # Only match P (prime) factors, not C (composite) cofactors
    AUTO_FACTOR = re.compile(r'P\d+\s*=\s*(\d+)')

    # Simple number lines (fallback)
    SIMPLE_NUMBER = re.compile(r'^\s*(\d+)\s*$')

    # Version detection
    YAFU_VERSION = re.compile(r'YAFU Version (\d+\.\d+(?:\.\d+)?)')

    # Sigma extraction for YAFU
    SIGMA_USING_FORMAT = re.compile(r'Using.*?sigma=(?:3:)?(\d+)')


def extract_sigma_for_factor(output: str, factor: str, factor_position: Optional[int] = None) -> Optional[str]:
    """
    Extract sigma value associated with a factor using fallback chain.

    Args:
        output: Full program output
        factor: Factor to find sigma for
        factor_position: Optional position in output where factor was found

    Returns:
        Sigma string (e.g., "3:12345") or None
    """
    # Strategy 1: GPU format with specific factor
    gpu_match = ECMPatterns.gpu_factor_search_pattern(factor).search(output)
    if gpu_match and gpu_match.group(1):
        return f"3:{gpu_match.group(1)}"

    # Strategy 2: Look backwards from factor position for "Using ... sigma=" line
    if factor_position is not None:
        lines = output[:factor_position].split('\n')
        for line in reversed(lines):
            sigma_match = YAFUPatterns.SIGMA_USING_FORMAT.search(line)
            if sigma_match:
                return f"3:{sigma_match.group(1)}"

    # Strategy 3: Global sigma parameter search
    sigma_match = ECMPatterns.SIGMA_PARAM.search(output)
    if sigma_match:
        return sigma_match.group(1)

    return None


def _extract_factors_with_patterns(output: str) -> List[Tuple[str, Optional[str], str]]:
    """
    Internal unified function: Extract all factors from output using all available patterns.

    This is the single source of truth for factor extraction, ensuring both
    parse_ecm_output() and parse_ecm_output_multiple() use identical logic.

    Strategy:
    - Start with PRIME_FACTOR matches (GMP-ECM's primality-tested results)
    - Add GPU_FACTOR/STANDARD_FACTOR matches that are NOT products of known primes
    - This avoids submitting composite factors like p1*p2 while keeping all legitimate factors

    Args:
        output: ECM program output

    Returns:
        List of (factor, sigma, pattern_name) tuples
        Empty list if no factors found
    """
    factors = []
    seen_factors = set()  # Deduplicate factors found by multiple patterns

    # Step 1: Collect all factors from GPU and standard patterns (these have sigma values)
    # Use dict to avoid duplicates: factor -> (sigma, pattern_name)
    gpu_and_standard_map = {}

    # Pattern 1: GPU format (includes sigma in match)
    for match in ECMPatterns.GPU_FACTOR.finditer(output):
        factor = match.group(1)
        if factor not in gpu_and_standard_map:
            sigma = f"3:{match.group(2)}" if match.group(2) else None
            gpu_and_standard_map[factor] = (sigma, "GPU_FACTOR")
            logger.debug(f"Found factor via GPU_FACTOR: {factor} with sigma {sigma}")

    # Pattern 2: Standard format
    for match in ECMPatterns.STANDARD_FACTOR.finditer(output):
        factor = match.group(1)
        if factor not in gpu_and_standard_map:
            sigma = extract_sigma_for_factor(output, factor, match.start())
            gpu_and_standard_map[factor] = (sigma, "STANDARD_FACTOR")
            logger.debug(f"Found factor via STANDARD_FACTOR: {factor} with sigma {sigma}")

    # Step 2: Process prime factor announcements (GMP-ECM's primality-tested results)
    # For each prime, find which GPU/STANDARD factor it divides and use that sigma
    used_gpu_factors = set()  # Track which GPU factors have been matched to primes

    for match in ECMPatterns.PRIME_FACTOR.finditer(output):
        prime = match.group(1)
        if prime in seen_factors:
            continue

        prime_int = int(prime)
        matched_sigma = None

        # Find the GPU/STANDARD factor that this prime divides
        # Strategy: Prefer exact matches first, then divisibility matches
        for gpu_factor, (gpu_sigma, pattern_name) in gpu_and_standard_map.items():
            gpu_factor_int = int(gpu_factor)
            # Exact match - this GPU report is the prime itself
            if gpu_factor_int == prime_int:
                matched_sigma = gpu_sigma
                used_gpu_factors.add(gpu_factor)  # Mark as used
                logger.debug(f"Prime {prime} exact match with GPU factor, using sigma {gpu_sigma}")
                break

        # If no exact match, look for a composite that contains this prime
        if matched_sigma is None:
            for gpu_factor, (gpu_sigma, pattern_name) in gpu_and_standard_map.items():
                gpu_factor_int = int(gpu_factor)
                if gpu_factor_int > prime_int and gpu_factor_int % prime_int == 0:
                    matched_sigma = gpu_sigma
                    logger.debug(f"Prime {prime} divides composite {gpu_factor}, using sigma {gpu_sigma}")
                    break

        # If no match found, try the fallback extraction
        if matched_sigma is None:
            matched_sigma = extract_sigma_for_factor(output, prime, match.start())
            logger.debug(f"Prime {prime} using fallback sigma extraction: {matched_sigma}")

        factors.append((prime, matched_sigma, "PRIME_FACTOR"))
        seen_factors.add(prime)
        logger.debug(f"Added prime factor: {prime} with sigma {matched_sigma}")

    # Step 3: Add remaining GPU/STANDARD factors that aren't composites or already used
    known_primes = [int(f[0]) for f in factors]

    for candidate_factor, (candidate_sigma, pattern_name) in gpu_and_standard_map.items():
        # Skip if already used (matched to a PRIME_FACTOR)
        if candidate_factor in used_gpu_factors or candidate_factor in seen_factors:
            logger.debug(f"Skipping {candidate_factor} (already used)")
            continue

        candidate_int = int(candidate_factor)

        # Check if this candidate is divisible by any known prime (making it composite)
        is_composite = False
        for prime in known_primes:
            if candidate_int > prime and candidate_int % prime == 0:
                cofactor = candidate_int // prime
                is_composite = True
                logger.debug(f"Filtering {candidate_factor} (composite: {prime} × {cofactor})")
                break

        # If not divisible by any known prime, include it
        if not is_composite:
            factors.append((candidate_factor, candidate_sigma, pattern_name))
            seen_factors.add(candidate_factor)
            known_primes.append(candidate_int)  # Add to known primes for future checks
            logger.debug(f"Added GPU/STANDARD factor: {candidate_factor}")

    return factors


def parse_ecm_output_multiple(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse GMP-ECM output for multiple prime factors only.
    Uses ECM's own "Found prime factor" identification to avoid composite factors.

    Returns:
        List of (factor, sigma) tuples. Empty list if no factors found.
    """
    factors_with_patterns = _extract_factors_with_patterns(output)

    # Strip pattern name and return just (factor, sigma) tuples
    factors = [(f[0], f[1]) for f in factors_with_patterns]

    if factors:
        logger.info(f"Parsed {len(factors)} factors from ECM output")

    return factors


def parse_ecm_output(output: str, debug: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse GMP-ECM output for first factor and sigma.

    Now uses the same comprehensive pattern matching as parse_ecm_output_multiple(),
    ensuring consistent behavior across all parsing operations.

    Args:
        output: ECM program output
        debug: If True, log output preview when no factors found (for debugging)

    Returns:
        Tuple of (factor, sigma) where both can be None
    """
    factors = _extract_factors_with_patterns(output)

    if factors:
        factor, sigma, pattern = factors[0]
        logger.info(f"Parsed factor {factor} using pattern {pattern}")
        return factor, sigma

    # Debug output when no factors found
    if debug:
        logger.warning("No factors found in output")
        logger.debug(f"Output preview (first 500 chars):\n{output[:500]}")
        logger.debug(f"Output preview (last 500 chars):\n{output[-500:]}")

    return None, None


def parse_yafu_ecm_output(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse YAFU ECM output for factors.
    Returns list of (factor, sigma) tuples.

    Note: YAFU lists factors multiple times to indicate multiplicity (exponents).
    We preserve duplicates as they represent the complete prime factorization.

    Important: YAFU reports factors in TWO places:
    1. During execution: "ecm: found prp15 factor = X"
    2. Final summary: "***factors found***" section
    We only parse the final summary to avoid duplicates.
    """
    lines = output.split('\n')
    factors: List[Tuple[str, Optional[str]]] = []
    in_factor_section = False

    for line in lines:
        # Check for factor section start (same as auto mode)
        if YAFUPatterns.FACTOR_SECTION_START.search(line):
            logger.debug("Entered YAFU factor section")
            in_factor_section = True
            continue

        if in_factor_section:
            # Parse factor lines - keep ALL occurrences (multiplicity)
            match = YAFUPatterns.AUTO_FACTOR.search(line)
            if match:
                factor = match.group(1)
                logger.debug(f"Found factor via AUTO_FACTOR pattern: {factor}")
                factors.append((factor, None))  # YAFU doesn't report sigma
                continue

            # Handle simple number lines - keep ALL occurrences (multiplicity)
            match = YAFUPatterns.SIMPLE_NUMBER.search(line)
            if match:
                factor = match.group(1)
                logger.debug(f"Found factor via SIMPLE_NUMBER pattern: {factor}")
                factors.append((factor, None))

    if factors:
        logger.info(f"Parsed {len(factors)} factors from YAFU ECM output")

    return factors


def parse_yafu_auto_factors(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse factors from YAFU automatic factorization.
    Returns list of (factor, sigma) tuples.

    Note: YAFU lists factors multiple times to indicate multiplicity (exponents).
    We preserve duplicates as they represent the complete prime factorization.
    """
    lines = output.split('\n')
    in_factor_section = False
    factors: List[Tuple[str, Optional[str]]] = []

    for line in lines:
        # Check for factor section start
        if YAFUPatterns.FACTOR_SECTION_START.search(line):
            logger.debug("Entered YAFU factor section")
            in_factor_section = True
            continue

        if in_factor_section:
            # Parse factor lines - keep ALL occurrences (multiplicity)
            match = YAFUPatterns.AUTO_FACTOR.search(line)
            if match:
                factor = match.group(1)
                logger.debug(f"Found factor via AUTO_FACTOR pattern: {factor}")
                factors.append((factor, None))
                continue

            # Handle simple number lines - keep ALL occurrences (multiplicity)
            match = YAFUPatterns.SIMPLE_NUMBER.search(line)
            if match:
                factor = match.group(1)
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

    progress: Dict[str, Any] = {
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
        match = ECMPatterns.GMP_ECM_VERSION.search(output)
        return match.group(1) if match else "unknown"
    elif program_type == 'yafu':
        match = YAFUPatterns.YAFU_VERSION.search(output)
        return match.group(1) if match else "unknown"
    else:
        return "unknown"
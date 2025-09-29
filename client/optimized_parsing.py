#!/usr/bin/env python3
"""
High-performance optimized parsing utilities.
Consolidates and optimizes all regex patterns with compiled caching.
"""
import re
import logging
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# Timeout constants (seconds)
class Timeouts:
    ECM_DEFAULT = 3600  # 1 hour
    YAFU_ECM = 7200     # 2 hours
    YAFU_AUTO = 14400   # 4 hours for NFS

class OptimizedPatterns:
    """
    Centralized, pre-compiled regex patterns for maximum performance.
    All patterns are compiled once at import time.
    """

    # ECM Patterns - compiled once for performance
    ECM_GPU_FACTOR = re.compile(r'GPU: factor (\d+) found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?')
    ECM_STANDARD_FACTOR = re.compile(r'Factor found in step \d+: (\d+)')
    ECM_PRIME_FACTOR = re.compile(r'Found prime factor of \d+ digits: (\d+)')
    ECM_SIGMA_PARAM = re.compile(r'-sigma (3:\d+|\d+)')
    ECM_STEP_COMPLETED = re.compile(r'Step 1 took')
    ECM_CURVE_COMPLETED = re.compile(r'Step 2 took (\d+)ms')
    ECM_CURVE_COMPLETED_ALT = re.compile(r'ECM: Step 2 took (\d+)ms')
    ECM_GPU_CURVE_COUNT = re.compile(r'\((\d+) curves\)')

    # YAFU Patterns - compiled once for performance
    YAFU_FACTOR_FOUND = re.compile(r'found factor[:\s]+(\d+)', re.IGNORECASE)
    YAFU_PQ_NOTATION = re.compile(r'[PQ]\d+\s*=\s*(\d+)')
    YAFU_CURVES_COMPLETED = re.compile(r'completed?\s+(\d+)\s+curves?', re.IGNORECASE)
    YAFU_CURVE_PROGRESS = re.compile(r'curve\s+(\d+)\s+of\s+(\d+)', re.IGNORECASE)
    YAFU_FACTOR_SECTION_START = re.compile(r'\*{3,}factors? found\*{3,}', re.IGNORECASE)
    YAFU_AUTO_FACTOR = re.compile(r'[PCQ]\d+\s*=\s*(\d+)')
    YAFU_SIMPLE_NUMBER = re.compile(r'^\s*(\d+)\s*$')

    # Version extraction patterns
    VERSION_ECM = re.compile(r'ecm\s+([\d.]+)', re.IGNORECASE)
    VERSION_YAFU = re.compile(r'yafu\s+([\d.]+)', re.IGNORECASE)

class StreamingParser:
    """
    High-performance streaming parser for real-time factor detection.
    Optimized for minimal latency in factor discovery.
    """

    def __init__(self):
        self.factors_found = []
        self.curves_completed = 0
        self.last_sigma = None

    def parse_line_ecm(self, line: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Parse single ECM output line for factors.
        Returns (factor, sigma) tuple if found, None otherwise.
        Optimized for speed - checks most common patterns first.
        """
        # Check GPU format first (most common in modern ECM)
        match = OptimizedPatterns.ECM_GPU_FACTOR.match(line)
        if match:
            factor = match.group(1)
            sigma = f"3:{match.group(2)}" if match.group(2) else None
            return (factor, sigma)

        # Check prime factor announcements
        match = OptimizedPatterns.ECM_PRIME_FACTOR.match(line)
        if match:
            return (match.group(1), self.last_sigma)

        # Check standard factor format
        match = OptimizedPatterns.ECM_STANDARD_FACTOR.match(line)
        if match:
            return (match.group(1), None)

        # Track sigma for context
        sigma_match = OptimizedPatterns.ECM_SIGMA_PARAM.search(line)
        if sigma_match:
            self.last_sigma = sigma_match.group(1)

        # Track curve progress
        if OptimizedPatterns.ECM_STEP_COMPLETED.search(line):
            self.curves_completed += 1

        return None

    def parse_line_yafu(self, line: str) -> Optional[str]:
        """
        Parse single YAFU output line for factors.
        Returns factor string if found, None otherwise.
        """
        # Check direct factor announcements first
        match = OptimizedPatterns.YAFU_FACTOR_FOUND.search(line)
        if match:
            return match.group(1)

        # Check P/Q notation
        match = OptimizedPatterns.YAFU_PQ_NOTATION.search(line)
        if match and len(match.group(1)) > 1:  # Skip trivial factors
            return match.group(1)

        return None

@lru_cache(maxsize=128)
def extract_program_version(output: str, program: str) -> str:
    """
    Extract program version with LRU caching for performance.

    Args:
        output: Program help/version output
        program: Program name ('ecm' or 'yafu')

    Returns:
        Version string or 'unknown'
    """
    if program.lower() == 'ecm':
        match = OptimizedPatterns.VERSION_ECM.search(output)
    elif program.lower() == 'yafu':
        match = OptimizedPatterns.VERSION_YAFU.search(output)
    else:
        return 'unknown'

    return match.group(1) if match else 'unknown'

def parse_ecm_output(output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Optimized ECM output parsing for single factor.
    Returns (factor, sigma) tuple.
    """
    # Check prime factors first (most reliable)
    match = OptimizedPatterns.ECM_PRIME_FACTOR.search(output)
    if match:
        factor = match.group(1)
        # Find corresponding sigma
        sigma_line_match = re.search(rf'GPU: factor {factor} found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?', output)
        sigma = f"3:{sigma_line_match.group(1)}" if sigma_line_match and sigma_line_match.group(1) else None
        return (factor, sigma)

    # Check GPU format
    match = OptimizedPatterns.ECM_GPU_FACTOR.search(output)
    if match:
        factor = match.group(1)
        sigma = f"3:{match.group(2)}" if match.group(2) else None
        return (factor, sigma)

    # Check standard format
    match = OptimizedPatterns.ECM_STANDARD_FACTOR.search(output)
    if match:
        return (match.group(1), None)

    return (None, None)

def parse_ecm_output_multiple(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Optimized ECM output parsing for multiple factors.
    Returns list of (factor, sigma) tuples.
    """
    factors = []

    # Prime factors first
    prime_matches = OptimizedPatterns.ECM_PRIME_FACTOR.findall(output)
    if prime_matches:
        for factor in prime_matches:
            # Find sigma for this factor
            sigma_match = re.search(rf'GPU: factor {factor} found in Step \d+ with curve \d+(?: \(-sigma 3:(\d+)\))?', output)
            sigma = f"3:{sigma_match.group(1)}" if sigma_match and sigma_match.group(1) else None
            factors.append((factor, sigma))
        return factors

    # GPU format fallback
    gpu_matches = OptimizedPatterns.ECM_GPU_FACTOR.findall(output)
    for match in gpu_matches:
        factor = match[0]
        sigma = f"3:{match[1]}" if match[1] else None
        factors.append((factor, sigma))

    return factors

def parse_yafu_ecm_output(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Optimized YAFU ECM output parsing.
    Returns list of (factor, sigma) tuples.
    """
    factors = []

    # Direct factor announcements
    factor_matches = OptimizedPatterns.YAFU_FACTOR_FOUND.findall(output)
    for factor in factor_matches:
        if len(factor) > 1:  # Skip trivial factors
            factors.append((factor, None))

    # P/Q notation
    pq_matches = OptimizedPatterns.YAFU_PQ_NOTATION.findall(output)
    for factor in pq_matches:
        if len(factor) > 1:  # Skip trivial factors
            factors.append((factor, None))

    return factors

def parse_yafu_auto_factors(output: str) -> List[Tuple[str, Optional[str]]]:
    """
    Optimized YAFU auto mode output parsing.
    Returns list of (factor, sigma) tuples.
    """
    factors = []

    # Look for factor section
    lines = output.split('\n')
    in_factor_section = False

    for line in lines:
        # Check for factor section start
        if OptimizedPatterns.YAFU_FACTOR_SECTION_START.search(line):
            in_factor_section = True
            continue

        if in_factor_section:
            # Check for factor lines
            match = OptimizedPatterns.YAFU_AUTO_FACTOR.search(line)
            if match:
                factor = match.group(1)
                if len(factor) > 1:  # Skip trivial factors
                    factors.append((factor, None))

            # Simple number lines
            match = OptimizedPatterns.YAFU_SIMPLE_NUMBER.match(line)
            if match:
                factor = match.group(1)
                if len(factor) > 1:  # Skip trivial factors
                    factors.append((factor, None))

    return factors

def count_ecm_steps_completed(output: str) -> int:
    """
    Optimized ECM step counting.
    Returns number of completed curves.
    """
    return len(OptimizedPatterns.ECM_STEP_COMPLETED.findall(output))

def extract_gpu_curve_count(output: str) -> int:
    """
    Extract actual curve count from GPU ECM output.
    Returns the number of curves that were actually run, or 0 if not found.

    Example output: "Using B1=50000, B2=0, sigma=3:1436497730-1436500801 (3072 curves)"
    """
    match = OptimizedPatterns.ECM_GPU_CURVE_COUNT.search(output)
    return int(match.group(1)) if match else 0

# Backward compatibility aliases
ECMPatterns = OptimizedPatterns
YAFUPatterns = OptimizedPatterns
"""
ECM Execution Engine

This module provides clean, config-based execution methods for ECM factorization.
Separates command building, execution, and result processing into focused functions.
"""

import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from .ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, FactorResult
from .parsing_utils import parse_ecm_output_multiple


logger = logging.getLogger(__name__)


class ECMCommandBuilder:
    """Builds GMP-ECM command line arguments from configuration."""

    @staticmethod
    def build_ecm_command(ecm_path: str, config: ECMConfig) -> List[str]:
        """
        Build GMP-ECM command from ECMConfig.

        Args:
            ecm_path: Path to GMP-ECM binary
            config: ECM configuration

        Returns:
            Command as list of strings
        """
        cmd = [ecm_path]

        # Verbosity
        if config.verbose:
            cmd.append('-v')

        # Parametrization (ECM only)
        if config.parametrization is not None:
            cmd.extend(['-param', str(config.parametrization)])

        # Sigma (ECM only)
        if config.sigma is not None:
            cmd.extend(['-sigma', str(config.sigma)])

        # Curves
        cmd.extend(['-c', str(config.curves)])

        # B1 parameter
        cmd.append(str(config.b1))

        # B2 parameter (optional)
        if config.b2 is not None:
            cmd.append(str(config.b2))

        return cmd

    @staticmethod
    def build_pm1_command(ecm_path: str, b1: int, b2: Optional[int] = None,
                         verbose: bool = False) -> List[str]:
        """Build P-1 command."""
        cmd = [ecm_path, '-pm1']
        if verbose:
            cmd.append('-v')
        cmd.append(str(b1))
        if b2 is not None:
            cmd.append(str(b2))
        return cmd

    @staticmethod
    def build_gpu_command(ecm_path: str, config: ECMConfig,
                         gpu_device: Optional[int] = None,
                         gpu_curves: Optional[int] = None) -> List[str]:
        """Build GPU-accelerated ECM command."""
        cmd = [ecm_path, '-gpu']

        if gpu_device is not None:
            cmd.extend(['-gpudevice', str(gpu_device)])

        if gpu_curves is not None:
            cmd.extend(['-gpucurves', str(gpu_curves)])

        if config.verbose:
            cmd.append('-v')

        if config.parametrization is not None:
            cmd.extend(['-param', str(config.parametrization)])

        if config.sigma is not None:
            cmd.extend(['-sigma', str(config.sigma)])

        cmd.extend(['-c', str(config.curves)])
        cmd.append(str(config.b1))

        if config.b2 is not None:
            cmd.append(str(config.b2))

        return cmd


class ECMExecutor:
    """Executes ECM commands and processes results."""

    def __init__(self, config: Dict[str, Any], run_subprocess_func):
        """
        Initialize executor.

        Args:
            config: Configuration dictionary (from BaseWrapper)
            run_subprocess_func: Function to run subprocess (from BaseWrapper)
        """
        self.config = config
        self.run_subprocess = run_subprocess_func
        self.ecm_path = config['programs']['gmp_ecm']['path']

    def execute_ecm(self, ecm_config: ECMConfig) -> FactorResult:
        """
        Execute ECM with given configuration.

        Args:
            ecm_config: ECM configuration object

        Returns:
            FactorResult with factors found and execution metadata
        """
        result = FactorResult()
        start_time = time.time()

        # Build command
        cmd = ECMCommandBuilder.build_ecm_command(self.ecm_path, ecm_config)

        logger.info(
            "Running ECM on %d-digit number: B1=%d, curves=%d, param=%d",
            len(ecm_config.composite), ecm_config.b1,
            ecm_config.curves, ecm_config.parametrization
        )

        try:
            # Execute subprocess
            output = self.run_subprocess(
                cmd,
                input=ecm_config.composite,
                timeout=ecm_config.timeout,
                verbose=ecm_config.verbose
            )

            result.raw_output = output
            result.curves_run = ecm_config.curves
            result.execution_time = time.time() - start_time

            # Parse output for factors
            factors_found = parse_ecm_output_multiple(output)

            for factor_str, sigma_from_parse in factors_found:
                # Each factor comes with its sigma from parsing, fallback to extraction if needed
                sigma = sigma_from_parse or self._extract_sigma_for_factor(output, factor_str)
                result.add_factor(factor_str, sigma)

            if result.factors:
                logger.info("Found %d factor(s): %s", len(result.factors), result.factors)
            else:
                logger.info("No factors found in %d curves", ecm_config.curves)

        except Exception as e:
            logger.error("ECM execution failed: %s", e)
            result.error_message = str(e)

        return result

    def execute_ecm_batch(self, ecm_config: ECMConfig,
                         batch_size: int = 50) -> FactorResult:
        """
        Execute ECM in batches to reduce subprocess overhead.

        Args:
            ecm_config: ECM configuration
            batch_size: Curves per batch (default: 50)

        Returns:
            Combined FactorResult from all batches
        """
        combined_result = FactorResult()
        remaining_curves = ecm_config.curves
        start_time = time.time()

        while remaining_curves > 0:
            # Create batch config
            curves_this_batch = min(remaining_curves, batch_size)
            batch_config = ECMConfig(
                composite=ecm_config.composite,
                b1=ecm_config.b1,
                b2=ecm_config.b2,
                curves=curves_this_batch,
                sigma=ecm_config.sigma,
                parametrization=ecm_config.parametrization,
                threads=ecm_config.threads,
                verbose=ecm_config.verbose,
                timeout=ecm_config.timeout
            )

            # Execute batch
            batch_result = self.execute_ecm(batch_config)

            # Combine results
            for factor, sigma in batch_result.factor_sigma_pairs:
                combined_result.add_factor(factor, sigma)

            combined_result.curves_run += batch_result.curves_run
            if batch_result.raw_output:
                combined_result.raw_output = (
                    (combined_result.raw_output or "") + "\n" + batch_result.raw_output
                )

            # Stop if factor found
            if batch_result.success:
                logger.info("Factor found, stopping batch execution")
                break

            remaining_curves -= curves_this_batch

        combined_result.execution_time = time.time() - start_time
        return combined_result

    def _extract_sigma_for_factor(self, output: str, factor: str) -> Optional[str]:
        """
        Extract sigma value for a specific factor from ECM output.

        Args:
            output: Raw ECM output
            factor: Factor string to find sigma for

        Returns:
            Sigma as string, or None if not found
        """
        import re

        # Look for lines like: "Using B1=50000, B2=5000000, sigma=3:123456"
        # or "Factor found in step 1: 123 (sigma=3:456789)"
        for line in output.split('\n'):
            if factor in line:
                # Try to extract sigma from this line or nearby lines
                sigma_match = re.search(r'sigma[=:](\d+:)?\s*(\d+)', line, re.IGNORECASE)
                if sigma_match:
                    if sigma_match.group(1):  # Has parametrization prefix
                        return sigma_match.group(1).rstrip(':') + ':' + sigma_match.group(2)
                    return sigma_match.group(2)

        return None


class ECMResultConverter:
    """Converts FactorResult to legacy dict format for backward compatibility."""

    @staticmethod
    def to_legacy_dict(result: FactorResult, composite: str, b1: int,
                      b2: Optional[int], method: str = "ecm") -> Dict[str, Any]:
        """
        Convert FactorResult to legacy dictionary format.

        Args:
            result: FactorResult object
            composite: Original composite number
            b1: B1 parameter used
            b2: B2 parameter used
            method: Method name

        Returns:
            Dictionary in legacy format
        """
        legacy_dict = {
            'composite': composite,
            'b1': b1,
            'b2': b2,
            'curves_requested': result.curves_run,
            'curves_completed': result.curves_run,
            'factor_found': result.factors[0] if result.factors else None,
            'factors_found': result.factors if result.factors else [],
            'raw_output': result.raw_output or "",
            'execution_time': result.execution_time,
            'method': method,
            'success': result.success,
            'parametrization': None  # Can be added if needed
        }

        return legacy_dict

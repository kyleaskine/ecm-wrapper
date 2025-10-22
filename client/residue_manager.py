"""
Residue File Manager Utility

Provides a unified interface for ECM residue file operations including
parsing metadata, splitting files for parallel processing, and correlating
factors to sigma values.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List

from parsing_utils import ECMPatterns

logger = logging.getLogger(__name__)


class ResidueFileManager:
    """
    Manage ECM residue file operations with consistent error handling.

    This utility consolidates residue file I/O patterns used across the ECM wrapper,
    particularly for two-stage ECM processing with residue files.
    """

    def __init__(self):
        """Initialize residue file manager."""
        self.logger = logging.getLogger(f"{__name__}.ResidueFileManager")

    def parse_metadata(
        self, file_path: str
    ) -> Optional[Tuple[str, int, int]]:
        """
        Parse metadata from ECM residue file.

        Extracts composite number (N), B1 parameter, and curve count from
        residue file headers.

        Args:
            file_path: Path to the residue file

        Returns:
            Tuple of (composite, b1, curve_count) or None if parsing fails
            - composite: The composite number being factored (as string)
            - b1: The B1 smoothness bound used
            - curve_count: Number of curves in the residue file
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Residue file not found: {file_path}")
            return None

        try:
            composite = None
            b1 = None
            curve_count = 0

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # Extract composite number
                    if not composite:
                        n_match = ECMPatterns.RESUME_N_PATTERN.search(line)
                        if n_match:
                            composite = n_match.group(1)

                    # Extract B1 parameter
                    if not b1:
                        b1_match = ECMPatterns.RESUME_B1_PATTERN.search(line)
                        if b1_match:
                            b1 = int(b1_match.group(1))

                    # Count sigma values (one per curve)
                    if ECMPatterns.RESUME_SIGMA_PATTERN.search(line):
                        curve_count += 1

            # Validate we found all required metadata
            if composite and b1 and curve_count > 0:
                self.logger.info(
                    f"Parsed residue file {file_path}: "
                    f"composite={composite[:20]}..., b1={b1}, curves={curve_count}"
                )
                return (composite, b1, curve_count)
            else:
                self.logger.warning(
                    f"Incomplete metadata in residue file {file_path}: "
                    f"composite={bool(composite)}, b1={bool(b1)}, curves={curve_count}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Error parsing residue file {file_path}: {e}")
            return None

    def split_into_chunks(
        self, file_path: str, num_chunks: int, output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Split residue file into chunks for parallel Stage 2 processing.

        Each chunk contains a subset of the curves from the original residue file,
        preserving the header information (N, B1, METHOD=ECM, etc.).

        Args:
            file_path: Path to the residue file to split
            num_chunks: Number of chunks to create
            output_dir: Directory for chunk files (default: same as input file)

        Returns:
            List of chunk file paths created, or empty list on failure
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Residue file not found: {file_path}")
            return []

        if num_chunks < 1:
            self.logger.warning(f"Invalid num_chunks: {num_chunks}")
            return []

        try:
            # Determine output directory
            if output_dir is None:
                output_dir = os.path.dirname(file_path)
            os.makedirs(output_dir, exist_ok=True)

            # Read entire residue file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Detect format: GPU format (single-line) vs old format (multi-line)
            # GPU format: each line has METHOD=...; PARAM=...; SIGMA=...; etc
            # Old format: separate N=, B1=, SIGMA= lines
            is_gpu_format = any('METHOD=ECM' in line and 'SIGMA=' in line and ';' in line for line in lines[:5])
            self.logger.debug(f"Detected {'GPU' if is_gpu_format else 'CPU'} residue file format")

            header_lines: List[str] = []
            curve_blocks: List[List[str]] = []

            if is_gpu_format:
                # GPU format: each line is a complete curve
                for line in lines:
                    if ECMPatterns.RESUME_SIGMA_PATTERN.search(line):
                        curve_blocks.append([line])
            else:
                # Old format: multi-line curves
                current_block: List[str] = []
                for line in lines:
                    # Header lines (N, B1, METHOD, etc.)
                    if any(marker in line for marker in ['N=', 'B1=', 'METHOD=ECM', 'CHECKSUM=']):
                        header_lines.append(line)
                    # Sigma line starts a new curve block
                    elif ECMPatterns.RESUME_SIGMA_PATTERN.search(line):
                        if current_block:
                            curve_blocks.append(current_block)
                        current_block = [line]
                    # Continuation of current curve block
                    elif current_block:
                        current_block.append(line)

                # Add final block
                if current_block:
                    curve_blocks.append(current_block)

            if not curve_blocks:
                self.logger.warning(f"No curve blocks found in {file_path}")
                # Debug: show first 20 lines of the file
                self.logger.warning(f"File content (first 20 lines):")
                for i, line in enumerate(lines[:20], 1):
                    self.logger.warning(f"  Line {i}: {line.rstrip()}")
                return []

            # Calculate curves per chunk
            total_curves = len(curve_blocks)
            curves_per_chunk = max(1, total_curves // num_chunks)
            # Handle remainder curves
            remainder = total_curves % num_chunks

            self.logger.info(
                f"Splitting {total_curves} curves into {num_chunks} chunks "
                f"(~{curves_per_chunk} curves per chunk)"
            )

            # Create chunk files - limit to num_chunks
            chunk_files = []
            base_name = Path(file_path).stem
            start_idx = 0

            for chunk_idx in range(num_chunks):
                # Calculate chunk size - distribute remainder across first chunks
                chunk_size = curves_per_chunk + (1 if chunk_idx < remainder else 0)

                # Get curves for this chunk
                chunk_curves = curve_blocks[start_idx:start_idx + chunk_size]

                if not chunk_curves:
                    break  # No more curves to process

                chunk_file = os.path.join(output_dir, f"{base_name}_chunk{chunk_idx}.res")

                # Write chunk file with header + curves
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    # Write header
                    f.writelines(header_lines)
                    # Write curve blocks
                    for block in chunk_curves:
                        f.writelines(block)

                chunk_files.append(chunk_file)
                start_idx += chunk_size

                self.logger.debug(
                    f"Created chunk {chunk_idx}: {chunk_file} "
                    f"with {len(chunk_curves)} curves"
                )

            self.logger.info(
                f"Split {file_path} into {len(chunk_files)} chunk files"
            )
            return chunk_files

        except Exception as e:
            self.logger.error(f"Error splitting residue file {file_path}: {e}")
            return []

    def correlate_factor_to_sigma(
        self, factor: str, file_path: str
    ) -> Optional[str]:
        """
        Correlate a found factor to its sigma value from residue file.

        When a factor is found during Stage 2, this method identifies which
        curve (sigma value) discovered it by matching the factor to curves
        in the residue file.

        Args:
            factor: The factor that was found (as string)
            file_path: Path to the residue file

        Returns:
            Sigma value that found the factor, or None if correlation fails
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Residue file not found: {file_path}")
            return None

        try:
            # Convert factor to integer for comparison
            factor_int = int(factor)

            # Parse residue file to extract composite and sigma values
            composite = None
            sigma_values = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # Extract composite number
                    if not composite:
                        n_match = ECMPatterns.RESUME_N_PATTERN.search(line)
                        if n_match:
                            composite = n_match.group(1)

                    # Extract sigma values
                    sigma_match = ECMPatterns.RESUME_SIGMA_PATTERN.search(line)
                    if sigma_match:
                        sigma_values.append(sigma_match.group(1))

            if not composite or not sigma_values:
                self.logger.warning(
                    f"Could not extract composite or sigma values from {file_path}"
                )
                return None

            # Check if factor divides the composite
            composite_int = int(composite)
            if composite_int % factor_int != 0:
                self.logger.warning(
                    f"Factor {factor} does not divide composite {composite[:20]}..."
                )
                return None

            # For now, we can't deterministically identify which specific sigma
            # found the factor without re-running each curve. Return the first sigma
            # as a reasonable approximation for tracking purposes.
            if sigma_values:
                sigma = sigma_values[0]
                self.logger.info(
                    f"Correlated factor {factor} to sigma {sigma} "
                    f"(first curve in residue file)"
                )
                return sigma

            return None

        except ValueError as e:
            self.logger.error(f"Invalid numeric value: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Error correlating factor {factor} to sigma in {file_path}: {e}"
            )
            return None

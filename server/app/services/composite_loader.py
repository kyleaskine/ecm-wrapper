"""
Utility for loading composites from various sources.

Provides methods to parse and validate composite numbers from:
- Text files
- CSV content
- Lists of numbers
"""

import csv
import io
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from ..utils.number_utils import validate_integer

logger = logging.getLogger(__name__)


class CompositeLoader:
    """Utility for loading composites from various sources."""

    @staticmethod
    def from_text_file(file_path: Union[str, Path]) -> List[str]:
        """
        Load composites from a text file.

        Args:
            file_path: Path to text file with one number per line

        Returns:
            List of valid composite numbers
        """
        numbers = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue

                # Extract number from line (handle various formats)
                number = CompositeLoader._extract_number(line)
                if number and validate_integer(number):
                    numbers.append(number)
                else:
                    logger.warning(f"Invalid number at line {line_num}: {line}")

        return numbers

    @staticmethod
    def from_csv_content(
        csv_content: str,
        number_column: str = 'number',
        priority_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load composites from CSV content.

        Args:
            csv_content: CSV data as string
            number_column: Name of column containing numbers
            priority_column: Optional priority column name

        Returns:
            List of dictionaries with number and optional metadata
        """
        numbers = []
        reader = csv.DictReader(io.StringIO(csv_content))

        for row_num, row in enumerate(reader, 1):
            if number_column not in row:
                logger.error(f"Column '{number_column}' not found in CSV")
                break

            number = CompositeLoader._extract_number(row[number_column])
            if not number or not validate_integer(number):
                logger.warning(f"Invalid number at row {row_num}: {row[number_column]}")
                continue

            composite_data = {'number': number}

            # Add priority if specified
            if priority_column and priority_column in row:
                try:
                    composite_data['priority'] = int(row[priority_column])
                except ValueError:
                    composite_data['priority'] = 0

            # Handle specific SNFS and composite fields with proper type conversion
            if 'current_composite' in row and row['current_composite']:
                composite_data['current_composite'] = (
                    CompositeLoader._extract_number(row['current_composite'])
                )

            if 'has_snfs_form' in row and row['has_snfs_form']:
                value = (
                    row['has_snfs_form'].lower()
                    if isinstance(row['has_snfs_form'], str)
                    else str(row['has_snfs_form'])
                )
                composite_data['has_snfs_form'] = value in ('true', '1', 'yes', 't', 'y')

            if 'snfs_difficulty' in row and row['snfs_difficulty']:
                try:
                    composite_data['snfs_difficulty'] = int(row['snfs_difficulty'])
                except ValueError:
                    logger.warning(
                        "Invalid snfs_difficulty at row %s: %s",
                        row_num, row['snfs_difficulty']
                    )

            # Add any other metadata
            excluded_keys = [
                number_column, priority_column, 'current_composite',
                'has_snfs_form', 'snfs_difficulty'
            ]
            for key, value in row.items():
                if key not in excluded_keys:
                    composite_data[key] = value

            numbers.append(composite_data)

        return numbers

    @staticmethod
    def from_number_list(numbers: List[str]) -> List[str]:
        """
        Validate a list of number strings.

        Args:
            numbers: List of number strings

        Returns:
            List of valid numbers
        """
        valid_numbers = []
        for i, number in enumerate(numbers):
            cleaned = CompositeLoader._extract_number(number)
            if cleaned and validate_integer(cleaned):
                valid_numbers.append(cleaned)
            else:
                logger.warning(f"Invalid number at index {i}: {number}")

        return valid_numbers

    @staticmethod
    def _extract_number(text: str) -> Optional[str]:
        """
        Extract a number from text, handling various formats.

        Args:
            text: Text potentially containing a number

        Returns:
            Extracted number string, or None if no valid number found
        """
        if not text:
            return None

        # Remove whitespace
        text = text.strip()

        # Handle common prefixes/suffixes
        text = re.sub(r'^(composite|number|n)[:=\s]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(digits?|bits?)$', '', text, flags=re.IGNORECASE)

        # Extract just the digits
        match = re.search(r'\d+', text)
        if match:
            return match.group()

        return None

"""
Residue file manager service for decoupled two-stage ECM.

Handles:
- Parsing residue file metadata
- Storing/retrieving residue files
- Work assignment for stage 2
- Lifecycle management (claim, complete, expire)
"""

import logging
import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..models.residues import ECMResidue
from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..config import get_settings
from .t_level_calculator import TLevelCalculator

logger = logging.getLogger(__name__)


class ResidueManager:
    """Manages ECM residue files for decoupled two-stage processing."""

    # Patterns for parsing residue file metadata
    PARAM_PATTERN = re.compile(r'PARAM=(\d+)')
    B1_PATTERN = re.compile(r'B1=(\d+)')
    N_PATTERN = re.compile(r'N=(\d+)')
    SIGMA_PATTERN = re.compile(r'SIGMA=(\d+)')

    def __init__(self):
        """Initialize the residue manager."""
        settings = get_settings()
        self.storage_dir = Path(settings.residue_storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.t_level_calculator = TLevelCalculator()

    def parse_residue_file(self, file_content: bytes) -> Dict[str, Any]:
        """
        Parse metadata from a residue file.

        Args:
            file_content: Raw bytes of the residue file

        Returns:
            Dict with keys: composite, b1, parametrization, curve_count

        Raises:
            ValueError: If file format is invalid or missing required fields
        """
        try:
            content = file_content.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Residue file is not valid UTF-8: {e}")

        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("Residue file is empty")

        # Parse first line to get metadata
        first_line = lines[0]

        # Check if it's GPU format (single-line with METHOD=ECM; PARAM=...; etc)
        if 'METHOD=ECM' in first_line and 'SIGMA=' in first_line and ';' in first_line:
            # GPU format - each line is a complete curve
            param_match = self.PARAM_PATTERN.search(first_line)
            b1_match = self.B1_PATTERN.search(first_line)
            n_match = self.N_PATTERN.search(first_line)

            if not all([param_match, b1_match, n_match]):
                raise ValueError("GPU format residue file missing required fields (PARAM, B1, or N)")

            composite = n_match.group(1)
            b1 = int(b1_match.group(1))
            parametrization = int(param_match.group(1))

            # Count curves (each line with SIGMA= is a curve)
            curve_count = sum(1 for line in lines if self.SIGMA_PATTERN.search(line))

        else:
            # CPU format - multi-line with header blocks
            # Look for N=, B1=, PARAM= in the first few lines
            composite = None
            b1 = None
            parametrization = None

            for line in lines[:20]:  # Check first 20 lines for metadata
                if composite is None:
                    n_match = self.N_PATTERN.match(line)
                    if n_match:
                        composite = n_match.group(1)
                if b1 is None:
                    b1_match = self.B1_PATTERN.match(line)
                    if b1_match:
                        b1 = int(b1_match.group(1))
                if parametrization is None and 'PARAM=' in line:
                    param_match = self.PARAM_PATTERN.search(line)
                    if param_match:
                        parametrization = int(param_match.group(1))

            if not all([composite, b1, parametrization is not None]):
                raise ValueError("CPU format residue file missing required fields (N, B1, or PARAM)")

            # Count curves (SIGMA= lines indicate individual curves)
            curve_count = sum(1 for line in lines if line.startswith('SIGMA='))

        if curve_count == 0:
            raise ValueError("No curves found in residue file")

        return {
            'composite': composite,
            'b1': b1,
            'parametrization': parametrization,
            'curve_count': curve_count
        }

    def calculate_checksum(self, file_content: bytes) -> str:
        """Calculate SHA-256 checksum of file content."""
        return hashlib.sha256(file_content).hexdigest()

    def store_residue_file(
        self,
        db: Session,
        file_content: bytes,
        client_id: str,
        stage1_attempt_id: Optional[int] = None,
        expiry_days: int = 7
    ) -> ECMResidue:
        """
        Store a residue file and create database record.

        Args:
            db: Database session
            file_content: Raw residue file bytes
            client_id: ID of the uploading client
            stage1_attempt_id: Optional ID of the stage 1 attempt to link
            expiry_days: Days until residue expires

        Returns:
            ECMResidue database record

        Raises:
            ValueError: If file format is invalid or composite not found
        """
        # Parse metadata from file
        metadata = self.parse_residue_file(file_content)
        composite_number = metadata['composite']

        # Look up composite in database
        composite = db.query(Composite).filter(
            Composite.current_composite == composite_number
        ).first()

        if not composite:
            # Try matching by number field as fallback
            composite = db.query(Composite).filter(
                Composite.number == composite_number
            ).first()

        if not composite:
            raise ValueError(f"Composite {composite_number[:50]}... not found in database")

        # Generate unique filename
        import uuid
        file_uuid = str(uuid.uuid4())
        composite_dir = self.storage_dir / str(composite.id)
        composite_dir.mkdir(parents=True, exist_ok=True)
        file_path = composite_dir / f"{file_uuid}.txt"

        # Calculate checksum
        checksum = self.calculate_checksum(file_content)

        # Check for duplicate by checksum
        existing = db.query(ECMResidue).filter(
            ECMResidue.checksum == checksum
        ).first()
        if existing:
            raise ValueError(f"Duplicate residue file (checksum matches residue ID {existing.id})")

        # Write file to storage
        file_path.write_bytes(file_content)
        logger.info(f"Stored residue file: {file_path} ({len(file_content)} bytes)")

        # Create database record
        residue = ECMResidue(
            composite_id=composite.id,
            client_id=client_id,
            stage1_attempt_id=stage1_attempt_id,
            b1=metadata['b1'],
            parametrization=metadata['parametrization'],
            curve_count=metadata['curve_count'],
            storage_path=str(file_path),
            file_size_bytes=len(file_content),
            checksum=checksum,
            status='available',
            expires_at=datetime.utcnow() + timedelta(days=expiry_days)
        )

        db.add(residue)
        db.flush()  # Get the ID

        logger.info(
            f"Created residue record ID {residue.id}: "
            f"composite={composite.id}, B1={metadata['b1']}, "
            f"curves={metadata['curve_count']}, param={metadata['parametrization']}"
        )

        return residue

    def get_available_work(
        self,
        db: Session,
        client_id: str,
        min_digits: Optional[int] = None,
        max_digits: Optional[int] = None,
        min_priority: Optional[int] = None
    ) -> Optional[ECMResidue]:
        """
        Find an available residue for stage 2 processing.

        Args:
            db: Database session
            client_id: ID of requesting client
            min_digits: Minimum composite digit size
            max_digits: Maximum composite digit size
            min_priority: Minimum composite priority

        Returns:
            ECMResidue if found, None otherwise
        """
        query = db.query(ECMResidue).join(
            Composite, ECMResidue.composite_id == Composite.id
        ).filter(
            ECMResidue.status == 'available',
            ECMResidue.expires_at > datetime.utcnow()
        )

        # Apply filters
        if min_digits is not None:
            query = query.filter(Composite.digit_length >= min_digits)
        if max_digits is not None:
            query = query.filter(Composite.digit_length <= max_digits)
        if min_priority is not None:
            query = query.filter(Composite.priority >= min_priority)

        # Prioritize by composite priority (descending), then by creation time (oldest first)
        query = query.order_by(
            Composite.priority.desc(),
            ECMResidue.created_at.asc()
        )

        residue = query.first()
        if residue:
            logger.info(f"Found available residue ID {residue.id} for client {client_id}")
        else:
            logger.info(f"No available residues for client {client_id}")

        return residue

    def claim_residue(
        self,
        db: Session,
        residue_id: int,
        client_id: str,
        claim_timeout_hours: int = 24
    ) -> ECMResidue:
        """
        Claim a residue for stage 2 processing.

        Args:
            db: Database session
            residue_id: ID of residue to claim
            client_id: ID of claiming client
            claim_timeout_hours: Hours until claim expires

        Returns:
            Updated ECMResidue record

        Raises:
            ValueError: If residue not found or not available
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()

        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        if residue.status != 'available':
            raise ValueError(f"Residue {residue_id} is not available (status: {residue.status})")

        residue.status = 'claimed'
        residue.claimed_at = datetime.utcnow()
        residue.claimed_by = client_id
        # Update expiration to claim timeout
        residue.expires_at = datetime.utcnow() + timedelta(hours=claim_timeout_hours)

        logger.info(f"Residue {residue_id} claimed by {client_id}")
        return residue

    def release_claim(self, db: Session, residue_id: int, client_id: str) -> ECMResidue:
        """
        Release a claimed residue back to available pool.

        Args:
            db: Database session
            residue_id: ID of residue to release
            client_id: ID of client releasing (must match claimer)

        Returns:
            Updated ECMResidue record

        Raises:
            ValueError: If residue not found, not claimed, or wrong client
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()

        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        if residue.status != 'claimed':
            raise ValueError(f"Residue {residue_id} is not claimed (status: {residue.status})")

        if residue.claimed_by != client_id:
            raise ValueError(f"Residue {residue_id} is claimed by {residue.claimed_by}, not {client_id}")

        residue.status = 'available'
        residue.claimed_at = None
        residue.claimed_by = None
        # Reset expiration to default
        residue.expires_at = ECMResidue.default_expiry()

        logger.info(f"Residue {residue_id} released by {client_id}")
        return residue

    def complete_residue(
        self,
        db: Session,
        residue_id: int,
        stage2_attempt_id: int
    ) -> Tuple[ECMResidue, Optional[float]]:
        """
        Mark residue as completed after stage 2 finishes.

        This supersedes the stage 1 attempt and deletes the residue file.

        Args:
            db: Database session
            residue_id: ID of the completed residue
            stage2_attempt_id: ID of the stage 2 ECM attempt

        Returns:
            Tuple of (residue, new_t_level)
            - new_t_level: Updated t-level after supersession (if applicable)

        Raises:
            ValueError: If residue or attempt not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        # Get the stage 2 attempt
        stage2_attempt = db.query(ECMAttempt).filter(ECMAttempt.id == stage2_attempt_id).first()
        if not stage2_attempt:
            raise ValueError(f"Stage 2 attempt {stage2_attempt_id} not found")

        # Mark stage 1 attempt as superseded (if linked)
        if residue.stage1_attempt_id:
            stage1_attempt = db.query(ECMAttempt).filter(
                ECMAttempt.id == residue.stage1_attempt_id
            ).first()
            if stage1_attempt:
                stage1_attempt.superseded_by = stage2_attempt_id
                db.flush()  # Ensure supersession is visible to subsequent queries
                logger.info(
                    f"Marked stage 1 attempt {residue.stage1_attempt_id} as superseded by {stage2_attempt_id}"
                )

        # Update residue status
        residue.status = 'completed'
        residue.completed_at = datetime.utcnow()

        # Delete the residue file
        try:
            file_path = Path(residue.storage_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted residue file: {file_path}")
            else:
                logger.warning(f"Residue file not found for deletion: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting residue file {residue.storage_path}: {e}")

        # Recalculate t-level for composite (excluding superseded attempts)
        new_t_level = self._recalculate_composite_t_level(db, residue.composite_id)

        logger.info(
            f"Completed residue {residue_id}: stage2_attempt={stage2_attempt_id}, "
            f"new_t_level={new_t_level}"
        )

        return residue, new_t_level

    def _recalculate_composite_t_level(self, db: Session, composite_id: int) -> Optional[float]:
        """
        Recalculate t-level for a composite, excluding superseded attempts.

        Args:
            db: Database session
            composite_id: ID of composite to recalculate

        Returns:
            New t-level value, or None if calculation fails
        """
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            return None

        # Use the t-level calculator's recalculate method which already
        # excludes superseded attempts
        old_t_level = composite.current_t_level or 0.0
        new_t_level = self.t_level_calculator.recalculate_composite_t_level(db, composite)

        logger.info(
            f"Recalculated t-level for composite {composite_id}: "
            f"{old_t_level:.2f} -> {new_t_level:.2f}"
        )

        return new_t_level

    def get_residue_file_path(self, db: Session, residue_id: int) -> Optional[Path]:
        """
        Get the filesystem path for a residue file.

        Args:
            db: Database session
            residue_id: ID of residue

        Returns:
            Path to file, or None if not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            return None

        file_path = Path(residue.storage_path)
        if file_path.exists():
            return file_path

        logger.warning(f"Residue file not found: {file_path}")
        return None

    def suggest_b2_for_residue(self, db: Session, residue_id: int) -> Optional[int]:
        """
        Suggest an appropriate B2 value for stage 2 based on B1.

        Args:
            db: Database session
            residue_id: ID of residue

        Returns:
            Suggested B2 value, or None if residue not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            return None

        # Standard B2 = 100 * B1 is a common heuristic
        # For GPU work, even larger ratios can be beneficial
        suggested_b2 = residue.b1 * 100

        # Cap at a reasonable maximum (e.g., 10 trillion)
        max_b2 = 10_000_000_000_000
        suggested_b2 = min(suggested_b2, max_b2)

        return suggested_b2

    def cleanup_expired_residues(self, db: Session) -> int:
        """
        Clean up expired residues (delete files and mark as expired).

        Args:
            db: Database session

        Returns:
            Number of residues cleaned up
        """
        expired = db.query(ECMResidue).filter(
            ECMResidue.expires_at < datetime.utcnow(),
            ECMResidue.status.in_(['available', 'claimed'])
        ).all()

        count = 0
        for residue in expired:
            try:
                file_path = Path(residue.storage_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted expired residue file: {file_path}")

                residue.status = 'expired'
                count += 1
            except Exception as e:
                logger.error(f"Error cleaning up residue {residue.id}: {e}")

        if count > 0:
            logger.info(f"Cleaned up {count} expired residues")

        return count

    def get_stats(self, db: Session) -> Dict[str, int]:
        """
        Get statistics about residues in the system.

        Args:
            db: Database session

        Returns:
            Dict with counts by status and total pending curves
        """
        stats = {
            'total_available': 0,
            'total_claimed': 0,
            'total_completed': 0,
            'total_expired': 0,
            'total_curves_pending': 0
        }

        # Count by status
        status_counts = db.query(
            ECMResidue.status,
            func.count(ECMResidue.id)
        ).group_by(ECMResidue.status).all()

        for status, count in status_counts:
            if status == 'available':
                stats['total_available'] = count
            elif status == 'claimed':
                stats['total_claimed'] = count
            elif status == 'completed':
                stats['total_completed'] = count
            elif status == 'expired':
                stats['total_expired'] = count

        # Sum curves in available residues
        curves_sum = db.query(func.sum(ECMResidue.curve_count)).filter(
            ECMResidue.status == 'available'
        ).scalar()
        stats['total_curves_pending'] = curves_sum or 0

        return stats

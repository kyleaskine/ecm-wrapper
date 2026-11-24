from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session
from typing import NamedTuple, Optional, List, Tuple, Union
from ..models.factors import Factor
from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..utils.number_utils import validate_integer, verify_complete_factorization
from .group_order import GroupOrderCalculator
import logging

logger = logging.getLogger(__name__)


class GroupOrderJob(NamedTuple):
    """Inputs of one factor's group order calculation (see fill_group_orders)."""
    factor_id: int
    factor: str
    sigma: Optional[str]          # ECM: the curve's sigma; None for P-1/P+1
    parametrization: int          # ECM parametrization (0-3)
    method: Optional[str]         # 'pm1'/'pp1' select p-1/p+1; else ECM


class FactorService:
    @staticmethod
    def add_factor(db: Session, composite_id: int, factor: str, attempt_id: Optional[int] = None,
                   sigma: Optional[str] = None) -> Tuple[Factor, bool]:
        """
        Add a factor to a composite.
        Returns (factor, created) where created is True if new factor was added.

        The group order is not calculated here: PARI/GP takes 40-330 ms and
        this runs inside the submit transaction, under the composite's row
        lock. Callers pass group_order_job() of a new factor to
        fill_group_orders() once the transaction has committed.

        Args:
            sigma: The sigma value that found this factor (ECM only) - stored as string for large param 0 values
        """
        # Validate factor
        if not validate_integer(factor):
            raise ValueError(f"Invalid factor format: {factor}")

        # Check if factor already exists for this composite
        existing = db.query(Factor).filter(
            Factor.composite_id == composite_id,
            Factor.factor == factor
        ).first()

        if existing:
            return existing, False

        # Add new factor
        new_factor = Factor(
            composite_id=composite_id,
            factor=factor,
            found_by_attempt_id=attempt_id,
            sigma=sigma,
        )

        db.add(new_factor)
        db.flush()  # Make visible within transaction
        db.refresh(new_factor)

        return new_factor, True

    @staticmethod
    def group_order_job(factor: Factor, parametrization: Optional[int],
                        method: Optional[str]) -> Optional[GroupOrderJob]:
        """The group order calculation a factor needs, or None.

        ECM factors (with a sigma) get the elliptic curve's group order, with
        parametrization 3 when none is known; P-1 and P+1 factors get p-1 or
        p+1 factored. parametrization and method describe the run that found
        the factor: for a factor just added, the submission's own values.
        """
        if factor.sigma is not None:
            return GroupOrderJob(factor.id, factor.factor, factor.sigma,
                                 3 if parametrization is None else parametrization, method)
        if method in ("pm1", "pp1"):
            return GroupOrderJob(factor.id, factor.factor, None, 3, method)
        return None

    @staticmethod
    def missing_group_order_job(db: Session, factor: Factor) -> Optional[GroupOrderJob]:
        """The calculation for a stored factor that has no group order, or None.

        Covers a background calculation that failed or was lost (the process
        was killed before it ran). Uses the parametrization and method of the
        attempt that found the factor.
        """
        if factor.group_order is not None:
            return None
        parametrization: Optional[int] = None
        method: Optional[str] = None
        if factor.found_by_attempt_id is not None:
            attempt = db.query(ECMAttempt.parametrization, ECMAttempt.method).filter(
                ECMAttempt.id == factor.found_by_attempt_id
            ).first()
            if attempt is not None:
                parametrization, method = attempt.parametrization, attempt.method
        return FactorService.group_order_job(factor, parametrization, method)

    @staticmethod
    def compute_group_order(job: GroupOrderJob) -> Optional[Tuple[str, Optional[str]]]:
        """Run PARI/GP for one job. Returns (order, factorization) or None."""
        try:
            calculator = GroupOrderCalculator()
            if job.sigma is not None:
                return calculator.calculate_group_order(job.factor, job.sigma, job.parametrization)
            if job.method in ("pm1", "pp1"):
                return calculator.calculate_p1_order(job.factor, job.method)
        except Exception as e:
            logger.warning(f"Failed to calculate group order for factor {job.factor[:20]}...: {e}")
        return None

    @staticmethod
    def get_factors_for_composite(db: Session, composite_id: int) -> List[Factor]:
        """Get all factors for a composite."""
        return db.query(Factor).filter(Factor.composite_id == composite_id).all()

    @staticmethod
    def verify_factorization(db: Session, composite_id: int) -> bool:
        """
        Check if we have a complete factorization of the composite.
        Returns True if the product of all factors equals the original composite.
        """
        try:
            composite = db.query(Composite).filter(Composite.id == composite_id).first()
            if not composite:
                return False

            factors = FactorService.get_factors_for_composite(db, composite_id)
            if not factors:
                return False

            factor_strings = [factor.factor for factor in factors]
            return verify_complete_factorization(composite.number, factor_strings)

        except Exception:
            # Log but don't raise - verification failure should be handled gracefully
            logger.exception(f"Factorization check failed for composite {composite_id}")
            return False


def fill_group_orders(bind: Union[Engine, Connection], jobs: List[GroupOrderJob]) -> None:
    """Calculate and store group orders after the submission has committed.

    Runs as a background task of /submit_result, after the response is sent,
    so PARI/GP no longer runs under the composite's row lock. It reads only
    the job's inputs and writes only the two group order columns of a factor
    row that already exists, each in its own short transaction. A failed
    calculation leaves them NULL, as before.

    Args:
        bind: The request session's engine (tests bind a different one)
        jobs: From FactorService.group_order_job()
    """
    for job in jobs:
        result = FactorService.compute_group_order(job)
        if not result:
            continue
        group_order, factorization = result
        with Session(bind=bind) as db:
            try:
                updated = db.query(Factor).filter(
                    Factor.id == job.factor_id,
                    Factor.factor == job.factor,  # the row this job was made for
                    Factor.group_order.is_(None),
                ).update({
                    Factor.group_order: group_order,
                    Factor.group_order_factorization: factorization,
                }, synchronize_session=False)
                db.commit()
                if updated:
                    logger.info(f"Stored group order for factor {job.factor_id}: {group_order}")
            except Exception:
                db.rollback()
                logger.exception(f"Failed to store group order for factor {job.factor_id}")


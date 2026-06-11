"""
Group Order Calculator Service

Calculates elliptic curve group orders for ECM factors using PARI/GP.
Based on the FindGroupOrder function for different ECM parametrizations.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from ..utils.process_executor import ExternalProgramExecutor

logger = logging.getLogger(__name__)


class GroupOrderCalculator:
    """Calculate elliptic curve group orders for ECM factors."""

    gp_binary: str
    script_path: Optional[str]
    executor: ExternalProgramExecutor

    def __init__(self, gp_binary: str = "gp", script_path: Optional[str] = None):
        """
        Initialize group order calculator.

        Args:
            gp_binary: Path to PARI/GP binary (default: "gp" in PATH)
            script_path: Path to group.gp script (default: /app/bin/group.gp or ./bin/group.gp)
        """
        self.gp_binary = gp_binary

        # Initialize executor for PARI/GP
        self.executor = ExternalProgramExecutor(
            gp_binary,
            binary_name="PARI/GP"
        )

        # Find the group.gp script
        if script_path:
            self.script_path = script_path
        else:
            # Try Docker location first, then local dev location
            possible_paths = [
                "/app/bin/group.gp",
                Path(__file__).parent.parent.parent / "bin" / "group.gp",
            ]
            self.script_path = None
            for path in possible_paths:
                if os.path.exists(str(path)):
                    self.script_path = str(path)
                    break

            if not self.script_path:
                logger.warning("group.gp script not found, will use inline script")
                self.script_path = None

    def _gp_args(self, factor: str) -> list:
        """Return PARI/GP command-line args, increasing stack size for large factors."""
        args = ["-q", "-f"]
        if len(factor) > 55:
            args.extend(["-s", "64M"])
        return args

    def calculate_group_order(
        self, factor: str, sigma: str, parametrization: int = 3
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        Calculate elliptic curve group order for a factor found by ECM.

        Args:
            factor: Prime factor (p)
            sigma: Sigma value used to find the factor
            parametrization: ECM parametrization (0, 1, 2, or 3)

        Returns:
            Tuple of (group_order, factorization) or None if calculation fails.
            Factorization may be None if factorization step fails.
        """
        # Parse sigma if it includes parametrization prefix (e.g., "3:12345")
        sigma_value = sigma
        if isinstance(sigma, str) and ':' in sigma:
            parts = sigma.split(':', 1)
            # Use parametrization from sigma if not explicitly provided
            if parametrization == 3:  # Default value
                try:
                    parametrization = int(parts[0])
                except ValueError:
                    pass
            sigma_value = parts[1]

        # Validate parametrization
        if parametrization not in [0, 1, 2, 3]:
            logger.warning(
                f"Invalid parametrization {parametrization}, defaulting to 3"
            )
            parametrization = 3

        # SECURITY: factor and sigma are interpolated into a GP script (which has
        # system()), so they must be plain integers. Callers validate upstream,
        # but enforce it here too — this is the injection boundary.
        if not factor.isdigit() or not str(sigma_value).isdigit():
            logger.error(
                f"Refusing group order calculation with non-numeric input: "
                f"factor={factor[:20]!r}... sigma={str(sigma_value)[:20]!r}"
            )
            return None

        # Build PARI/GP script to load function and call it
        if self.script_path:
            # Use external script file
            script = f'read("{self.script_path}");FindGroupOrder({factor},{sigma_value},{parametrization})\nquit\n'
        else:
            # Fallback to inline condensed version
            inline_script = 'FindGroupOrder(p,s,param=0)={{A=0;b=0;if(param==0,v=Mod(4*s,p);u=Mod(s^2-5,p);x=u^3;A=(3*u+v)*(v-u)^3/(4*x*v)-2;x=x/v^3;b=x*(x*(x+A)+1),param==1,A=Mod(4*s^2,p)/2^64-2;b=4*A+10,param==2,E=ellinit([0,Mod(36,p)]);[x,y]=ellmul(E,[-3,3],s);x3=(3*x+y+6)/(2*(y-3));A=-(3*x3^4+6*x3^2-1)/(4*x3^3);b=1/(4*A+10),param==3,A=Mod(4*s,p)/2^32-2;b=4*A+10);if(param>=0&&param<=3,E=ellinit([0,b*A,0,b^2,0]);ellcard(E),0)}};'
            script = f'{inline_script}FindGroupOrder({factor},{sigma_value},{parametrization})\nquit\n'

        try:
            # Execute PARI/GP using executor
            success, group_order = self.executor.execute_and_get_last_line(
                args=self._gp_args(factor),
                input_data=script,
                timeout=30
            )

            if not success or not group_order:
                return None

            # Try to factor the group order for interesting structure
            factorization = self._factor_group_order(group_order)

            logger.info(
                f"Calculated group order for factor {factor[:20]}... "
                f"with sigma {sigma_value}: {group_order}"
            )

            return (group_order, factorization)

        except Exception as e:
            logger.error(
                f"Error calculating group order for factor {factor[:20]}...: {e}"
            )
            return None

    def calculate_p1_order(
        self, factor: str, method: str
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        Calculate p-1 or p+1 for a factor found by P-1 or P+1 method.

        For PM1, the group is the multiplicative group mod p, with order p-1.
        For PP1, the relevant group has order p+1.

        Args:
            factor: Prime factor (p)
            method: "pm1" or "pp1"

        Returns:
            Tuple of (order, factorization) or None if calculation fails.
        """
        if method == "pm1":
            order_expr = f"{factor} - 1"
            label = "p-1"
        elif method == "pp1":
            order_expr = f"{factor} + 1"
            label = "p+1"
        else:
            return None

        # SECURITY: factor is interpolated into a GP script — must be a plain integer
        if not factor.isdigit():
            logger.error(f"Refusing {label} calculation with non-numeric factor: {factor[:20]!r}...")
            return None

        # Compute the value and factor it in one PARI/GP call
        script = f"n = {order_expr}; print(n); factor(n)\nquit\n"

        try:
            success, lines = self.executor.execute_and_parse_lines(
                args=self._gp_args(factor),
                input_data=script,
                timeout=30,
                filter_empty=True
            )

            if not success or not lines:
                return None

            # First line is the order value
            order_value = lines[0].strip()

            # Remaining lines are the factorization matrix
            factors = []
            for line in lines[1:]:
                if line.startswith('[') and line.endswith(']'):
                    parts = line[1:-1].split()
                    if len(parts) == 2:
                        base = parts[0]
                        exp = parts[1]
                        if exp == '1':
                            factors.append(base)
                        else:
                            factors.append(f"{base}^{exp}")

            factorization = " * ".join(factors) if factors else None

            logger.info(
                f"Calculated {label} for factor {factor[:20]}...: "
                f"{order_value} = {factorization}"
            )

            return (order_value, factorization)

        except Exception as e:
            logger.error(
                f"Error calculating {label} for factor {factor[:20]}...: {e}"
            )
            return None

    def _factor_group_order(self, group_order: str) -> Optional[str]:
        """
        Factor the group order using PARI/GP.

        Args:
            group_order: The group order to factor

        Returns:
            Factorization string in format "2^5 * 3^2 * 5^2 * ..." or None if factorization fails
        """
        script = f"factor({group_order})\nquit\n"

        try:
            # Execute PARI/GP factorization
            success, lines = self.executor.execute_and_parse_lines(
                args=self._gp_args(group_order),
                input_data=script,
                timeout=10,
                filter_empty=True
            )

            if not success or not lines:
                return None

            # Parse PARI/GP matrix format: [2 5]\n[3 2]\n[5 2] means 2^5 * 3^2 * 5^2
            factors = []
            for line in lines:
                if line.startswith('[') and line.endswith(']'):
                    # Remove brackets and split
                    parts = line[1:-1].split()
                    if len(parts) == 2:
                        base = parts[0]
                        exp = parts[1]
                        if exp == '1':
                            factors.append(base)
                        else:
                            factors.append(f"{base}^{exp}")

            if factors:
                return " * ".join(factors)

            return None

        except Exception:
            return None

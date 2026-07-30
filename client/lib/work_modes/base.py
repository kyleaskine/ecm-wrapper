#!/usr/bin/env python3
"""
Base class for the WorkMode strategy pattern.

`WorkMode` provides the work-loop template (request -> execute -> submit ->
complete -> cleanup) and the 3-level Ctrl+C handling. Subclasses live in
sibling modules: stage1_producer, stage2_consumer, p1_sweep, standard,
composite_target, adaptive.

`WorkLoopContext` bundles the wrapper, client_id, and parsed args so mode
constructors don't need long parameter lists.

Public surface is re-exported from `lib.work_modes` for backward compat.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, TYPE_CHECKING
import signal

from ..ecm_config import FactorResult
from ..work_args import WorkArgs
from ..work_helpers import print_work_status
from ..error_helpers import check_work_limit_reached
from ..cleanup_helpers import handle_shutdown
from ..api_client import ResourceNotFoundError

if TYPE_CHECKING:
    from ..ecm_executor import ECMWrapper


# Circuit breaker threshold
MAX_CONSECUTIVE_FAILURES = 3


class SubmissionFailedError(RuntimeError):
    """
    The work executed, but reporting its result to the server failed.

    Distinct from an execution error: the curves are done and the result is in
    the submission queue, so cleanup can hold the assignment instead of handing
    the composite to another client. Only the work loop's submit step raises
    this - an ECM crash or a Ctrl+C must not be mistaken for it.
    """


@dataclass
class WorkLoopContext:
    """
    Shared context for all work loop modes.

    Contains all the state and configuration needed to execute work loops.
    Passed to WorkMode constructors to avoid long argument lists.
    """
    wrapper: 'ECMWrapper'
    client_id: str
    args: WorkArgs
    work_count_limit: Optional[int] = None
    finish_after_current: bool = field(default=False, init=False)

    def __post_init__(self):
        """Ensure API clients are initialized."""
        self.wrapper._ensure_api_clients()


class WorkMode(ABC):
    """
    Abstract base class for auto-work execution modes.

    Implements the Template Method pattern: the run() method provides
    the work loop structure, while subclasses implement the specific
    behavior for each step.

    Subclasses must implement:
    - mode_name: Human-readable name for logging
    - request_work(): Get work from server
    - execute_work(): Run factorization
    - submit_results(): Submit to API
    - complete_work(): Finalize assignment

    Optional overrides:
    - cleanup_on_failure(): Mode-specific cleanup
    - cleanup_on_shutdown(): Cleanup for graceful shutdown
    """

    # Subclasses should set this
    mode_name: str = "Unknown Mode"

    def __init__(self, ctx: WorkLoopContext):
        self.ctx = ctx
        self.wrapper = ctx.wrapper
        self.api_client = ctx.wrapper._get_api_client()
        self.logger = ctx.wrapper.logger
        self.args = ctx.args

        # Work tracking state
        self.current_work_id: Optional[str] = None
        self.current_residue_id: Optional[int] = None
        self.completed_count: int = 0
        self.consecutive_failures: int = 0

        # Set by _submit_stage2_results when the server bundled residue
        # completion into the submission (residue_completed in response)
        self._residue_completed_in_submit: bool = False
        self._submit_new_t_level: Optional[float] = None

        # Graceful shutdown state (3-level)
        self._first_interrupt_received: bool = False
        self._second_interrupt_received: bool = False
        self._original_sigint_handler: Any = None

    @abstractmethod
    def request_work(self) -> Optional[Dict[str, Any]]:
        """
        Request work assignment from server.

        Returns:
            Work assignment dictionary, or None if no work available.
            Should handle retry/wait logic internally.
        """
        pass

    @abstractmethod
    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        """
        Execute factorization on the work assignment.

        Args:
            work: Work assignment from request_work()

        Returns:
            FactorResult with execution results
        """
        pass

    @abstractmethod
    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        """
        Submit results to API server.

        Args:
            work: Original work assignment
            result: Execution result from execute_work()

        Returns:
            True if submission succeeded, False otherwise
        """
        pass

    def complete_work(self, work: Dict[str, Any]) -> None:
        """
        Mark work assignment as complete on server.

        Default implementation completes via work_id. Override in subclasses
        that use different identifiers (e.g., Stage2ConsumerMode uses residue_id).

        Args:
            work: Work assignment to complete
        """
        if not self.current_work_id:
            return
        try:
            if not self.api_client.complete_work(self.current_work_id, self.ctx.client_id):
                self.wrapper.submission_queue.enqueue_work_completion(
                    self.current_work_id, self.ctx.client_id
                )
        except ResourceNotFoundError:
            self.logger.warning(f"Work {self.current_work_id} already expired/completed on server, skipping")

    def cleanup_on_failure(self, work: Optional[Dict[str, Any]], error: BaseException) -> None:
        """
        Mode-specific cleanup after a failure.

        Override in subclasses for custom cleanup behavior.
        Default implementation abandons work if we have a work_id.
        If the abandon call fails (e.g. network down), queues an abandonment
        for later retry so the assignment doesn't stay active.

        Exception: on SubmissionFailedError the curves ran and only the report
        failed, so the result is already in the queue. Abandoning then would
        release the composite for another client to re-run work we have
        finished, so the assignment is held instead and a work_complete is
        chained onto the queued result. The server's assignment expiry (1 day)
        is the backstop if this client never returns. Every other failure -
        an execution error, a Ctrl+C - abandons as before, even if an unrelated
        result for this assignment happens to be sitting in the queue (t-level
        mode submits per B1 batch, so that is a real possibility).

        Args:
            work: Work assignment that failed (may be None)
            error: The exception that occurred (can be Exception or KeyboardInterrupt)
        """
        if self.current_work_id:
            queue = self.wrapper.submission_queue
            if isinstance(error, SubmissionFailedError) and queue.attach_work_completion(
                    self.current_work_id, self.ctx.client_id):
                self.logger.info(
                    f"Holding work {self.current_work_id} assignment - "
                    "completed result is queued for retry"
                )
            elif not self.wrapper.abandon_work(self.current_work_id, reason="execution_error"):
                # Network likely down - queue abandonment so assignment gets released on reconnect
                queue.enqueue_work_abandonment(
                    self.current_work_id, self.ctx.client_id
                )
            self.current_work_id = None

    def cleanup_on_shutdown(self) -> None:
        """
        Cleanup for graceful shutdown (Ctrl+C).

        Override in subclasses for mode-specific shutdown behavior.
        """
        pass

    def on_work_started(self, work: Dict[str, Any]) -> None:
        """
        Called when work is received, before execution.

        Override to store work-specific state or print headers.
        Default implementation stores work_id.
        """
        self.current_work_id = work.get('work_id')

    def on_work_completed(self, work: Dict[str, Any], result: FactorResult) -> None:
        """
        Called after successful completion.

        Override for mode-specific completion handling.
        """
        self.current_work_id = None
        self.completed_count += 1
        self.consecutive_failures = 0
        # Reset graceful shutdown flags after each work unit completes
        # (in case finish_after_current is cleared and loop continues)
        self.wrapper.graceful_shutdown_requested = False
        self.wrapper.shutdown_level = 0
        self.wrapper.stop_event.clear()

    # --- Shared submission helpers ---

    def _submit_ecm_results(
        self,
        results_dict: Dict[str, Any],
        program_name: str = 'gmp-ecm-ecm',
        error_label: str = "ECM"
    ) -> bool:
        """
        Submit standard ECM results to server.

        Common pattern for B1/B2-mode ECM and multiprocess submissions.
        Validates curves > 0, submits via wrapper, handles failure.

        Args:
            results_dict: Pre-built results dict with composite, b1, b2, etc.
            program_name: Program identifier for API (e.g., 'gmp-ecm-ecm', 'gmp-ecm-pm1')
            error_label: Label for error messages

        Returns:
            True if submission succeeded
        """
        if results_dict.get('curves_completed', 0) == 0:
            self.logger.error(f"Zero curves completed for {error_label}, execution may have failed (check ECM binary path)")
            return False

        submit_response = self.wrapper.submit_result(
            results_dict, self.args.project, program_name
        )

        if not submit_response:
            self.logger.error(f"Failed to submit {error_label} results")
            return False

        return True

    def _submit_stage2_results(
        self,
        work: Dict[str, Any],
        result: FactorResult,
        b2: Optional[int],
        factor: Optional[str],
        sigma: Optional[str],
        raw_output: str,
        residue_id: Optional[int],
        residue_checksum: Optional[str],
    ) -> tuple[bool, Optional[int], bool]:
        """
        Submit stage 2 results and extract attempt_id from primary endpoint.

        Common pattern for Stage2ConsumerMode and AdaptiveCPUMode stage 2 paths.

        Args:
            work: Work assignment dict (needs 'composite', 'b1', 'curve_count', 'parametrization')
            result: FactorResult from execution
            b2: B2 value used (-1 means GMP-ECM default, stored as None)
            factor: Primary factor found (or None)
            sigma: Sigma value that found the factor
            raw_output: Aggregated raw output from workers
            residue_id: Server residue ID (for fallback message)
            residue_checksum: Residue file checksum (for orphan detection)

        Returns:
            Tuple of (success, stage2_attempt_id, primary_submission_failed)
        """
        if not result.success:
            self.logger.error(result.error_message or "Stage 2 execution failed")
            return False, None, False

        # Reset bundled-completion state for this work item
        self._residue_completed_in_submit = False
        self._submit_new_t_level = None

        results = {
            'composite': work['composite'],
            'b1': work['b1'],
            'b2': None if b2 == -1 else b2,
            'curves_requested': work['curve_count'],
            'curves_completed': result.curves_run,
            'factors_found': result.factors,
            'factor_found': factor,
            'sigma': sigma,
            'raw_output': raw_output or f"Stage 2 from residue {residue_id}",
            'method': 'ecm',
            'parametrization': work.get('parametrization', 3),
            'execution_time': result.execution_time,
            'residue_checksum': residue_checksum,
        }

        print("Submitting stage 2 results...")
        # If every endpoint fails the queue records this chain so a later drain
        # can call complete_residue with the attempt_id returned on retry,
        # finalizing the work without the client re-executing stage 2.
        completion_chain = None
        if residue_id is not None:
            completion_chain = {
                "residue_id": residue_id,
                "client_id": self.ctx.client_id,
            }
        submit_response = self.wrapper.submit_result(
            results, self.args.project, 'gmp-ecm-ecm',
            completion_chain=completion_chain,
        )

        if not submit_response:
            self.logger.error("Failed to submit stage 2 results")
            return False, None, False

        primary = submit_response.primary_response
        if primary:
            stage2_attempt_id = primary.get('attempt_id')
            if not stage2_attempt_id:
                self.logger.error("No attempt_id returned from primary endpoint")
                return False, None, False
            print(f"Stage 2 attempt ID: {stage2_attempt_id}")
            # Newer servers complete the residue inside /submit_result;
            # complete_work uses this to skip the redundant completion call
            self._residue_completed_in_submit = bool(primary.get('residue_completed'))
            self._submit_new_t_level = primary.get('new_t_level')
            return True, stage2_attempt_id, False
        else:
            self.logger.warning("Primary endpoint submission failed (other endpoints may have succeeded)")
            self.logger.warning("Skipping residue completion - failed submission saved for retry via resend_failed.py")
            return True, None, True

    def _setup_signal_handler(self) -> None:
        """
        Install signal handler for graceful shutdown (3 levels).

        1st Ctrl+C: Set finish_after_current flag - let the entire current
                    assignment finish (all remaining curves), submit results, then exit.
                    Does NOT interrupt execution in progress.
        2nd Ctrl+C: Signal workers to finish current curve then stop.
                    Sets graceful_shutdown_requested + stop_event.
        3rd Ctrl+C: Raise KeyboardInterrupt for immediate abort.
        """
        def handler(signum, frame):
            if not self._first_interrupt_received:
                # First interrupt: finish entire current assignment, then exit
                self._first_interrupt_received = True
                self.ctx.finish_after_current = True
                # Do NOT set graceful_shutdown_requested - let the full assignment complete
                print("\n")
                print("=" * 60)
                print("Will complete current assignment, then exit.")
                print("Press Ctrl+C again to stop after current curve.")
                print("=" * 60)
            elif not self._second_interrupt_received:
                # Second interrupt: stop after current curve
                self._second_interrupt_received = True
                self.wrapper.graceful_shutdown_requested = True
                self.wrapper.stop_event.set()
                self.wrapper._signal_subprocesses_interrupt()
                print("\n")
                print("=" * 60)
                print("Stopping after current curve...")
                print("Press Ctrl+C again to abort immediately.")
                print("=" * 60)
            else:
                # Third interrupt: immediate abort
                raise KeyboardInterrupt()

        self._original_sigint_handler = signal.signal(signal.SIGINT, handler)

    def _restore_signal_handler(self) -> None:
        """Restore original signal handler and reset graceful shutdown flags."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None
        # Reset graceful shutdown flags for wrapper
        self.wrapper.graceful_shutdown_requested = False
        self.wrapper.shutdown_level = 0
        self.wrapper.stop_event.clear()

    def _drain_queue(self) -> None:
        """Drain the submission queue, retrying any pending items."""
        queue = self.wrapper.submission_queue
        if queue.count() > 0:
            queue.drain(self.api_client)

    def should_continue(self) -> bool:
        """
        Check if work loop should continue.

        Returns:
            True to continue, False to exit loop
        """
        # Check for graceful shutdown request (first Ctrl+C)
        if self.ctx.finish_after_current:
            return False

        # Check for hard interruption
        if self.wrapper.interrupted:
            return False

        # Check work count limit
        if self.ctx.work_count_limit and self.completed_count >= self.ctx.work_count_limit:
            return False

        return True

    def run(self) -> int:
        """
        Main work loop - Template Method pattern.

        This method provides the skeleton algorithm that all modes share.
        Subclasses customize behavior by overriding the abstract methods.

        Returns:
            Number of work assignments completed
        """
        self._print_startup_banner()
        self._setup_signal_handler()

        # Drain submission queue on startup (retry any pending items from previous runs)
        self._drain_queue()

        try:
            b2_dict: Dict[int, int] = dict()
            k_dict: Dict[int, int] = dict()
            self._b2_dictionary: Optional[Dict[int, int]] = None
            if self.args.b2_dictionary is not None:
                from lib.arg_parser import load_b2_dictionary
                b2_dict, k_dict = load_b2_dictionary(self.args.b2_dictionary)
                if b2_dict:
                    self._b2_dictionary = b2_dict

            while self.should_continue():
                # Drain submission queue before each work request
                self._drain_queue()

                # Request work from server
                work = self.request_work()
                if not work:
                    continue

                if 'b1' in work and work['b1'] in b2_dict:
                    b1 = work['b1']
                    work['b2_from_dict'] = b2_dict[b1]
                    print(f"Using B2 = {b2_dict[b1]} from dictionary.")
                    if b1 in k_dict:
                        work['k_from_dict'] = k_dict[b1]

                # Track work assignment
                self.on_work_started(work)

                try:
                    # Execute factorization
                    result = self.execute_work(work)

                    # After execution returns, check if Ctrl+C was pressed during execution.
                    # The executor's signal handler sets shutdown_level but does NOT set
                    # finish_after_current (which is a work-loop concern). Sync the state here.
                    if self.wrapper.shutdown_level >= 1 and not self.ctx.finish_after_current:
                        self.ctx.finish_after_current = True

                    # Check for hard interruption during execution
                    if self.wrapper.interrupted:
                        self.logger.info(f"{self.mode_name} interrupted by user, cleaning up...")
                        self.cleanup_on_failure(work, KeyboardInterrupt())
                        break

                    # Submit results
                    if not self.submit_results(work, result):
                        self.consecutive_failures += 1
                        self.cleanup_on_failure(work, SubmissionFailedError("Submission failed"))

                        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            self.logger.error(
                                f"Too many consecutive failures ({self.consecutive_failures}), exiting..."
                            )
                            break
                        continue

                    # Mark complete
                    self.complete_work(work)
                    self.on_work_completed(work, result)

                    # Print status and check limit
                    if print_work_status(self.mode_name, self.completed_count, self.ctx.work_count_limit):
                        break

                except Exception as e:
                    self.consecutive_failures += 1
                    self.logger.exception(f"Error in {self.mode_name}: {e}")
                    self.cleanup_on_failure(work, e)

                    if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        self.logger.error(
                            f"Too many consecutive failures ({self.consecutive_failures}), exiting..."
                        )
                        break

                    if check_work_limit_reached(self.completed_count, self.ctx.work_count_limit):
                        break

            # Check if we exited due to graceful shutdown
            if self.ctx.finish_after_current:
                self._handle_graceful_exit()

        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()

        finally:
            self._restore_signal_handler()

        return self.completed_count

    def _print_startup_banner(self) -> None:
        """Print mode startup banner."""
        print("=" * 60)
        if self.ctx.work_count_limit:
            print(f"{self.mode_name} - will process {self.ctx.work_count_limit} assignment(s)")
        else:
            print(f"{self.mode_name} - requesting work from server")
        print("Ctrl+C once: finish current assignment, then exit")
        print("Ctrl+C twice: stop after current curve")
        print("Ctrl+C three times: abort immediately")
        print("=" * 60)
        print()

    def _handle_graceful_exit(self) -> None:
        """Handle graceful exit after completing current work."""
        print()
        print("=" * 60)
        print(f"{self.mode_name} - graceful shutdown complete")
        print(f"Completed {self.completed_count} assignment(s)")
        print("=" * 60)

    def _handle_keyboard_interrupt(self) -> None:
        """Handle immediate abort (second Ctrl+C)."""
        self.cleanup_on_shutdown()
        handle_shutdown(
            wrapper=self.wrapper,
            current_work_id=self.current_work_id,
            current_residue_id=self.current_residue_id,
            mode_name=self.mode_name,
            completed_count=self.completed_count
        )

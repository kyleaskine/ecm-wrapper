#!/usr/bin/env python3
"""
Unit tests for work_modes module.

Tests:
- WorkLoopContext initialization and state
- WorkMode.should_continue() logic (work count limit, graceful shutdown, interrupts)
- Signal handler behavior (first/second Ctrl+C)
- Circuit breaker (MAX_CONSECUTIVE_FAILURES)
- Work lifecycle callbacks (on_work_started, on_work_completed)
- get_work_mode factory function
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.work_modes import (
    WorkLoopContext, WorkMode, MAX_CONSECUTIVE_FAILURES, SubmissionFailedError,
    StandardAutoWorkMode, Stage1ProducerMode, Stage2ConsumerMode,
    CompositeTargetMode, get_work_mode
)
from lib.work_args import WorkArgs
from lib.ecm_config import FactorResult
from lib.submission_queue import SubmissionQueue


class MockWrapper:
    """Mock wrapper for testing WorkLoopContext."""

    def __init__(self):
        import threading
        self.logger = Mock()
        self.interrupted = False
        self.graceful_shutdown_requested = False
        self.shutdown_level = 0
        self.stop_event = threading.Event()
        self._api_clients_initialized = False
        self.api_client = Mock()
        self.typed_config = None  # resolve_worker_count falls back to CPU count
        # spec'd: cleanup_on_failure is a cross-module contract with
        # SubmissionQueue, and an unspecced Mock would keep passing after a
        # method there was renamed while the real client raised AttributeError.
        self.submission_queue = Mock(spec=SubmissionQueue)

    def _ensure_api_clients(self):
        self._api_clients_initialized = True

    def _get_api_client(self):
        return self.api_client

    def abandon_work(self, work_id, reason=None):
        pass


def create_mock_args(**kwargs):
    """Create a WorkArgs instance with overrides applied."""
    return WorkArgs(**kwargs)


class TestWorkLoopContext:
    """Tests for WorkLoopContext dataclass."""

    def test_initialization(self):
        """Test basic context initialization."""
        wrapper = MockWrapper()
        args = create_mock_args()

        ctx = WorkLoopContext(
            wrapper=wrapper,
            client_id='test-client',
            args=args,
            work_count_limit=10
        )

        assert ctx.wrapper is wrapper
        assert ctx.client_id == 'test-client'
        assert ctx.work_count_limit == 10
        assert ctx.finish_after_current is False

    def test_ensures_api_clients_on_init(self):
        """Test that API clients are initialized on context creation."""
        wrapper = MockWrapper()
        args = create_mock_args()

        assert wrapper._api_clients_initialized is False

        ctx = WorkLoopContext(wrapper=wrapper, client_id='test', args=args)

        assert wrapper._api_clients_initialized is True

    def test_finish_after_current_default(self):
        """Test finish_after_current defaults to False."""
        wrapper = MockWrapper()
        ctx = WorkLoopContext(
            wrapper=wrapper,
            client_id='test',
            args=create_mock_args()
        )

        assert ctx.finish_after_current is False


class TestWorkModeShouldContinue:
    """Tests for WorkMode.should_continue() logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.wrapper = MockWrapper()
        self.args = create_mock_args()

    def test_should_continue_true_by_default(self):
        """Test should_continue returns True with no limits set."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args,
            work_count_limit=None
        )

        mode = StandardAutoWorkMode(ctx)

        assert mode.should_continue() is True

    def test_should_continue_false_when_interrupted(self):
        """Test should_continue returns False when wrapper is interrupted."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args
        )
        self.wrapper.interrupted = True

        mode = StandardAutoWorkMode(ctx)

        assert mode.should_continue() is False

    def test_should_continue_false_when_finish_after_current(self):
        """Test should_continue returns False after graceful shutdown request."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args
        )

        mode = StandardAutoWorkMode(ctx)
        ctx.finish_after_current = True

        assert mode.should_continue() is False

    def test_should_continue_false_when_work_limit_reached(self):
        """Test should_continue returns False when work count limit reached."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args,
            work_count_limit=5
        )

        mode = StandardAutoWorkMode(ctx)
        mode.completed_count = 5

        assert mode.should_continue() is False

    def test_should_continue_true_when_under_limit(self):
        """Test should_continue returns True when under work count limit."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args,
            work_count_limit=5
        )

        mode = StandardAutoWorkMode(ctx)
        mode.completed_count = 3

        assert mode.should_continue() is True


class TestWorkModeLifecycle:
    """Tests for work lifecycle callbacks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.wrapper = MockWrapper()
        self.args = create_mock_args()

    def test_on_work_started_stores_work_id(self):
        """Test on_work_started stores the work_id."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args
        )
        mode = StandardAutoWorkMode(ctx)

        work = {'work_id': 'test-work-123', 'composite': '12345', 'digit_length': 5}
        mode.on_work_started(work)

        assert mode.current_work_id == 'test-work-123'

    def test_on_work_completed_clears_work_id(self):
        """Test on_work_completed clears current_work_id."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args
        )
        mode = StandardAutoWorkMode(ctx)
        mode.current_work_id = 'test-work-123'

        work = {'work_id': 'test-work-123'}
        result = FactorResult()
        mode.on_work_completed(work, result)

        assert mode.current_work_id is None

    def test_on_work_completed_increments_count(self):
        """Test on_work_completed increments completed_count."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args
        )
        mode = StandardAutoWorkMode(ctx)

        assert mode.completed_count == 0

        mode.on_work_completed({'work_id': '1'}, FactorResult())
        assert mode.completed_count == 1

        mode.on_work_completed({'work_id': '2'}, FactorResult())
        assert mode.completed_count == 2

    def test_on_work_completed_resets_failure_count(self):
        """Test on_work_completed resets consecutive failure count."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args
        )
        mode = StandardAutoWorkMode(ctx)
        mode.consecutive_failures = 2

        mode.on_work_completed({'work_id': '1'}, FactorResult())

        assert mode.consecutive_failures == 0

    def test_on_work_completed_resets_graceful_shutdown(self):
        """Test on_work_completed resets graceful_shutdown_requested flag."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=self.args
        )
        mode = StandardAutoWorkMode(ctx)
        self.wrapper.graceful_shutdown_requested = True

        mode.on_work_completed({'work_id': '1'}, FactorResult())

        assert self.wrapper.graceful_shutdown_requested is False


class TestCircuitBreaker:
    """Tests for circuit breaker (consecutive failure limit)."""

    def test_max_consecutive_failures_constant(self):
        """Test MAX_CONSECUTIVE_FAILURES is defined."""
        assert MAX_CONSECUTIVE_FAILURES == 3

    def test_failure_count_increments(self):
        """Test consecutive_failures can be incremented."""
        wrapper = MockWrapper()
        ctx = WorkLoopContext(
            wrapper=wrapper,
            client_id='test',
            args=create_mock_args()
        )
        mode = StandardAutoWorkMode(ctx)

        assert mode.consecutive_failures == 0
        mode.consecutive_failures += 1
        assert mode.consecutive_failures == 1


class TestGetWorkModeFactory:
    """Tests for get_work_mode factory function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.wrapper = MockWrapper()

    def test_returns_standard_mode_by_default(self):
        """Test factory returns StandardAutoWorkMode by default."""
        args = create_mock_args()
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=args
        )

        mode = get_work_mode(ctx)

        assert isinstance(mode, StandardAutoWorkMode)

    def test_returns_composite_target_mode_when_composite_set(self):
        """Test factory returns CompositeTargetMode when composite is specified."""
        args = create_mock_args(composite='123456789')
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=args
        )

        mode = get_work_mode(ctx)

        assert isinstance(mode, CompositeTargetMode)

    def test_returns_stage1_producer_mode(self):
        """Test factory returns Stage1ProducerMode when stage1_only is True."""
        args = create_mock_args(stage1_only=True, b1=50000)
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=args
        )

        mode = get_work_mode(ctx)

        assert isinstance(mode, Stage1ProducerMode)

    def test_returns_stage2_consumer_mode(self):
        """Test factory returns Stage2ConsumerMode when stage2_only is True."""
        args = create_mock_args(stage2_only=True, b2=5000000)
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=args
        )

        mode = get_work_mode(ctx)

        assert isinstance(mode, Stage2ConsumerMode)


class TestWorkModeNames:
    """Tests for mode name constants."""

    def setup_method(self):
        """Set up test fixtures."""
        self.wrapper = MockWrapper()

    def test_standard_mode_name(self):
        """Test StandardAutoWorkMode has correct mode_name."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=create_mock_args()
        )
        mode = StandardAutoWorkMode(ctx)
        assert mode.mode_name == "Auto-work"

    def test_stage1_producer_mode_name(self):
        """Test Stage1ProducerMode has correct mode_name."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=create_mock_args(stage1_only=True, b1=50000)
        )
        mode = Stage1ProducerMode(ctx)
        assert mode.mode_name == "Stage 1 Producer (GPU)"

    def test_stage2_consumer_mode_name(self):
        """Test Stage2ConsumerMode has correct mode_name."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=create_mock_args(stage2_only=True, b2=5000000)
        )
        mode = Stage2ConsumerMode(ctx)
        assert mode.mode_name == "Stage 2 Consumer (CPU)"

    def test_composite_target_mode_name(self):
        """Test CompositeTargetMode has correct mode_name."""
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=create_mock_args(composite='123')
        )
        mode = CompositeTargetMode(ctx)
        assert mode.mode_name == "Composite Target"


class TestCleanupOnFailure:
    """Tests for cleanup_on_failure behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.wrapper = MockWrapper()
        self.wrapper.abandon_work = Mock(return_value=True)
        # No queued result for the work by default: the abandon path applies.
        self.wrapper.submission_queue.attach_work_completion.return_value = False

    def _make_mode(self) -> StandardAutoWorkMode:
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test',
            args=create_mock_args()
        )
        return StandardAutoWorkMode(ctx)

    def test_cleanup_abandons_work_if_work_id_set(self):
        """Test cleanup_on_failure abandons work when work_id is set."""
        mode = self._make_mode()
        mode.current_work_id = 'work-123'

        mode.cleanup_on_failure({'work_id': 'work-123'}, RuntimeError("test error"))

        self.wrapper.abandon_work.assert_called_once_with('work-123', reason='execution_error')
        assert mode.current_work_id is None

    def test_cleanup_does_nothing_if_no_work_id(self):
        """Test cleanup_on_failure does nothing when no work_id set."""
        mode = self._make_mode()

        mode.cleanup_on_failure(None, RuntimeError("test error"))

        self.wrapper.abandon_work.assert_not_called()

    def test_queues_abandonment_when_abandon_call_fails(self):
        """Network down and nothing queued: abandonment retries on reconnect."""
        self.wrapper.abandon_work.return_value = False
        mode = self._make_mode()
        mode.current_work_id = 'work-123'

        mode.cleanup_on_failure({'work_id': 'work-123'}, RuntimeError("test error"))

        self.wrapper.submission_queue.enqueue_work_abandonment.assert_called_once_with(
            'work-123', 'test',
        )

    def test_holds_assignment_when_result_is_queued(self):
        """
        The curves ran and only the submission failed (network down): the result
        sits in the queue, so the assignment must be held rather than released -
        abandoning would let another client re-run finished work.
        """
        self.wrapper.submission_queue.attach_work_completion.return_value = True
        mode = self._make_mode()
        mode.current_work_id = 'work-123'

        mode.cleanup_on_failure({'work_id': 'work-123'}, SubmissionFailedError("Submission failed"))

        self.wrapper.submission_queue.attach_work_completion.assert_called_once_with(
            'work-123', 'test',
        )
        self.wrapper.abandon_work.assert_not_called()
        self.wrapper.submission_queue.enqueue_work_abandonment.assert_not_called()
        assert mode.current_work_id is None  # Still cleared from in-memory state

    def test_execution_error_abandons_despite_queued_result(self):
        """
        Only a submission failure may hold. T-level mode submits per B1 batch,
        so an earlier batch's result can be sitting in the queue when a later
        batch crashes - that assignment is partially executed and must be
        released, not held and later marked complete.
        """
        self.wrapper.submission_queue.attach_work_completion.return_value = True
        mode = self._make_mode()
        mode.current_work_id = 'work-123'

        mode.cleanup_on_failure({'work_id': 'work-123'}, RuntimeError("ECM binary crashed"))

        self.wrapper.submission_queue.attach_work_completion.assert_not_called()
        self.wrapper.abandon_work.assert_called_once_with('work-123', reason='execution_error')

    def test_keyboard_interrupt_abandons_despite_queued_result(self):
        """A hard Ctrl+C releases the composite immediately rather than holding
        it for the full server-side expiry while this client is gone."""
        self.wrapper.submission_queue.attach_work_completion.return_value = True
        mode = self._make_mode()
        mode.current_work_id = 'work-123'

        mode.cleanup_on_failure({'work_id': 'work-123'}, KeyboardInterrupt())

        self.wrapper.submission_queue.attach_work_completion.assert_not_called()
        self.wrapper.abandon_work.assert_called_once_with('work-123', reason='execution_error')


class TestStage2CleanupOnFailure:
    """Stage2ConsumerMode.cleanup_on_failure: hold-vs-abandon decision."""

    def setup_method(self):
        self.wrapper = MockWrapper()

    def _make_mode(self, has_pending: bool) -> Stage2ConsumerMode:
        self.wrapper.submission_queue.has_pending_result_for_residue.return_value = has_pending
        self.wrapper.submission_queue.enqueue_residue_abandonment = Mock()
        self.wrapper.api_client.abandon_residue = Mock(return_value=True)
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test-client',
            args=create_mock_args(stage2_only=True, b2=5000000),
        )
        mode = Stage2ConsumerMode(ctx)
        mode.current_residue_id = 32225
        return mode

    def test_holds_claim_when_result_is_queued(self):
        mode = self._make_mode(has_pending=True)
        mode.cleanup_on_failure({}, RuntimeError("submit failed"))

        # Claim must NOT be released - the queued result needs to finalize it.
        self.wrapper.api_client.abandon_residue.assert_not_called()
        self.wrapper.submission_queue.enqueue_residue_abandonment.assert_not_called()
        assert mode.current_residue_id is None  # Still cleared from in-memory state

    def test_abandons_when_no_pending_result(self):
        mode = self._make_mode(has_pending=False)
        mode.cleanup_on_failure({}, RuntimeError("execution failed"))

        self.wrapper.api_client.abandon_residue.assert_called_once_with(
            'test-client', 32225,
        )

    def test_queues_abandonment_when_abandon_call_fails(self):
        mode = self._make_mode(has_pending=False)
        self.wrapper.api_client.abandon_residue.return_value = False  # Network down

        mode.cleanup_on_failure({}, RuntimeError("execution failed"))

        self.wrapper.submission_queue.enqueue_residue_abandonment.assert_called_once_with(
            32225, 'test-client',
        )


class TestStage1CleanupOnFailure:
    """Stage1ProducerMode.cleanup_on_failure: hold-vs-abandon decision."""

    def setup_method(self):
        self.wrapper = MockWrapper()
        self.wrapper.abandon_work = Mock(return_value=True)

    def _make_mode(self, has_pending: bool) -> Stage1ProducerMode:
        self.wrapper.submission_queue.has_pending_result_for_work.return_value = has_pending
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test-client',
            args=create_mock_args(stage1_only=True, b1=110000000, curves=3000),
        )
        mode = Stage1ProducerMode(ctx)
        mode.current_work_id = 'work-123'
        return mode

    def test_holds_assignment_when_stage1_result_is_queued(self):
        """
        A multi-hour GPU batch whose submission failed is queued together with
        its residue; releasing the composite would have another GPU redo it.
        """
        mode = self._make_mode(has_pending=True)

        mode.cleanup_on_failure({}, SubmissionFailedError("Submission failed"))

        self.wrapper.abandon_work.assert_not_called()
        self.wrapper.submission_queue.enqueue_work_abandonment.assert_not_called()
        assert mode.current_work_id is None

    def test_execution_error_abandons(self):
        mode = self._make_mode(has_pending=True)

        mode.cleanup_on_failure({}, RuntimeError("GPU stage 1 crashed"))

        self.wrapper.abandon_work.assert_called_once_with('work-123', reason='stage1_failed')

    def test_abandons_when_nothing_is_queued(self):
        mode = self._make_mode(has_pending=False)

        mode.cleanup_on_failure({}, SubmissionFailedError("Submission failed"))

        self.wrapper.abandon_work.assert_called_once_with('work-123', reason='stage1_failed')

    def test_queues_abandonment_when_abandon_call_fails(self):
        mode = self._make_mode(has_pending=False)
        self.wrapper.abandon_work.return_value = False  # Network down

        mode.cleanup_on_failure({}, RuntimeError("execution failed"))

        self.wrapper.submission_queue.enqueue_work_abandonment.assert_called_once_with(
            'work-123', 'test-client',
        )


class TestAdaptiveStage2CleanupOnFailure:
    """AdaptiveMode's stage-2 branch must hold the claim like Stage2ConsumerMode."""

    def setup_method(self):
        self.wrapper = MockWrapper()
        self.wrapper.api_client.abandon_residue = Mock(return_value=True)

    def _make_mode(self, has_pending: bool):
        from lib.work_modes import AdaptiveCPUMode
        self.wrapper.submission_queue.has_pending_result_for_residue.return_value = has_pending
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test-client',
            args=create_mock_args(adaptive=True, workers=2),
        )
        mode = AdaptiveCPUMode(ctx)
        mode._current_mode = 'stage2'
        mode.current_residue_id = 32225
        return mode

    def test_holds_claim_when_result_is_queued(self):
        mode = self._make_mode(has_pending=True)

        mode.cleanup_on_failure({}, RuntimeError("submit failed"))

        self.wrapper.api_client.abandon_residue.assert_not_called()
        self.wrapper.submission_queue.enqueue_residue_abandonment.assert_not_called()
        assert mode.current_residue_id is None

    def test_abandons_when_no_pending_result(self):
        mode = self._make_mode(has_pending=False)

        mode.cleanup_on_failure({}, RuntimeError("execution failed"))

        self.wrapper.api_client.abandon_residue.assert_called_once_with(
            'test-client', 32225,
        )


class TestTLevelSubmissionFailure:
    """
    T-level mode submits per B1 batch inside the executor, so submit_results is
    the only place a queued batch can be reported to the work loop.
    """

    def setup_method(self):
        self.wrapper = MockWrapper()

    def _make_mode(self) -> StandardAutoWorkMode:
        ctx = WorkLoopContext(
            wrapper=self.wrapper,
            client_id='test-client',
            args=create_mock_args(),
        )
        mode = StandardAutoWorkMode(ctx)
        mode._is_tlevel_mode = True
        return mode

    def test_queued_batch_reports_failure(self):
        """
        Returning True here completes the assignment with no t-level progress
        recorded, and /ecm-work hands the same composite straight back out.
        """
        mode = self._make_mode()
        result = FactorResult(success=True, curves_run=1200)
        result.submission_failed = True

        assert mode.submit_results({'work_id': 'work-123'}, result) is False

    def test_successful_batches_report_success(self):
        mode = self._make_mode()
        result = FactorResult(success=True, curves_run=1200)

        assert mode.submit_results({'work_id': 'work-123'}, result) is True


def main():
    """Run all tests."""
    print("Running work_modes tests...\n")
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    main()

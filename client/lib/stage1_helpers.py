#!/usr/bin/env python3
"""
Helper functions for Stage 1 ECM result submission and residue management.

Consolidates the complete stage1 submission workflow that appears multiple times
in the codebase (auto-work, manual mode, two-stage pipeline).
"""
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


def submit_stage1_complete_workflow(
    wrapper,
    results: Dict[str, Any],
    residue_file: Path,
    work_id: Optional[str],
    project: Optional[str],
    client_id: str,
    factor_found: Optional[str],
    cleanup_residue: bool = True,
    upload_residue: bool = True
) -> Optional[str]:
    """
    Complete Stage 1 submission workflow: submit results, upload residue, cleanup.

    This consolidates the full workflow that appears in multiple locations:
    - Submit stage1 results to API
    - Handle submission failures (abandon work, cleanup)
    - Extract attempt_id from response
    - Upload residue file if no factor found
    - Clean up local residue file

    Args:
        wrapper: ECMWrapper instance (for API client and logger)
        results: Stage1 results dictionary (from _build_stage1_results())
        residue_file: Path to residue file
        work_id: Optional work assignment ID (for auto-work mode)
        project: Optional project name
        client_id: Client identifier
        factor_found: Factor string if found, None otherwise
        cleanup_residue: Whether to delete local residue file after processing
        upload_residue: Whether to upload residue to server. Set False when the
            caller has determined the residue will not be used (e.g., stage 1
            alone already met the composite's target t-level).

    Returns:
        Stage1 attempt_id from API response, or None if submission failed

    Side Effects:
        - Submits results to API
        - May abandon work assignment on failure
        - May upload residue file to server
        - May delete local residue file
    """
    print("Submitting stage 1 results...")
    program_name = 'gmp-ecm-ecm'

    # If the result submission fails (e.g. a deploy/outage), the residue is the
    # valuable product of a long GPU batch and must not be thrown away. Attach a
    # residue-upload completion chain: when the queued result is retried and
    # returns its attempt_id, the preserved residue is uploaded and linked. The
    # residue is only needed when there's no factor and the caller wants it
    # uploaded; enqueue_result() copies the file into the queue at failure time.
    completion_chain = None
    if upload_residue and not factor_found and residue_file.exists():
        completion_chain = {
            "action": "residue_upload",
            "residue_file": str(residue_file),
            "client_id": client_id,
            "expiry_days": 7,
        }

    # Submit stage1 results to API
    submit_response = wrapper.submit_result(
        results, project, program_name, completion_chain=completion_chain
    )

    if not submit_response:
        # Transient failure: submit_result queued the result for automatic retry,
        # and (via the chain) preserved the residue alongside it. Do NOT abandon
        # the work here - in auto-work mode the work loop's cleanup_on_failure
        # releases the assignment; in manual mode there is no assignment. Removing
        # the local residue is safe because enqueue_result copied it into the queue.
        wrapper.logger.error(
            "Failed to submit stage 1 results - result and residue queued for retry"
        )

        if cleanup_residue and residue_file.exists():
            residue_file.unlink()

        return None

    # Extract attempt_id from response
    stage1_attempt_id = submit_response.get('attempt_id')
    if stage1_attempt_id:
        print(f"Stage 1 attempt ID: {stage1_attempt_id}")

    # Upload residue file if needed (skips if factor found or caller opted out)
    if upload_residue:
        wrapper._upload_residue_if_needed(
            residue_file=residue_file,
            stage1_attempt_id=stage1_attempt_id,
            factor_found=factor_found,
            client_id=client_id
        )
    else:
        print("Skipping residue upload: stage 1 already met target t-level")

    # Clean up local residue file
    if cleanup_residue and residue_file.exists():
        residue_file.unlink()

    return stage1_attempt_id


def handle_stage1_failure(
    wrapper,
    work_id: Optional[str],
    residue_file: Optional[Path],
    error_msg: str
) -> None:
    """
    Handle stage1 execution failure: log error, abandon work, cleanup.

    Args:
        wrapper: ECMWrapper instance
        work_id: Optional work assignment ID to abandon
        residue_file: Optional residue file path to cleanup
        error_msg: Error message to log
    """
    wrapper.logger.error(error_msg)

    # Abandon work assignment if applicable
    if work_id:
        wrapper.abandon_work(work_id, reason="execution_error")

    # Clean up residue file if it exists
    if residue_file and residue_file.exists():
        residue_file.unlink()

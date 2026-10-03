"""Adaptive stage 2 must execute the bounds advertised by dictionary lookup."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.work_args import WorkArgs
from lib.work_modes import AdaptiveCPUMode, WorkLoopContext


@pytest.mark.parametrize('dictionary, explicit_b2, multiplier, expected_b2, expected_k', [
    ({'b2_from_dict': 300_000_000}, 999_000_000, 500, 300_000_000, None),
    ({'b2_from_dict': 300_000_000, 'k_from_dict': 4}, None, None, 300_000_000, 4),
    ({'b2_from_dict': 300_000_000, 'k_from_dict': 0}, None, None, 300_000_000, None),
    ({}, -1, 500, -1, None),
    ({}, None, 100, 300_000_000, None),
    ({}, None, None, 1_500_000_000, None),
])
def test_selected_bounds_reach_stage2_executor(
    tmp_path, dictionary, explicit_b2, multiplier, expected_b2, expected_k,
):
    wrapper = Mock()
    wrapper._get_api_client.return_value = wrapper.api_client
    wrapper.typed_config.execution.residue_dir = str(tmp_path)
    args = WorkArgs(workers=8, b2=explicit_b2, b2_multiplier=multiplier)
    mode = AdaptiveCPUMode(WorkLoopContext(wrapper=wrapper, client_id='consumer', args=args))
    mode._current_mode = 'stage2'
    work = {
        'residue_id': 123, 'composite': '1234567891', 'digit_length': 10,
        'b1': 3_000_000, 'curve_count': 16, 'suggested_b2': 1_500_000_000,
        **dictionary,
    }

    def download(**kwargs):
        Path(kwargs['output_path']).write_text('test residue\n')
        return True

    wrapper.api_client.download_residue.side_effect = download
    executor = Mock()
    executor.execute.return_value = (None, [], 16, 1.0, None)
    executor.raw_output = 'test output'
    mode.Stage2Executor = Mock(return_value=executor)

    # The base work loop attaches dictionary entries before this callback.
    mode.on_work_started(work)
    result = mode.execute_work(work)

    assert result.success
    assert result.curves_run == 16
    command_args = mode.Stage2Executor.call_args.args
    assert command_args[2:6] == (3_000_000, expected_b2, expected_k, 8)

    # A dictionary miss on the next assignment must not retain a prior k.
    next_work = {k: v for k, v in work.items() if k not in ('b2_from_dict', 'k_from_dict')}
    mode.on_work_started(next_work)
    assert mode._s2_k is None

"""Stage-1 work requests describe the B1 selection without forcing one day."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.api_client import APIClient
from lib.work_args import WorkArgs
from lib.work_modes import Stage1ProducerMode, WorkLoopContext


@pytest.mark.parametrize('days', [None, 1, 2, 5])
def test_api_sends_only_explicit_timeout(days):
    response = Mock()
    response.json.return_value = {'message': 'No work available'}
    with patch('lib.api_client.requests.get', return_value=response) as get:
        APIClient('https://example.invalid/api/v1').get_ecm_work(
            'producer', timeout_days=days,
        )
    params = get.call_args.kwargs['params']
    if days is None:
        assert 'timeout_days' not in params
    else:
        assert params['timeout_days'] == days
    assert 'stage1_only' not in params
    assert 'requested_b1' not in params


@pytest.mark.parametrize('current_t, override_b1, expected_b1', [
    (59.99, None, 260_000_000),
    (60.0, None, 850_000_000),
    (64.99, None, 850_000_000),
    (65.0, None, 2_900_000_000),
    (0.0, 850_000_000, 850_000_000),
    (0.0, 2_900_000_000, 2_900_000_000),
    (65.0, 43_000_000, 43_000_000),
])
def test_stage1_request_passes_mode_and_override(current_t, override_b1, expected_b1):
    response = Mock()
    response.json.return_value = {
        'work_id': 'stage1-work', 'composite': '123456789', 'digit_length': 9,
        'current_t_level': current_t, 'target_t_level': 85.0,
    }
    wrapper = Mock()
    wrapper._get_api_client.return_value = APIClient('https://example.invalid/api/v1')
    wrapper.typed_config.programs.gmp_ecm.gpu.curves_per_batch = 1000
    wrapper.typed_config.programs.gmp_ecm.default_curves = 1000
    mode = Stage1ProducerMode(WorkLoopContext(
        wrapper=wrapper, client_id='producer',
        args=WorkArgs(stage1_only=True, b1=override_b1),
    ))

    with patch('lib.api_client.requests.get', return_value=response) as get:
        work = mode.request_work()
    params = get.call_args.kwargs['params']
    assert params['stage1_only'] is True
    assert 'timeout_days' not in params
    if override_b1 is None:
        assert 'requested_b1' not in params
    else:
        assert params['requested_b1'] == override_b1
    assert work['target_t_level'] == 85.0
    assert mode._calculate_stage1_params(work)[0] == expected_b1

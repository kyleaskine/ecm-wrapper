"""The client leaves B1-based claim defaults to the server."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.api_client import APIClient
from lib.work_args import WorkArgs
from lib.work_modes import AdaptiveCPUMode, Stage2ConsumerMode, WorkLoopContext


@pytest.mark.parametrize('hours', [None, 48, 120])
def test_api_sends_only_explicit_claim_timeout(hours):
    response = Mock()
    response.json.return_value = {'message': 'No residues available'}
    with patch('lib.api_client.requests.get', return_value=response) as get:
        APIClient('https://example.invalid/api/v1').get_residue_work(
            'consumer', min_b1=850_000_000, claim_timeout_hours=hours,
        )
    params = get.call_args.kwargs['params']
    assert params['min_b1'] == 850_000_000
    if hours is None:
        assert 'claim_timeout_hours' not in params
    else:
        assert params['claim_timeout_hours'] == hours


@pytest.mark.parametrize('mode_type', [Stage2ConsumerMode, AdaptiveCPUMode])
def test_work_modes_do_not_override_server_default(mode_type):
    wrapper = Mock()
    wrapper.typed_config = None
    wrapper._get_api_client.return_value = wrapper.api_client
    wrapper.api_client.get_residue_work.return_value = {'residue_id': 123, 'b1': 2_900_000_000}
    ctx = WorkLoopContext(
        wrapper=wrapper, client_id='consumer',
        args=WorkArgs(workers=8, max_b1=2_900_000_000),
    )
    work = mode_type(ctx).request_work()
    assert work['residue_id'] == 123
    assert 'claim_timeout_hours' not in wrapper.api_client.get_residue_work.call_args.kwargs

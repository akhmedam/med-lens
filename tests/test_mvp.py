"""Shipping-policy identity and dependency boundaries, without model execution."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from medlens.api_contract import SearchRequest
from medlens.mvp import (POLICY_ID, ANSWER_STYLE, ANSWER_MODEL, INFERENCE_ENGINE,
                         EMBEDDING_DEVICE, policy_snapshot)


class MVPPolicyTests(unittest.TestCase):
    def test_snapshot_matches_the_accepted_request_and_fixed_runtime(self):
        snapshot = policy_snapshot()
        request = SearchRequest('What is known about inflammation?')
        self.assertEqual(snapshot['id'], POLICY_ID)
        self.assertEqual(snapshot['source']['candidate_limit']['default'], request.limit)
        self.assertEqual(snapshot['retrieval']['top_k']['default'], request.top_k)
        self.assertEqual(snapshot['answer']['style'], request.answer_style)
        self.assertEqual((ANSWER_STYLE, ANSWER_MODEL, INFERENCE_ENGINE, EMBEDDING_DEVICE),
                         ('source-findings', '4b', 'mlx', 'mps'))
        self.assertFalse(snapshot['answer']['independent_validation'])
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)

    def test_snapshot_mutation_does_not_change_later_job_policy(self):
        expected = policy_snapshot()
        client_copy = policy_snapshot()
        client_copy['answer']['style'] = 'draft'
        client_copy['source']['candidate_limit']['max'] = 999
        client_copy['retrieval']['top_k']['default'] = 1
        self.assertEqual(policy_snapshot(), expected)

    def test_isolated_policy_import_does_not_load_pipeline_or_http_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        code = '''import json,sys
sys.path.insert(0, SRC)
from medlens.mvp import policy_snapshot
snapshot=policy_snapshot()
assert snapshot['answer']['style']=='source-findings'
assert snapshot['answer']['independent_validation'] is False
assert not any(name.startswith(('numpy','torch','transformers','mlx','fastapi','uvicorn',
    'medlens.api','medlens.live','medlens.pipeline','medlens.inference')) for name in sys.modules)
print(json.dumps(snapshot,allow_nan=False))
'''.replace('SRC', repr(str(root / 'src')))
        child = subprocess.run([sys.executable, '-I', '-S', '-c', code],
                               capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), policy_snapshot())


if __name__ == '__main__':
    unittest.main()

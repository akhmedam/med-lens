"""Output validation must reject malformed reranker responses."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("smoke", Path(__file__).resolve().parents[1] / "scripts/smoke_inference.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class RerankerOutputTests(unittest.TestCase):
    def test_user_agent_fix_keeps_other_headers(self):
        request = SimpleNamespace(headers={"user-agent": "kernels/test; ", "accept": "application/json"})
        smoke.trim_user_agent(request)
        self.assertEqual(request.headers, {"user-agent": "kernels/test;", "accept": "application/json"})

    def test_accepts_only_complete_permutations(self):
        ids = ["D1", "D2", "D3"]
        self.assertTrue(smoke.valid_permutation('["D3", "D1", "D2"]', ids))
        for invalid in ('["D1", "D1", "D3"]', '["D1", "D2"]',
                        '["D1", "D2", "D4"]', '{"ids": ["D1", "D2", "D3"]}',
                        '[1, 2, 3]', '```json\n["D1", "D2", "D3"]\n```'):
            with self.subTest(output=invalid):
                self.assertFalse(smoke.valid_permutation(invalid, ids))


if __name__ == "__main__":
    unittest.main()

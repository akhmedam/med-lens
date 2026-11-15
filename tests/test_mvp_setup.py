"""MVP asset preparation never requires research-only downloads."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load_tool():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare_models.py'
    spec = importlib.util.spec_from_file_location('prepare_mvp_assets', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MVPSetupTests(unittest.TestCase):
    def fixture(self, root):
        models = {}
        for name in ('bge', 'bge_tokenizer'):
            directory = root / 'assets' / name
            directory.mkdir(parents=True)
            body = (name + ' payload').encode()
            (directory / 'model.bin').write_bytes(body)
            models[name] = {'repo_id': 'fixture/' + name, 'revision': 'pinned',
                'directory': str(directory.relative_to(root)),
                'files': {'model.bin': {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}}}
        # These intentionally unavailable research assets must not be touched.
        models['qwen'] = {'directory': 'missing9b', 'files': {'huge.bin': {'bytes': 100, 'sha256': '0' * 64}}}
        models['medcpt'] = {'directory': 'missing-medcpt', 'files': {}}
        pins = {'schema_version': 1, 'models': models,
            'metal_kernel': {'directory': 'missing-kernel', 'files': {'kernel': {'bytes': 3, 'sha256': '0' * 64}}}}
        (root / 'configs').mkdir()
        (root / 'configs/model-assets.json').write_text(json.dumps(pins))
        return pins

    def test_default_profile_prepares_only_bge_and_tokenizer_offline(self):
        tool = load_tool()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            output = root / 'report.json'
            with patch.object(tool, 'ROOT', root):
                self.assertEqual(tool.main(['--output', str(output)]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result['profile'], 'mvp')
            self.assertEqual([item['name'] for item in result['models']], ['bge', 'bge_tokenizer'])
            self.assertNotIn('metal_kernel', result)
            self.assertFalse((root / 'missing9b').exists())
            self.assertFalse((root / 'missing-medcpt').exists())
            self.assertFalse((root / 'missing-kernel').exists())

    def test_research_profile_retains_explicit_missing_asset_failure(self):
        tool = load_tool()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch.object(tool, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Missing assets'):
                tool.main(['--profile', 'research', '--output', str(root / 'report.json')])

    def test_mvp_cannot_silently_omit_the_tokenizer(self):
        tool = load_tool()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = self.fixture(root)
            del pins['models']['bge_tokenizer']
            (root / 'configs/model-assets.json').write_text(json.dumps(pins))
            with patch.object(tool, 'ROOT', root), self.assertRaisesRegex(ValueError, 'model and tokenizer'):
                tool.main(['--output', str(root / 'report.json')])


if __name__ == '__main__':
    unittest.main()

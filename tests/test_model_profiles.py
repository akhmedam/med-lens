"""Pinned model/tokenizer selection without native imports or weight dependencies."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medlens.inference import profiles


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.payloads = {'tokenizer.json':b'{"fictional":true}', 'tokenizer_config.json':b'{}',
                         'chat_template.jinja':b'FICTIONAL TEMPLATE'}
        self.files = {name:{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
                      for name,body in self.payloads.items()}
        patched = patch.object(profiles, 'TOKENIZER_FILES', self.files);patched.start();self.addCleanup(patched.stop)

    def prepare(self, model='4b'):
        profile = profiles.get_profile(model)
        directory = self.root/profile['tokenizer']['directory'];directory.mkdir(parents=True)
        for name,body in self.payloads.items():(directory/name).write_bytes(body)
        if model == '4b':(directory/'tokenizer-source.json').write_bytes(profiles.canonical(profiles.tokenizer_manifest(profile)))
        return directory

    def test_default_selection_and_copied_nested_profiles(self):
        self.assertEqual(profiles.DEFAULT_MODEL,'4b')
        profile = profiles.get_profile();self.assertEqual(profile['model_id'],'mlx-community/Qwen3.5-4B-MLX-4bit')
        self.assertEqual(profile['answer_policy'],'compact-v1')
        self.assertIsNone(profile['upstream_revision'])
        profile['quantization']['bits'] = 8;profile['tokenizer']['revision'] = 'changed'
        self.assertEqual(profiles.get_profile()['quantization']['bits'],4)
        self.assertEqual(profiles.get_profile('9b')['answer_policy'],'legacy-v6')
        for name in ('4B','default','Qwen/Qwen3.5-4B','',None):
            with self.assertRaises(ValueError):profiles.get_profile(name)

    def test_standalone_tokenizer_needs_no9b_files_and_records_official_identity(self):
        directory = self.prepare()
        token = SimpleNamespace()
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=token) as load:
            self.assertIs(profiles.load_answer_tokenizer(self.root), token)
            load.assert_called_once_with(directory,local_files_only=True,token=False,trust_remote_code=False)
        self.assertEqual(token._medlens_tokenizer_source['model_id'],'Qwen/Qwen3.5-4B')
        self.assertEqual(token._medlens_tokenizer_source['revision'],'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
        self.assertFalse((self.root/'.cache/huggingface').exists())

    def test_tokenizer_manifest_hash_precision_and_path_conflicts_reject_before_loading(self):
        directory = self.prepare()
        (directory/'tokenizer.json').write_bytes(b'{"fictional":false}')
        with patch('transformers.AutoTokenizer.from_pretrained') as load:
            with self.assertRaisesRegex(ValueError,'checksum'):profiles.load_answer_tokenizer(self.root)
            load.assert_not_called()
        (directory/'tokenizer.json').write_bytes(self.payloads['tokenizer.json'])
        value = profiles.tokenizer_manifest(profiles.get_profile());value['revision'] = 'other'
        (directory/'tokenizer-source.json').write_bytes(profiles.canonical(value))
        with self.assertRaisesRegex(ValueError,'identity'):profiles.verify_tokenizer(self.root)

    def test_standalone_tokenizer_symlink_and_duplicate_metadata_rejected(self):
        directory = self.prepare();name='tokenizer.json'
        real=self.root/'tokenizer-copy';(directory/name).rename(real);(directory/name).symlink_to(real)
        with self.assertRaisesRegex(ValueError,'regular'):profiles.verify_tokenizer(self.root)
        (directory/name).unlink();(directory/name).write_bytes(self.payloads[name])
        (directory/'tokenizer-source.json').write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):profiles.verify_tokenizer(self.root)

    def test_legacy9b_snapshot_is_explicit_and_within_checkout_cache_links_work(self):
        directory = self.prepare('9b')
        real=self.root/'legacy-tokenizer';(directory/'tokenizer.json').rename(real);(directory/'tokenizer.json').symlink_to(real)
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=SimpleNamespace()) as load:
            token=profiles.load_answer_tokenizer(self.root,'9b')
            self.assertEqual(token._medlens_tokenizer_source['model_id'],'Qwen/Qwen3.5-9B')
            self.assertEqual(load.call_args.args[0],directory)
        (directory/'tokenizer.json').unlink();(directory/'tokenizer.json').symlink_to('/etc/hosts')
        with self.assertRaisesRegex(ValueError,'escaping'):profiles.verify_tokenizer(self.root,'9b')


if __name__ == '__main__':unittest.main()

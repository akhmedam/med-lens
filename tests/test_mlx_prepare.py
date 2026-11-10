"""Pinned provenance and corrupt-asset rejection without loading MLX or model weights."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from medlens.inference.mlx_convert import (COVERAGES, MODEL_ID, REVISION, canonical,
    sha_file, tensor_headers, validate_converted, verify_source, write_manifest)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self):
        directory = self.root/'source'; directory.mkdir()
        (directory/'config.json').write_bytes(b'{"test":"source"}')
        files = {'config.json': {'bytes': 17, 'sha256': sha_file(directory/'config.json')}}
        return {'repo_id': MODEL_ID, 'revision': REVISION, 'directory': 'source', 'files': files}

    def shard(self, name='model.safetensors', header=None, body=b'\x00\x00'):
        if header is None:
            header = {'tiny': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}
        encoded = canonical(header)
        path = self.root/name
        path.write_bytes(struct.pack('<Q', len(encoded))+encoded+body)
        return path

    def converted(self, coverage=COVERAGES[0]):
        (self.root/'config.json').write_text(json.dumps({'model_type': 'qwen3_5',
            'quantization': {'bits': 4, 'group_size': 64, 'mode': 'affine'},
            'text_config': {'dtype': 'float16'}}))
        dtype = 'U32' if coverage == COVERAGES[0] else 'F16'
        shape = [248320, 512 if dtype == 'U32' else 4096]
        headers = {'language_model.model.embed_tokens.weight': {'dtype': dtype, 'shape': shape},
                   'language_model.lm_head.weight': {'dtype': dtype, 'shape': shape}}
        for i in range(32):
            if i % 4 != 3:
                headers[f'language_model.model.layers.{i}.linear_attn.A_log'] = {'dtype': 'F32', 'shape': [32]}
                headers[f'language_model.model.layers.{i}.linear_attn.dt_bias'] = {'dtype': 'F16', 'shape': [32]}
        return headers

    def test_source_is_hashed_and_does_not_change_pinned_identity(self):
        spec = self.source()
        result = verify_source(self.root, spec)
        self.assertEqual(result['model_id'], MODEL_ID)
        self.assertEqual(result['revision'], REVISION)
        self.assertEqual(result['validation'], 'all-source-file-sizes-and-sha256')
        self.assertEqual(result['files'], spec['files'])
        (self.root/'source/config.json').write_bytes(b'{"test":"change"}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify_source(self.root, spec)

    def test_source_in_checkout_symlink_allowed_but_escape_and_other_revision_rejected(self):
        spec = self.source()
        real = self.root/'blob'; (self.root/'source/config.json').rename(real)
        (self.root/'source/config.json').symlink_to(real)
        verify_source(self.root, spec)
        spec['revision'] = 'other'
        with self.assertRaisesRegex(ValueError, 'pinned Qwen'):
            verify_source(self.root, spec)
        spec['revision'] = REVISION
        (self.root/'source/config.json').unlink()
        (self.root/'source/config.json').symlink_to('/etc/hosts')
        with self.assertRaisesRegex(ValueError, 'escaping source'):
            verify_source(self.root, spec)

    def test_source_missing_and_filename_traversal_rejected(self):
        spec = self.source(); (self.root/'source/config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            verify_source(self.root, spec)
        spec['files'] = {'../config.json': {'bytes': 0, 'sha256': '0'*64}}
        with self.assertRaisesRegex(ValueError, 'filename'):
            verify_source(self.root, spec)

    def test_header_reader_rejects_truncation_unsupported_dtype_and_bad_offsets(self):
        path = self.shard(); self.assertEqual(tensor_headers(self.root)['tiny']['dtype'], 'F16')
        path.write_bytes(b'not-long')
        with self.assertRaisesRegex(ValueError, 'header size'):
            tensor_headers(self.root)
        for value in ({'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]},
                      {'dtype': 'F16', 'shape': [2], 'data_offsets': [0, 2]},
                      {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 4]}):
            with self.subTest(value=value):
                self.shard(header={'tiny': value})
                with self.assertRaises(ValueError): tensor_headers(self.root)

    def test_duplicate_and_overlapping_tensors_rejected(self):
        self.shard(); second = self.shard(name='second.safetensors')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            tensor_headers(self.root)
        second.unlink()
        self.shard(header={'a': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]},
                           'b': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}})
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            tensor_headers(self.root)

    def test_converted_coverage_and_recurrent_precision_are_explicit(self):
        for coverage in COVERAGES:
            with self.subTest(coverage=coverage):
                tensors = self.converted(coverage)
                with patch('medlens.inference.mlx_convert.tensor_headers', return_value=tensors):
                    result = validate_converted(self.root, coverage)
                    self.assertEqual(result['a_log_count'], 24)
                    wrong = COVERAGES[1] if coverage == COVERAGES[0] else COVERAGES[0]
                    with self.assertRaisesRegex(ValueError, 'coverage'):
                        validate_converted(self.root, wrong)
                    tensors['language_model.model.layers.0.linear_attn.A_log']['dtype'] = 'F16'
                    with self.assertRaisesRegex(ValueError, 'recurrent'):
                        validate_converted(self.root, coverage)

    def test_config_false_dtype_undeclared_precision_and_index_mismatch_rejected(self):
        tensors = self.converted()
        with patch('medlens.inference.mlx_convert.tensor_headers', return_value=tensors):
            config_path = self.root/'config.json'; original = config_path.read_bytes()
            config = json.loads(original); config['text_config']['dtype'] = 'bfloat16'
            config_path.write_bytes(canonical(config))
            with self.assertRaisesRegex(ValueError, 'declared policy'):
                validate_converted(self.root, COVERAGES[0])
            config_path.write_bytes(original)
            tensors['unexpected.weight'] = {'dtype': 'F32', 'shape': [1]}
            with self.assertRaisesRegex(ValueError, 'Undeclared FP32'):
                validate_converted(self.root, COVERAGES[0])
            tensors.pop('unexpected.weight')
            (self.root/'model.safetensors.index.json').write_text('{"weight_map":{}}')
            with self.assertRaisesRegex(ValueError, 'index mismatch'):
                validate_converted(self.root, COVERAGES[0])

    def test_output_symlinks_or_subdirectories_rejected(self):
        self.converted(); (self.root/'linked').symlink_to(self.root/'config.json')
        with self.assertRaisesRegex(ValueError, 'ordinary flat'):
            validate_converted(self.root, COVERAGES[0])
        (self.root/'linked').unlink(); (self.root/'nested').mkdir()
        with self.assertRaisesRegex(ValueError, 'ordinary flat'):
            validate_converted(self.root, COVERAGES[0])

    def test_manifest_identity_covers_conversion_source_policy_and_all_files(self):
        self.shard(); (self.root/'config.json').write_text('{}')
        source = {'model_id': MODEL_ID, 'revision': REVISION, 'directory': '/local/private/path',
                  'files': {'source': {'bytes': 3, 'sha256': 'a'*64}}}
        with patch('medlens.inference.mlx_convert.validate_converted', return_value={'fixture': True}):
            value = write_manifest(self.root, source, COVERAGES[0], {'mlx': '0.32.2'}, {'seconds': 1})
        self.assertNotIn('directory', value['source'])
        self.assertEqual(set(value['files']), {'config.json', 'model.safetensors'})
        self.assertEqual(value['files']['model.safetensors']['sha256'], sha_file(self.root/'model.safetensors'))
        identity = value.pop('artifact_id')
        self.assertEqual(identity, hashlib.sha256(canonical(value)).hexdigest())
        value['quantization']['coverage'] = COVERAGES[1]
        self.assertNotEqual(identity, hashlib.sha256(canonical(value)).hexdigest())

    def test_driver_keeps_venv_executable_symlink_and_offline_environment(self):
        script_path = Path(__file__).resolve().parents[1]/'scripts/prepare_mlx.py'
        spec = importlib.util.spec_from_file_location('prepare_mlx_fixture', script_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        source = self.source()
        config = self.root/'configs'; config.mkdir()
        (config/'model-assets.json').write_bytes(canonical({'models': {'qwen': source}}))
        executable = self.root/'venv/bin/python'; executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)
        output = self.root/'converted'
        def fake_native(command, **options):
            self.assertEqual(command[0], str(executable.absolute()))
            self.assertNotEqual(command[0], str(executable.resolve()))
            self.assertEqual(command[1], '-I')
            self.assertEqual(options['env']['HF_HUB_OFFLINE'], '1')
            self.assertEqual(options['env']['TRANSFORMERS_OFFLINE'], '1')
            self.assertEqual(options['env']['HF_HUB_DISABLE_IMPLICIT_TOKEN'], '1')
            prepared = Path(command[command.index('--output')+1]); prepared.mkdir()
            (prepared/'config.json').write_bytes(b'{}')
            value = {'schema_version': 1, 'kind': 'medlens-mlx-model', 'status': 'complete',
                'files': {'config.json': {'bytes': 2, 'sha256': sha_file(prepared/'config.json')}}}
            value['artifact_id'] = hashlib.sha256(canonical(value)).hexdigest()
            (prepared/'medlens-source.json').write_bytes(canonical(value))
        with patch.object(module, 'ROOT', self.root), patch.object(module.subprocess, 'run', side_effect=fake_native):
            self.assertEqual(module.main(['--worker-python', str(executable), '--output', str(output)]), 0)
        self.assertTrue((output/'medlens-source.json').is_file())
        states = list(self.root.glob('.converted.preparing-*/preparation-status.json'))
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0].read_bytes())['status'], 'complete')

    def test_invalid_conversion_does_not_publish_complete_manifest(self):
        self.shard(); (self.root/'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'declared policy'):
            write_manifest(self.root, {}, COVERAGES[0], {}, {})
        self.assertFalse((self.root/'medlens-source.json').exists())


if __name__ == '__main__':
    unittest.main()

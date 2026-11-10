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
            self.assertEqual(module.main(['--model', '9b', '--worker-python', str(executable), '--output', str(output)]), 0)
        self.assertTrue((output/'medlens-source.json').is_file())
        states = list(self.root.glob('.converted.preparing-*/preparation-status.json'))
        self.assertEqual(len(states), 1)
        self.assertEqual(json.loads(states[0].read_bytes())['status'], 'complete')

    def test_invalid_conversion_does_not_publish_complete_manifest(self):
        self.shard(); (self.root/'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'declared policy'):
            write_manifest(self.root, {}, COVERAGES[0], {}, {})
        self.assertFalse((self.root/'medlens-source.json').exists())


class Prepare4BTests(unittest.TestCase):
    def setUp(self):
        import copy
        import medlens.inference.profiles as profiles
        self.profiles=profiles
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        script=Path(__file__).resolve().parents[1]/'scripts/prepare_mlx_4b.py'
        spec=importlib.util.spec_from_file_location('prepare_4b_test',script)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.token_payload={'tokenizer.json':b'{"fictional":true}','tokenizer_config.json':b'{}','chat_template.jinja':b'template'}
        self.token_files={n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}for n,v in self.token_payload.items()}
        pins_patch=patch.object(profiles,'TOKENIZER_FILES',self.token_files);pins_patch.start();self.addCleanup(pins_patch.stop)
        self.source=self.root/'raw';self.source.mkdir()
        self.config={'model_type':'qwen3_5','quantization':{'bits':4,'group_size':64,'mode':'affine'},
            'text_config':{'hidden_size':2560,'num_hidden_layers':32,'vocab_size':248320,'eos_token_id':248044,
                           'dtype':'bfloat16','tie_word_embeddings':True}}
        (self.source/'config.json').write_bytes(canonical(self.config));(self.source/'model.safetensors').write_bytes(b'FICTIONAL WEIGHTS')
        self.token_source=self.root/'official-tokenizer';self.token_source.mkdir()
        for name,body in self.token_payload.items():(self.token_source/name).write_bytes(body)
        self.spec={**{k:profiles.get_profile()[k]for k in('model_id','revision','directory','dtype','quantization','tie_word_embeddings')},
            'source_directory':'raw','upstream_revision':None,'official_source_revision_verified':False,
            'files':{n:{'bytes':(self.source/n).stat().st_size,'sha256':sha_file(self.source/n)}for n in('config.json','model.safetensors')},
            'tokenizer':{**profiles.get_profile()['tokenizer'],'files':self.token_files}}
        (self.root/'configs').mkdir();self.save_pins()

    def save_pins(self):
        (self.root/'configs/mlx-assets.json').write_bytes(canonical({'schema_version':1,'models':{'4b':self.spec}}))

    def test_standalone_preparation_copies_verified_raw_bytes_without9b_or_native(self):
        with patch.object(self.module,'validate_model',return_value={'fictional':True}):
            value=self.module.prepare(self.root,tokenizer_source=self.token_source)
        output=self.root/self.spec['directory']
        self.assertEqual(value['model']['model_id'],'mlx-community/Qwen3.5-4B-MLX-4bit')
        self.assertEqual(value['dtype'],'bfloat16');self.assertIsNone(value['source']['upstream_revision'])
        self.assertFalse(value['conversion']['performed'])
        self.assertEqual((output/'model.safetensors').read_bytes(),b'FICTIONAL WEIGHTS')
        self.assertTrue((self.root/'.cache/tokenizers/qwen3.5-4b/tokenizer-source.json').is_file())
        self.assertFalse((self.root/'.cache/huggingface').exists());self.assertFalse((self.root/'.cache/venvs').exists())
        self.assertFalse(any(p.is_symlink()for p in output.iterdir()))
        identity=value.pop('artifact_id');self.assertEqual(hashlib.sha256(canonical(value)).hexdigest(),identity)

    def test_bad_model_or_tokenizer_pins_and_source_escape_rejected(self):
        import copy
        original=copy.deepcopy(self.spec)
        for change in (lambda s:s.update(revision='other'),lambda s:s['tokenizer'].update(revision='other'),
                       lambda s:s.update(source_directory='../outside'),lambda s:s['files']['model.safetensors'].update(sha256='bad')):
            self.spec=copy.deepcopy(original);change(self.spec);self.save_pins()
            with self.assertRaises(ValueError):self.module.load_pins(self.root)

    def test_existing_corrupt_or_linked_files_never_get_overwritten_or_downloaded(self):
        file=self.source/'model.safetensors';old=file.read_bytes();file.write_bytes(b'CORRUPTED WEIGHTS')
        with patch.object(self.module,'urlopen') as network:
            with self.assertRaisesRegex(ValueError,'checksum'):
                self.module.ensure_source(self.source,self.spec['files'],self.spec['model_id'],self.spec['revision'],download=True)
            network.assert_not_called()
        self.assertEqual(file.read_bytes(),b'CORRUPTED WEIGHTS')
        file.unlink();target=self.root/'real';target.write_bytes(old);file.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'regular'):
            self.module.ensure_source(self.source,self.spec['files'],self.spec['model_id'],self.spec['revision'])

    def test_failed_preparation_never_publishes_complete_model(self):
        with patch.object(self.module,'validate_model',side_effect=ValueError('bad tensor header')):
            with self.assertRaisesRegex(ValueError,'bad tensor'):
                self.module.prepare(self.root,tokenizer_source=self.token_source)
        output=self.root/self.spec['directory'];self.assertFalse(output.exists())
        states=list(output.parent.glob('.'+output.name+'.preparing-*/preparation-status.json'))
        self.assertEqual(len(states),1);self.assertEqual(json.loads(states[0].read_bytes())['status'],'failed')

    def test_download_requires_explicit_flag_checks_size_hash_and_pinned_url(self):
        import io
        target=self.root/'download';body=b'PUBLIC FICTIONAL ASSET'
        files={'config.json':{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}}
        with patch.object(self.module,'urlopen',return_value=io.BytesIO(body)) as network:
            with self.assertRaisesRegex(ValueError,'--download'):
                self.module.ensure_source(target,files,'repo/name','fixed-revision')
            network.assert_not_called()
            self.module.ensure_source(target,files,'repo/name','fixed-revision',download=True)
            self.assertEqual(network.call_args.args[0],'https://huggingface.co/repo/name/resolve/fixed-revision/config.json')
        self.assertEqual((target/'config.json').read_bytes(),body)
        failed=self.root/'failed-download'
        with patch.object(self.module,'urlopen',return_value=io.BytesIO(body+b'OVERFLOW')):
            with self.assertRaisesRegex(ValueError,'pinned size'):
                self.module.ensure_source(failed,files,'repo/name','fixed',download=True)
        self.assertFalse((failed/'config.json').exists());self.assertEqual(list(failed.iterdir()),[])

    def model_header(self, mutate=None):
        header={};offset=0
        def tensor(name,dtype,shape,width):
            nonlocal offset
            size=__import__('math').prod(shape)*width
            header[name]={'dtype':dtype,'shape':shape,'data_offsets':[offset,offset+size]};offset+=size
        tensor('language_model.model.embed_tokens.weight','U32',[248320,320],4)
        for i in range(32):
            if i%4!=3:
                tensor(f'language_model.model.layers.{i}.linear_attn.A_log','F32',[32],4)
                tensor(f'language_model.model.layers.{i}.linear_attn.dt_bias','BF16',[32],2)
        if mutate:mutate(header)
        encoded=canonical(header)
        with(self.source/'model.safetensors').open('wb')as stream:
            stream.write(struct.pack('<Q',len(encoded))+encoded);stream.truncate(8+len(encoded)+offset)

    def test_header_checks_bfloat16_recurrent_precision_and_tied_quantized_head(self):
        self.model_header();self.assertTrue(self.module.validate_model(self.source)['tie_word_embeddings'])
        self.model_header(lambda h:h['language_model.model.layers.0.linear_attn.A_log'].update(dtype='BF16'))
        with self.assertRaises(ValueError):self.module.validate_model(self.source)
        self.model_header(lambda h:h['language_model.model.embed_tokens.weight'].update(dtype='BF16'))
        with self.assertRaises(ValueError):self.module.validate_model(self.source)
        self.config['text_config']['tie_word_embeddings']=False
        (self.source/'config.json').write_bytes(canonical(self.config))
        self.model_header()
        with self.assertRaisesRegex(ValueError,'config'):self.module.validate_model(self.source)


if __name__ == '__main__':
    unittest.main()

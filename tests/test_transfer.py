"""Transfer integrity, cache symlinks, fresh restore and destination protection."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from medlens.transfer import inventory, pack, restore, verify


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.bundle, self.dest = base/'source', base/'bundle', base/'dest'
        self.root.mkdir(); self.dest.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test', '-c',
            'user.email=test@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
        self.put('data/parsed/pilot/a.json', b'example' * 1000)
        self.put('.cache/huggingface/hub/blobs/hash', b'weights' * 1000)
        link = self.root/'.cache/huggingface/hub/models/snapshots/revision/model.bin'
        link.parent.mkdir(parents=True)
        link.symlink_to('../../../blobs/hash')
        self.put('.env', b'never bundle')
        self.put('.cache/huggingface/token', b'never bundle')
        self.put('.venv/bin/python', b'never bundle')
        self.put('.cache/huggingface/hub/.locks/a.lock', b'never bundle')

    def put(self, name, data):
        p = self.root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)

    def save_manifest(self, value):
        (self.bundle/'manifest.json').write_text(json.dumps(value))

    def test_split_restore_preserves_link_and_omits_machine_local_files(self):
        made = pack(self.root, self.bundle, part_bytes=2048)
        self.assertGreater(len(made['parts']), 1)
        self.assertTrue(all(p['bytes'] <= 2048 for p in made['parts']))
        self.assertEqual(verify(self.bundle), made)
        restored = restore(self.bundle, self.dest)
        self.assertEqual(restored['file_count'], 3)
        self.assertEqual((self.dest/'data/parsed/pilot/a.json').read_bytes(), b'example' * 1000)
        link = self.dest/'.cache/huggingface/hub/models/snapshots/revision/model.bin'
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.read_bytes(), b'weights' * 1000)
        self.assertFalse((self.dest/'.env').exists())
        self.assertFalse((self.dest/'.venv').exists())
        self.assertEqual(restore(self.bundle, self.dest)['existing_identical_files'], 3)

    def test_corruption_fails_before_destination_writes(self):
        manifest = pack(self.root, self.bundle, part_bytes=2048)
        part = self.bundle/manifest['parts'][0]['name']
        data = bytearray(part.read_bytes()); data[0] ^= 1; part.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            restore(self.bundle, self.dest)
        self.assertEqual(list(self.dest.iterdir()), [])

    def test_conflicting_file_is_not_overwritten_or_followed_by_partial_restore(self):
        pack(self.root, self.bundle)
        p = self.dest/'data/parsed/pilot/a.json'; p.parent.mkdir(parents=True)
        p.write_text('different')
        with self.assertRaisesRegex(ValueError, 'conflict'):
            restore(self.bundle, self.dest)
        self.assertEqual(p.read_text(), 'different')
        self.assertFalse((self.dest/'.cache').exists())

    def test_destination_directory_symlink_rejected(self):
        pack(self.root, self.bundle)
        outside = self.dest.parent/'outside'; outside.mkdir()
        (self.dest/'data').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink conflict'):
            restore(self.bundle, self.dest)
        self.assertEqual(list(outside.iterdir()), [])

    def test_external_dangling_and_cyclic_source_links_rejected(self):
        path = self.root/'data/link'
        for target in ('/etc/passwd', '../../outside', 'missing', 'link'):
            path.symlink_to(target)
            with self.assertRaises(ValueError):
                inventory(self.root)
            path.unlink()

    def test_manifest_path_escape_part_escape_and_file_ancestor_rejected(self):
        made = pack(self.root, self.bundle)
        for name in ('../escape', '/absolute', '.env'):
            changed = json.loads(json.dumps(made)); changed['entries'][0]['path'] = name
            self.save_manifest(changed)
            with self.assertRaises(ValueError):
                verify(self.bundle)
        changed = json.loads(json.dumps(made)); changed['parts'][0]['name'] = '../part'
        self.save_manifest(changed)
        with self.assertRaises(ValueError):
            verify(self.bundle)
        changed = json.loads(json.dumps(made))
        changed['entries'].append({'path': changed['entries'][0]['path']+'/child',
            'kind': 'file', 'bytes': 0, 'sha256': '0'*64})
        self.save_manifest(changed)
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            verify(self.bundle)

    def test_fresh_path_and_nonrecursive_output_required(self):
        pack(self.root, self.bundle)
        with self.assertRaises(FileExistsError):
            pack(self.root, self.bundle)
        with self.assertRaisesRegex(ValueError, 'own artifact'):
            pack(self.root, self.root/'data/new-bundle')

    def test_archived_file_hash_is_checked_even_if_part_hash_matches(self):
        made = pack(self.root, self.bundle)
        for entry in made['entries']:
            if entry['kind'] == 'file': entry['sha256'] = '0'*64
        self.save_manifest(made)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            restore(self.bundle, self.dest)


if __name__ == '__main__':
    unittest.main()

"""Pack, verify or restore local MedLens artifacts with Python's standard library."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medlens.transfer import pack, restore, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pack', 'verify', 'restore'))
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--part-mib', type=int, default=2048)
    args = parser.parse_args()
    if args.action == 'pack':
        report = pack(args.root, args.bundle, part_bytes=args.part_mib * 1024**2)
        report = {k: report[k] for k in ('status', 'source_commit', 'file_count', 'artifact_bytes', 'archive_bytes', 'parts')}
    elif args.action == 'verify':
        report = verify(args.bundle)
        report = {k: report[k] for k in ('status', 'source_commit', 'file_count', 'artifact_bytes')}
    else:
        report = restore(args.bundle, args.root)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

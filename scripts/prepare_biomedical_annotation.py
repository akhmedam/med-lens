"""Build, verify and freeze manual biomedical relevance packets; never score quality."""

import argparse
import json
from pathlib import Path

from medlens.evaluation.biomedical_annotation import (build_packet, freeze_export, load_packet,
    validate_export)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Build a blank packet from the exact frozen case plan and search-only sources')
    build.add_argument('--plan', type=Path, required=True)
    build.add_argument('--runs-root', type=Path, required=True, help='Contains <query_id>/run.json and its snapshot/artifacts')
    build.add_argument('--output', type=Path, required=True, help='Fresh annotation packet directory')
    build.add_argument('--depth', type=int, help='Must exactly match annotation_depth_per_method in the frozen plan')
    verify = commands.add_parser('verify', help='Verify frozen source/item/template/script bindings; no labels or scores')
    verify.add_argument('--package', type=Path, required=True, help='Packet directory or package.json')
    validate = commands.add_parser('validate', help='Validate an existing human export without modifying it')
    validate.add_argument('--package', type=Path, required=True)
    validate.add_argument('--human-export', type=Path, required=True)
    validate.add_argument('--require-finalized', action='store_true')
    freeze = commands.add_parser('freeze', help='Preserve a finalized MANUAL HUMAN original byte-for-byte with a SHA256 receipt')
    freeze.add_argument('--package', type=Path, required=True)
    freeze.add_argument('--human-export', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True, help='Fresh reference directory separate from the packet')
    args = parser.parse_args(argv)
    if args.command == 'build':
        value = build_packet(args.plan, args.runs_root, args.output, depth=args.depth)
        result = {'status':'complete', 'package_id':value['package_id'], 'questions':value['question_count'],
                  'pairs':value['pair_count'], 'human_labels_generated':False, 'quality_scores_computed':False,
                  'index':str((args.output/'index.html').resolve())}
    elif args.command == 'verify':
        value = load_packet(args.package)
        result = {'status':'passed', 'package_id':value.package['package_id'],
                  'questions':len(value.package['queries']), 'pairs':len(value.package['items']),
                  'quality_scores_computed':False}
    elif args.command == 'validate':
        result = validate_export(args.package, args.human_export, require_finalized=args.require_finalized)
    else: result = freeze_export(args.package, args.human_export, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())

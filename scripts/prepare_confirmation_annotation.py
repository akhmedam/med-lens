"""Prepare and preserve a blank manual-human confirmation reference; never score it."""

import argparse
import json
from pathlib import Path

from medlens.evaluation.confirmation_annotation import (
    build_packet, freeze_export, load_packet, validate_export,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    build=commands.add_parser('build',help='Create a blank, source-bound article/answerability form')
    build.add_argument('--plan',type=Path,required=True)
    build.add_argument('--runs-root',type=Path,required=True)
    build.add_argument('--output',type=Path,required=True)
    verify=commands.add_parser('verify',help='Verify sources, questions and the embedded form without reading labels')
    verify.add_argument('--package',type=Path,required=True)
    validate=commands.add_parser('validate',help='Check a user export structurally; do not display grades or rationales')
    validate.add_argument('--package',type=Path,required=True)
    validate.add_argument('--human-export',type=Path,required=True)
    validate.add_argument('--require-finalized',action='store_true')
    freeze=commands.add_parser('freeze',help='Preserve the original MANUAL HUMAN JSON read-only, without scoring')
    freeze.add_argument('--package',type=Path,required=True)
    freeze.add_argument('--human-export',type=Path,required=True)
    freeze.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='build':
        value=build_packet(args.plan,args.runs_root,args.output)
        result={'status':'complete','package_id':value['package_id'],'questions':len(value['queries']),
            'article_cards':len(value['items']),'human_labels_generated':False,
            'answer_generation_performed':False,'quality_scores_computed':False,
            'index':str((args.output/'index.html').resolve())}
    elif args.command=='verify':
        value=load_packet(args.package)
        result={'status':'passed','package_id':value.package['package_id'],'questions':len(value.package['queries']),
            'article_cards':len(value.package['items']),'quality_scores_computed':False}
    elif args.command=='validate':
        result=validate_export(args.package,args.human_export,require_finalized=args.require_finalized)
    else:
        result=freeze_export(args.package,args.human_export,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())

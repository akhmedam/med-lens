"""Build or inspect a source-verified shared development corpus without changing snapshots."""

import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'

from medlens.corpus import build_collection,inspect_collection
from medlens.resources import load_collection_profile
from medlens.retrieval.bge_tokenizer import MODEL_ID,REVISION,load_tokenizer
from medlens.retrieval.embeddings import BGEEncoder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    inspect=commands.add_parser('inspect');inspect.add_argument('collection',type=Path)
    build=commands.add_parser('build');build.add_argument('--collections',type=Path,required=True)
    build.add_argument('--output',type=Path,required=True)
    build.add_argument('--cache',type=Path,default=ROOT/'.cache/embeddings/bge-input-v1')
    build.add_argument('--source',action='append',help='Include only these source labels; repeatable')
    build.add_argument('--no-seed',action='store_true',help='Measure fresh encoding instead of importing existing native vectors')
    build.add_argument('--reuse',type=Path,help='Import verified vectors from an earlier shared revision')
    build.add_argument('--resolutions',type=Path,help='Explicit PMID -> latest dataset_id tie resolutions')
    build.add_argument('--model',type=Path,default=ROOT/'.cache/models/bge-base-en-v1.5')
    build.add_argument('--tokenizer',type=Path,default=ROOT/'.cache/tokenizers/bge-base-en-v1.5')
    args=parser.parse_args(argv)
    if args.action=='inspect':report=inspect_collection(args.collection)
    else:
        _,_,sources=load_collection_profile(ROOT,args.collections)
        if args.source:
            labels={s['query_id'] for s in sources}
            if len(set(args.source))!=len(args.source) or not set(args.source)<=labels:parser.error('Unknown or duplicate source labels')
            sources=[s for s in sources if s['query_id'] in args.source]
        tokenizer,info=load_tokenizer(args.tokenizer)
        # The first verified native source declares the unchanged pinned BGE policy.
        encoder_info=json.loads((sources[0]['embeddings']/'manifest.json').read_bytes())['encoder']
        if (encoder_info['model']['model_id']!=MODEL_ID or encoder_info['model']['revision']!=REVISION
                or encoder_info['device']!='mps' or encoder_info['dtype']!='float32'
                or encoder_info['dimension']!=768 or encoder_info['normalization']!='L2'
                or encoder_info['pooling']!='last_hidden_state[:,0,:]' or encoder_info['attention']!='eager'):
            parser.error('Source must declare the pinned BGE MPS policy')
        def factory():return BGEEncoder(args.model,args.tokenizer,device='mps',batch_size=encoder_info['batch_size'])
        resolutions=json.loads(args.resolutions.read_bytes()) if args.resolutions else None
        report=build_collection(sources,tokenizer,info,args.cache,args.output,encoder_factory=factory,
            encoder_info=encoder_info,resolutions=resolutions,seed=not args.no_seed,reuse=args.reuse)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())

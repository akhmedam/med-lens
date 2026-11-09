"""Fetch a bounded permitted PMC subset and prepare paired local representations."""

import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
from medlens.pmc import download_pilot,load_snapshot,build_collection
from medlens.retrieval.bge_tokenizer import load_tokenizer
from medlens.retrieval.embeddings import BGEEncoder
from medlens.retrieval.dense import DenseIndex
from medlens.resources import load_collection_profile


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fetch=commands.add_parser('fetch');fetch.add_argument('--base',type=Path,required=True);fetch.add_argument('--output',type=Path,required=True)
    fetch.add_argument('--limit',type=int,default=6);fetch.add_argument('--max-candidates',type=int,default=36)
    build=commands.add_parser('build');build.add_argument('--snapshot',type=Path,required=True);build.add_argument('--output',type=Path,required=True)
    build.add_argument('--mode',choices=('abstract','fulltext'),required=True)
    build.add_argument('--cache',type=Path,default=ROOT/'.cache/embeddings/pmc-pilot-v1')
    inspect=commands.add_parser('inspect');inspect.add_argument('collection',type=Path)
    args=parser.parse_args(argv)
    if args.command=='fetch':result=download_pilot(args.base,args.output,limit=args.limit,max_candidates=args.max_candidates)
    elif args.command=='build':
        tokenizer,info=load_tokenizer(ROOT/'.cache/tokenizers/bge-base-en-v1.5')
        _,_,sources=load_collection_profile(ROOT,ROOT/'configs/collections/recovery-20261106.json')
        encoder_info=json.loads((sources[0]['embeddings']/'manifest.json').read_bytes())['encoder']
        def factory():return BGEEncoder(ROOT/'.cache/models/bge-base-en-v1.5',
            ROOT/'.cache/tokenizers/bge-base-en-v1.5',device='mps',batch_size=encoder_info['batch_size'])
        result=build_collection(args.snapshot,tokenizer,info,args.cache,args.output,encoder_info,factory,mode=args.mode)
    else:
        idx=DenseIndex(args.collection/'dataset',args.collection/'windows',args.collection/'embeddings')
        from medlens.generation.context import source_section
        for w in idx.windows:source_section(idx.articles[w['pmid']],w)
        result={'dataset_id':idx.dataset.manifest['dataset_id'],'articles':len(idx.articles),'windows':len(idx.windows),'status':'verified'}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

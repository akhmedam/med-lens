"""Pinned MedCPT pair scoring with explicit input budgets and no truncation."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

MODEL_ID = 'ncbi/MedCPT-Cross-Encoder'
REVISION = '71caf65d4927987813984f54c284405a13fcca49'
MAX_TOKENS = 512
CLS_ID, SEP_ID, VOCAB_SIZE = 2, 3, 30522
ASSETS = ('config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json',
          'added_tokens.json', 'special_tokens_map.json', 'vocab.txt', 'LICENSE', 'README.md')


class PairBudgetError(ValueError):
    """The complete query/document pair exceeds the model's input budget."""


def sha_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def verify_assets(directory):
    directory = Path(directory)
    source = json.loads((directory/'model-source.json').read_bytes())
    if (source.get('schema_version') != 1 or source.get('model_id') != MODEL_ID
            or source.get('revision') != REVISION or set(source.get('files', {})) != set(ASSETS)):
        raise ValueError('Unsupported MedCPT asset manifest')
    for name, expected in source['files'].items():
        path = directory/name
        if path.is_symlink() or sha_file(path) != expected:
            raise ValueError(f'MedCPT asset checksum mismatch: {name}')
    return source


def download_assets(directory):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import get_session
    directory = Path(directory)
    if (directory/'model-source.json').exists():
        return verify_assets(directory)
    if directory.is_symlink() or any((directory/name).is_symlink() for name in ASSETS):
        raise ValueError('MedCPT download assets must not be symlinks')
    # The same trailing-header compatibility issue as the existing Qwen runtime.
    def trim_user_agent(request):
        value = request.headers.get('user-agent')
        if value is not None:
            request.headers['user-agent'] = value.rstrip()
    get_session().event_hooks['request'].append(trim_user_agent)
    try:
        snapshot_download(MODEL_ID, revision=REVISION, token=False,
            allow_patterns=list(ASSETS), local_dir=directory)
    finally:
        get_session().event_hooks['request'].remove(trim_user_agent)
    source = {'schema_version':1, 'model_id':MODEL_ID, 'revision':REVISION,
        'files':{name:sha_file(directory/name) for name in ASSETS}}
    with (directory/'model-source.json').open('x') as stream:
        stream.write(json.dumps(source,indent=2)+'\n')
    return verify_assets(directory)


def load_tokenizer(directory, *, verified=False):
    from transformers import AutoTokenizer
    if not verified:
        verify_assets(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
        token=False, trust_remote_code=False, use_fast=True)
    if (not tokenizer.is_fast or tokenizer.num_special_tokens_to_add(pair=True) != 3
            or tokenizer.cls_token_id != CLS_ID or tokenizer.sep_token_id != SEP_ID
            or tokenizer.pad_token_id != 0 or len(tokenizer) != VOCAB_SIZE):
        raise ValueError('Unexpected MedCPT BERT tokenizer')
    info = {'model_id':MODEL_ID, 'revision':REVISION,
        'backend_sha256':hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        'pair_format':'[CLS] question [SEP] title-newline-excerpt [SEP]',
        'max_input_tokens':MAX_TOKENS, 'cls_token_id':CLS_ID, 'sep_token_id':SEP_ID, 'truncation':False}
    return tokenizer, info


def prepare_pair(question, title, text, tokenizer):
    if not all(isinstance(value,str) for value in (question,title,text)):
        raise ValueError('Question, title and excerpt must be strings')
    if not question.strip() or not (title.strip() or text.strip()):
        raise ValueError('A nonempty question and document are required')
    document = title + '\n' + text
    encoded = tokenizer(question, document, add_special_tokens=True, truncation=False,
        padding=False, return_token_type_ids=True, verbose=False)
    row = {key:encoded[key] for key in ('input_ids','attention_mask','token_type_ids')}
    validate_encoding(row)
    return row


def validate_encoding(row):
    count = len(row['input_ids'])
    if count > MAX_TOKENS:
        raise PairBudgetError(f'MedCPT pair has {count} tokens; maximum is {MAX_TOKENS}; no text was truncated')
    if (not 3 <= count <= MAX_TOKENS or row['input_ids'][0] != CLS_ID or row['input_ids'][-1] != SEP_ID
            or any(type(i) is not int or not 0 <= i < VOCAB_SIZE for i in row['input_ids'])
            or row['attention_mask'] != [1]*count
            or any(type(i) is not int for i in row['attention_mask'])
            or len(row['token_type_ids']) != count
            or any(type(i) is not int or i not in {0,1} for i in row['token_type_ids'])):
        raise ValueError('Invalid MedCPT pair encoding')


def score_encoded(rows, model, *, device, batch_size):
    """Keep pair order, masks and token types; return untransformed classifier logits."""
    import torch
    if device not in {'cpu','mps'} or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Choose cpu/mps and a positive integer batch size')
    for row in rows:
        validate_encoding(row)
    scores, max_tensor, max_driver = [], 0, 0
    if device == 'mps':torch.mps.synchronize()
    started = perf_counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch = rows[start:start+batch_size]
            length = max(len(row['input_ids']) for row in batch)
            inputs = {key:torch.zeros((len(batch),length),dtype=torch.long,device=device)
                      for key in ('input_ids','attention_mask','token_type_ids')}
            for i,row in enumerate(batch):
                size = len(row['input_ids'])
                for key in inputs:
                    inputs[key][i,:size] = torch.tensor(row[key],dtype=torch.long,device=device)
            logits = model(**inputs).logits
            if logits.shape != (len(batch),1) or not torch.isfinite(logits).all().item():
                raise ValueError('Expected one finite MedCPT logit per pair')
            scores.extend(logits[:,0].float().cpu().tolist())
            if device == 'mps':
                max_tensor = max(max_tensor,torch.mps.current_allocated_memory())
                max_driver = max(max_driver,torch.mps.driver_allocated_memory())
    if device == 'mps':torch.mps.synchronize()
    return {'scores':scores, 'forward_seconds':perf_counter()-started,
        'batch_size':batch_size, 'batch_count':(len(rows)+batch_size-1)//batch_size,
        'sampled_max_mps_tensor_bytes':max_tensor if device == 'mps' else None,
        'sampled_max_mps_driver_bytes':max_driver if device == 'mps' else None}


class MedCPTScorer:
    def __init__(self, directory, *, device='mps', batch_size=4):
        import os
        import torch
        from transformers import AutoModelForSequenceClassification
        if device not in {'cpu','mps'} or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Choose cpu/mps and a positive integer batch size')
        if device == 'mps' and (not torch.backends.mps.is_available()
                or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'):
            raise RuntimeError('MPS is required with implicit CPU fallback disabled')
        started = perf_counter()
        source = verify_assets(directory)
        self.tokenizer, token_info = load_tokenizer(directory,verified=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory,
            local_files_only=True, token=False, trust_remote_code=False, use_safetensors=False,
            weights_only=True, dtype=torch.float32, attn_implementation='eager').to(device).eval()
        cfg = self.model.config
        if (cfg.model_type != 'bert' or cfg.hidden_size != 768 or cfg.num_labels != 1
                or cfg.max_position_embeddings != MAX_TOKENS or cfg.vocab_size != 30522):
            raise ValueError('Unexpected MedCPT model architecture')
        if {p.device.type for p in self.model.parameters()} != {device}:
            raise RuntimeError('MedCPT parameters are on an unexpected device')
        if device == 'mps':torch.mps.synchronize()
        self.device, self.batch_size = device, batch_size
        self.metadata = {'model_id':MODEL_ID,'revision':REVISION,'source':source,
            'tokenizer':token_info,'device':device,'dtype':'float32','attention':'eager',
            'default_batch_size':batch_size,
            'score':'raw-single-logit-higher-is-more-relevant','parameter_count':sum(p.numel() for p in self.model.parameters()),
            'load_seconds':perf_counter()-started,'torch_threads':torch.get_num_threads(),
            'packages':{name:version(name) for name in ('torch','transformers','tokenizers','huggingface-hub')}}

    def score(self, pairs, *, batch_size=None):
        size = self.batch_size if batch_size is None else batch_size
        if type(size) is not int or size <= 0:
            raise ValueError('Batch size must be a positive integer')
        started = perf_counter()
        # Validate the complete request before the first forward call.
        rows = [prepare_pair(*pair,self.tokenizer) for pair in pairs]
        tokenize_seconds = perf_counter()-started
        result = score_encoded(rows,self.model,device=self.device,batch_size=size)
        return {**result,'pair_count':len(rows),'pair_lengths':[len(r['input_ids']) for r in rows],
            'pairs_sha256':hashlib.sha256(canonical(rows).encode()).hexdigest(),
            'tokenize_seconds':tokenize_seconds,'score_seconds':perf_counter()-started}

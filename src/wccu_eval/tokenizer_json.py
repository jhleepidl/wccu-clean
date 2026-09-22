"""Offline, deterministic byte-BPE encoder for the supplied Qwen tokenizer JSON.

No tokenizer vocabulary is redistributed. Verified separately against archived
complete-record token counts. Only supports the exact NFC / regex / byte-BPE
configuration asserted here; it is not a generic Transformers replacement.
"""
from __future__ import annotations
import json
import unicodedata
from functools import lru_cache
from pathlib import Path
import regex

class LocalQwenBPE:
    def __init__(self, path: str | Path):
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        assert obj['normalizer'] == {'type':'NFC'}, 'Unexpected tokenizer normalizer'
        pts = obj['pre_tokenizer']['pretokenizers']
        assert len(pts)==2 and pts[0]['type']=='Split' and pts[0]['behavior']=='Isolated'
        assert pts[1]=={'type':'ByteLevel','add_prefix_space':False,'trim_offsets':False,'use_regex':False}
        model=obj['model']
        assert model['type']=='BPE' and not model['dropout'] and not model['ignore_merges']
        self.pattern=regex.compile(pts[0]['pattern']['Regex'])
        self.vocab=model['vocab']
        self.ranks={tuple(p if isinstance(p,list) else p.split(' ')):i for i,p in enumerate(model['merges'])}
        bs=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord('®'),ord('ÿ')+1))
        cs=bs[:]; n=0
        for b in range(256):
            if b not in bs: bs.append(b); cs.append(256+n); n+=1
        self.byte_map={b:chr(c) for b,c in zip(bs,cs)}
        self.special={a['content']:a['id'] for a in obj.get('added_tokens',[])}
        self.special_pattern=regex.compile('('+'|'.join(regex.escape(x) for x in sorted(self.special,key=len,reverse=True))+')') if self.special else None

    @lru_cache(maxsize=200000)
    def _piece(self, piece: str) -> tuple[int,...]:
        seq=[self.byte_map[b] for b in piece.encode('utf-8')]
        while len(seq)>1:
            ranked=[(self.ranks.get((seq[i],seq[i+1]),float('inf')),i) for i in range(len(seq)-1)]
            rank, i=min(ranked)
            if rank==float('inf'):break
            seq[i:i+2]=[seq[i]+seq[i+1]]
        return tuple(self.vocab[x] for x in seq)

    def encode(self, text: str) -> list[int]:
        result=[]
        parts=self.special_pattern.split(text) if self.special_pattern else [text]
        for part in parts:
            if not part:continue
            if part in self.special:result.append(self.special[part]);continue
            part=unicodedata.normalize('NFC',part)
            pieces=self.pattern.findall(part)
            if ''.join(pieces)!=part:raise ValueError('Tokenizer regex left unmatched text')
            for p in pieces:result.extend(self._piece(p))
        return result

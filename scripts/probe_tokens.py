import sys
sys.path.insert(0, '.')
from bigram import BigramTokenizer

tok = BigramTokenizer.load('nano1/tokenizer.json')

# Get the underlying tokenizer
ht = tok._tok

# Test schema -- ultra-compact
tests = [
    '{"name":"search","description":"tk","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}',
    '{"name":"x","description":"y","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}',
    '<tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>',
]
for s in tests:
    base, _ = __import__('bigram.tokenizer.tonal', fromlist=['split_tone']).split_tone(s)
    enc = ht.encode(base)
    pieces = [ht.id_to_token(i) for i in enc.ids]
    print(f"\n{s}")
    print(f"  base: {base}")
    print(f"  pieces ({len(pieces)}): {pieces}")

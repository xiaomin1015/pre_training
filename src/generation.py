import json


import regex
import torch


from tokenizer.pretokenization import DEFAULT_PRETOKEN_PATTERN
from tokenizer.tokenizer import encode_document, decode_valid_document, bpe_encode_pretoken, load_merges, load_vocab
from training.load_checkingPoint import Load_verify_checkPoint
from src.model import decode
from training.train import get_device


checkpoint_path = "checkpoints/checkpoint_6000.pt"
print(checkpoint_path)
vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
num_layers = 4
num_heads = 16
rope_theta = 10000
batch_size = 32




# paras: 23m, 0.085GB
learning_rate=1e-3
betas=(0.9, 0.999)
weight_decay =0.01
eps=1e-8




device = get_device()
print(device)
model, opt = Load_verify_checkPoint(checkpoint_path)


vocab_path="vocab.json"
merges_path="merges.txt"
vocab = load_vocab(vocab_path)
def encode_document(document: str,):
    """
    Apply exactly the same regex pre-tokenization used during BPE training
    Then run BPE independently on each pre-token.
    """
    output_ids = []
    pattern = DEFAULT_PRETOKEN_PATTERN
    pre_tokens = regex.compile( pattern).findall( document )
   
    merges = load_merges(merges_path)
    MERGE_RANK = {
        pair: rank
        for rank, pair in enumerate(merges)
    }
    for pre_token in pre_tokens:
        pretoken_bytes = pre_token.encode("utf-8")
        curr = bpe_encode_pretoken( pretoken_bytes, vocab,  MERGE_RANK)
        output_ids.extend( curr)
    return output_ids


prompt = "Once upon a time"
prompt_ids = encode_document(prompt)
print(prompt_ids)
prompt_ids = torch.tensor([prompt_ids], device=device)
eos_token_id = encode_document("<|endoftext|>")[0]
print(eos_token_id)
# [batch_size, total_sequence_length]
output_ids = decode( model, prompt_ids, max_new_tokens=200,  temperature=0.8, top_p=0.9, eos_token_id=eos_token_id )
output = output_ids[0].tolist()
with open(vocab_path,"r",encoding="utf-8",) as f:
    data = json.load(f)
id_to_token = {
    int(token_id): bytes.fromhex(token_hex)
    for token_id, token_hex in data.items()
}
token_bytes = b"".join(
    id_to_token[token_id]
    for token_id in output
)


text = token_bytes.decode(
    "utf-8",
    errors="replace",
)


print(text)

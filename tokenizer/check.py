import os
import json
import numpy as np


BIN_PATH = "TinyStories-train.bin"
VOCAB_PATH = "vocab.json"
SPECIAL_TOKEN = b"<|endoftext|>"


with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    raw_vocab = json.load(f)

id_to_bytes = {
    int(token_id): bytes.fromhex(token_hex)
    for token_id, token_hex in raw_vocab.items()
}

bytes_to_id = {
    token_bytes: token_id
    for token_id, token_bytes in id_to_bytes.items()
}

special_id = bytes_to_id[SPECIAL_TOKEN]


# Memory-map; does NOT load entire 1 GB file into RAM
tokens = np.memmap( BIN_PATH,dtype=np.uint16, mode="r")

print("=" * 60)
print("FAST TOKEN FILE CHECK")
print("=" * 60)
print(f"File size:       {os.path.getsize(BIN_PATH):,} bytes")
print(f"Token count:     {len(tokens):,}")
print(f"Vocabulary size: {len(id_to_bytes):,}")


# Token range
min_id = int(tokens.min())
max_id = int(tokens.max())

print(f"Min token ID:    {min_id}")
print(f"Max token ID:    {max_id}")

if min_id < 0 or max_id >= len(id_to_bytes):
    raise ValueError(f"Invalid token IDs: {min_id}..{max_id}")
print("Token IDs:       PASS")


# Special-token count
# NumPy performs this in optimized compiled code.
special_count = int(
    np.count_nonzero(tokens == special_id)
)
print(f"Special tokens:  {special_count:,}")

special_tokens_positions = [i for i, token in tokens if token == special_id]
print(f"first 10 Special tokens position:  {special_tokens_positions[:10]:,}")
# Bytes-per-token without actually decoding entire corpus
length_lookup = np.zeros(
    max(id_to_bytes) + 1,
    dtype=np.uint16,
)

for token_id, token_bytes in id_to_bytes.items():
    length_lookup[token_id] = len(token_bytes)


# Process in large vectorized blocks.
BLOCK = 20_000_000
decoded_byte_count = 0
for start in range(0, len(tokens), BLOCK):
    chunk = tokens[start:start + BLOCK]

    decoded_byte_count += int(
        length_lookup[chunk].sum(
            dtype=np.uint64
        )
    )


print(
    f"Decoded bytes:   "
    f"{decoded_byte_count:,}"
)

print(
    f"Bytes/token:     "
    f"{decoded_byte_count / len(tokens):.3f}"
)


# ------------------------------------------------------------
# Decode some samples
# ------------------------------------------------------------

def decode_sample(start, count=100):

    ids = tokens[
        start:start + count
    ]

    decoded = b"".join(
        id_to_bytes[int(token_id)]
        for token_id in ids
    )

    return decoded.decode(
        "utf-8",
        errors="replace",
    )


print()
print("START SAMPLE:")
print(repr(
    decode_sample(0, 256)
))


middle = len(tokens) // 2

print()
print("MIDDLE SAMPLE:")
print(repr(
    decode_sample(middle, 256)
))


end = max(
    0,
    len(tokens) - 100
)

print()
print("END SAMPLE:")
print(repr(
    decode_sample(end, 256)
))

print()
print("around special token:")
print(repr(
    decode_sample(special_tokens_positions[5]-150, 256)
))



print()
print("=" * 60)
print("FAST CHECK COMPLETE")
print("=" * 60)
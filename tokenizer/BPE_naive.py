from collections import Counter
import time

def get_pair_counts(word_freqs):
    pair_counts = Counter()
    for tokens, freq in word_freqs.items():
        for i in range(len(tokens) - 1):
            pair = tokens[i], tokens[i + 1]
            pair_counts[pair] += freq
    return pair_counts 


def merge_pair(tokens, pair):
    merged = []
    i = 0
    while i < len(tokens):
        if (
            i < len(tokens) - 1
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    return tuple(merged)


def train_bpe_from_counts(counts: Counter, vocab_size: int, special_tokens: list[str]):
    print(f"here in train_bpe_from_counts, has vocab {vocab_size}, count {len(counts)}")
    print(list(counts.items())[:5])
    start = time.perf_counter()
    # Initial vocabulary: all 256 bytes. int -> bytes
    vocab = {
        i: bytes([i])
        for i in range(256)
    }
    # append special tokens 
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Convert pre-token strings into tuples of byte tokens.
    word_freqs = {}
    for token, freq in counts.items():
        byte_tokens = tuple(
            bytes([b])
            for b in token.encode("utf-8")
        )
        word_freqs[byte_tokens] = freq
    print(list(word_freqs.items())[:5])
   
    # merge
    merges = []
    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break
        # Frequency first, tie-breaking second.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair),)

        # Record merge.
        merges.append(best_pair)

        # append new token to vocabulary
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # Apply merge to every pre-token.
        new_word_freqs = {}

        for tokens, freq in word_freqs.items():
            new_tokens = merge_pair(tokens, best_pair)
            new_word_freqs[new_tokens] = freq

        word_freqs = new_word_freqs
    end = time.perf_counter()
    print(f"Pre-tokenization time: {end - start:.4f} seconds")
    return vocab, merges
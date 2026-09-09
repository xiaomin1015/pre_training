from collections import Counter, defaultdict
import time

from collections import Counter, defaultdict


"""Return the Counter of adjacent pairs in one token sequence."""
def get_pairs(tokens):
    pairs = Counter()
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs

"""
    Merge all non-overlapping occurrences of pair in tokens.
    Example:
        [a, b, b, c]
        pair = (b, b)

        -> [a, bb, c]
"""
def merge_pair_in_place(tokens, pair):
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            tokens[i:i + 2] = [tokens[i] + tokens[i + 1]]
        else:
            i += 1


def train_bpe_from_counts(counts, vocab_size, special_tokens,):
    # 1. Initial vocabulary
    vocab = {
        i: bytes([i])
        for i in range(256)
    }
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 2. Convert pre-tokens into mutable lists of bytes
    word_tokens = {} # id -> bytes
    word_freqs = {}  # id -> freq

    for word_id, (word, freq) in enumerate(counts.items()):
        # list[bytes], mutable
        word_tokens[word_id] = [
            bytes([b])
            for b in word.encode("utf-8") # bytes(Iterating) -> integer -> single bytes object
        ]
        word_freqs[word_id] = freq

    # pair -> freq
    pair_counts = Counter()

    # pair -> set(word IDs containing that pair)
    # automatically create an empty set the first time we encounter that pair.
    pair_to_words = defaultdict(set)
    for word_id, tokens in word_tokens.items():
        pairs = get_pairs(tokens)
        for pair, count in pairs.items():
            freq = word_freqs[word_id]
            pair_counts[pair] += count * freq
            pair_to_words[pair].add(word_id)

    merges = []
    # 4. BPE loop
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        # Remove zero-frequency pairs if necessary.
        pair_counts = Counter({
            pair: freq
            for pair, freq in pair_counts.items()
            if freq > 0
        })
        if not pair_counts:
            break

        # Highest frequency.
        # Lexicographically largest pair breaks ties.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair],pair,),)
        merges.append(best_pair)

        # append merged token to vocabulary.
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # Only process words containing best_pair
        affected_words = list(pair_to_words[best_pair])
        for word_id in affected_words:
            tokens = word_tokens[word_id]
            freq = word_freqs[word_id]

            old_pairs = get_pairs(tokens)

            # Merge in place
            merge_pair_in_place(tokens,best_pair,)

            new_pairs = get_pairs(tokens)

            # Remove old pair contributions
            for pair, count in old_pairs.items():
                pair_counts[pair] -= count * freq

            # Add new pair contributions
            for pair, count in new_pairs.items():
                pair_counts[pair] += count * freq

            # Update pair -> words index
            for pair in old_pairs:
                if pair not in new_pairs:
                    pair_to_words[pair].discard(word_id)

            for pair in new_pairs:
                pair_to_words[pair].add(word_id)

        # best_pair should no longer exist for its occurrences have been merged
        pair_to_words[best_pair].clear()
    return vocab, merges
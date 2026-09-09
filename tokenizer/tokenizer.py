import os
import json
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import time
import regex
from pretokenization import (
    find_chunk_boundaries,
    DEFAULT_PRETOKEN_PATTERN,
)

SPECIAL_TOKEN = b"<|endoftext|>"
# Make more chunks than workers.
# This greatly reduces RAM required by each worker while keeping all CPUs busy.
CHUNKS_PER_WORKER = 4
# Limit per-worker pre-token cache.
# Set to None for unlimited cache.
MAX_CACHE_SIZE = 250_000
def load_vocab(vocab_path: str) -> dict[bytes, int]:
    """
    Load vocab.json produced by the BPE training script: 
    format: token_id -> token_bytes.hex()
        {
            "0": "00",
            "1": "01",
            ...
        }
    tokenizer needs:
        token_bytes -> token_id
    """
    with open(vocab_path,"r",encoding="utf-8",) as f:
        data = json.load(f)
    return {
        bytes.fromhex(token_hex): int(token_id)
        for token_id, token_hex in data.items()
    }


def load_merges(merges_path: str) -> list[tuple[bytes, bytes]]:
    # Each line contains:token1_hex token2_hex
    merges = []
    with open(merges_path,"r",encoding="utf-8",) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token1_hex, token2_hex = line.split()
            merges.append(
                (
                    bytes.fromhex(token1_hex),
                    bytes.fromhex(token2_hex),
                )
            )
    return merges


# Apply BPE to ONE regex pre-token, NOT be applied across pre-token boundaries.
def bpe_encode_pretoken(
    pretoken: bytes,
    vocab: dict[bytes, int],
    merge_rank: dict[tuple[bytes, bytes], int],
) -> tuple[int, ...]:
    if not pretoken:
        return ()

    # Start with individual bytes. list[bytes]
    tokens = [bytes([b])for b in pretoken]

    while len(tokens) >= 2:
        best_pair = None
        best_rank = None
        # highest priority (= lowest merge rank)
        for i in range(len(tokens) - 1):
            pair = (tokens[i],tokens[i + 1],)
            rank = merge_rank.get(pair)
            if rank is not None:
                if (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

        if best_pair is None:
            break

        # Apply this merge everywhere, left-to-right
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == best_pair[0]
                and tokens[i + 1] == best_pair[1]
            ):
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i] )
                i += 1
        tokens = new_tokens

    try:
        return tuple( vocab[token]for token in tokens)

    except KeyError as e:
        raise KeyError(
            f"BPE produced token not present in vocabulary: "
            f"{e.args[0]!r}"
        ) from e


# Multiprocessing globals
_WORKER_VOCAB = None
_WORKER_MERGE_RANK = None
_WORKER_PATTERN = None
_WORKER_CACHE = None
_WORKER_SPECIAL_ID = None
_WORKER_DTYPE = None
def init_worker(vocab, merges, pattern, dtype_str):
    global _WORKER_VOCAB
    global _WORKER_MERGE_RANK
    global _WORKER_PATTERN
    global _WORKER_CACHE
    global _WORKER_SPECIAL_ID
    global _WORKER_DTYPE

    _WORKER_VOCAB = vocab
    # Build this ONCE per worker instead of once per chunk / pre-token.
    _WORKER_MERGE_RANK = {
        pair: rank
        for rank, pair in enumerate(merges)
    }

    # Compile regex once per worker.
    _WORKER_PATTERN = regex.compile( pattern)
    # Cache for words
    #     pre-token bytes -> tuple[token IDs]
    _WORKER_CACHE = {}

    try:
        _WORKER_SPECIAL_ID = vocab[SPECIAL_TOKEN]
    except KeyError as e:
        raise ValueError(
            f"Special token "
            f"{SPECIAL_TOKEN!r} "
            f"is missing from vocabulary."
        ) from e

    _WORKER_DTYPE = np.dtype(dtype_str)

def decode_valid_document(
    raw_document: bytes,
):
    """
    Match process_chunk() cleaning behavior from pretokenization.py.
    Returns:
        str  -> valid document
        None -> document should be skipped
    """

    if not raw_document or b"\x00" in raw_document:
        return None

    # Same strict UTF-8 behavior as training.
    try:
        document = raw_document.decode("utf-8",errors="strict",)
    except UnicodeDecodeError:
        return None

    # Same control-character validation as training.
    has_bad_control = any(
        ord(ch) < 32
        and ch not in ("\n", "\r", "\t")
        for ch in document
    )
    if has_bad_control:
        return None

    return document

# Encode one normal document
def encode_document(
    document: str,
    output_ids: list[int],
):
    """
    Apply exactly the same regex pre-tokenization used during BPE training
    Then run BPE independently on each pre-token.
    """
    global _WORKER_CACHE
    pre_tokens = _WORKER_PATTERN.findall( document )
    for pre_token in pre_tokens:
        pretoken_bytes = pre_token.encode("utf-8")
        # Cache lookup
        cached = _WORKER_CACHE.get( pretoken_bytes)
        if cached is None:
            cached = bpe_encode_pretoken( pretoken_bytes, _WORKER_VOCAB,  _WORKER_MERGE_RANK,)
            # Keep cache from growing without bound.
            if (
                MAX_CACHE_SIZE is None
                or len(_WORKER_CACHE) < MAX_CACHE_SIZE
            ):
                _WORKER_CACHE[ pretoken_bytes ] = cached

        output_ids.extend( cached)


# Encode one file chunk
def tokenize_chunk(args):
    file_path, start, end = args
    # Read this chunk
    with open(file_path,"rb", buffering=1024 * 1024,) as f:
        f.seek(start)
        raw_chunk = f.read( end - start )

    # Split SPECIAL_TOKEN but PRESERVE it in output.
    # bytes.split() is sufficient because currently only one special token.
    raw_documents = raw_chunk.split(  SPECIAL_TOKEN )

    output_ids = []
    accepted_docs = 0
    skipped_docs = 0

    last_index = len(raw_documents) - 1

    for doc_index, raw_document in enumerate(raw_documents):
        # Encode normal document
        if raw_document:
            document = decode_valid_document( raw_document )
            if document is None:
                skipped_docs += 1
            else:
                encode_document(document, output_ids,)
                accepted_docs += 1

        # Restore special token that .split() removed: append special token's id at the tail
        # Example:
        #   lastWord<|endoftext|>  -> 
        #       encode("lastWord")
        #       special_token_id
        if doc_index < last_index:
            output_ids.append( _WORKER_SPECIAL_ID)

    # --------------------------------------------------------
    # Convert here in the worker.
    # Do NOT send millions of Python ints back to parent.
    # uint16/uint32 arrays are dramatically smaller.
    # --------------------------------------------------------
    token_array = np.asarray( output_ids, dtype=_WORKER_DTYPE)
    return token_array, accepted_docs, skipped_docs


# Full-file tokenization
def tokenize_file(
    input_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    num_workers: int | None = None,
):
    # CPU configuration
    logical_cpus = os.cpu_count() or 1
    if num_workers is None:
        # Leave one logical CPU available to Windows.
        num_workers = max( 1, logical_cpus - 1,)
    print("=" * 70)
    print("BPE File Tokenization")
    print("=" * 70)
    print(f"Logical CPUs: {logical_cpus}")
    print(f"Workers:      {num_workers}")
    # Load tokenizer
    print()
    print("Loading vocabulary...")
    vocab = load_vocab(vocab_path)
    print(f"Vocabulary entries: "f"{len(vocab):,}")
    print("Loading merges...")
    merges = load_merges(merges_path)
    print(f"Merges: "f"{len(merges):,}")

    # Sanity checks
    if not vocab:
        raise ValueError("Vocabulary is empty.")

    # Every raw byte should normally exist in BPE vocabulary.
    missing_bytes = [
        b
        for b in range(256)
        if bytes([b]) not in vocab
    ]
    if missing_bytes:
        raise ValueError(
            "Vocabulary does not contain all 256 "
            f"raw byte tokens. Missing {len(missing_bytes)} bytes."
        )

    # Find safe chunk boundaries
    special_token = b"<|endoftext|>"
    print()
    print("Finding safe chunk boundaries...")

    max_token_id = max( vocab.values())
    if ( max_token_id <= np.iinfo(np.uint16).max):
        dtype = np.dtype(np.uint16)
    elif ( max_token_id <= np.iinfo(np.uint32).max ):
        dtype = np.dtype( np.uint32)
    else:
        dtype = np.dtype(  np.uint64)
    print( f"Output dtype:       "f"{dtype}")
    # Having several chunks per worker:
    #   - lowers per-task RAM
    #   - improves CPU load balancing
    #   - still reuses worker cache
    desired_num_chunks = max(num_workers, num_workers * CHUNKS_PER_WORKER)
    print()
    print( f"Desired chunks:     " f"{desired_num_chunks}")
    print( "Finding safe chunk boundaries...")
    with open(input_path,"rb",buffering=1024 * 1024,) as f:
        boundaries = find_chunk_boundaries(f,desired_num_chunks,special_token,)

    chunks = [
        (
            input_path,
            boundaries[i],
            boundaries[i + 1],
        )
        for i in range(len(boundaries) - 1)
    ]
    print(f"Actual chunks: " f"{len(chunks)}")

    # Parallel tokenization
    # writes each completed ordered result to disk as it arrives.
    print()
    print(f"Tokenizing with "f"{num_workers} processes...")
    total_tokens = 0
    total_accepted_docs = 0
    total_skipped_docs = 0
    # Remove an old partial output if present.
    if os.path.exists(output_path):
        os.remove(output_path )
    with open( output_path, "wb", buffering=8 * 1024 * 1024,) as output_file:
        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(
                vocab,
                merges,
                DEFAULT_PRETOKEN_PATTERN,
                dtype.str,
            ),
        ) as pool:
            # imap() preserves input order.
            # This is essential because corpus token order must match file order
            iterator = pool.imap(tokenize_chunk, chunks, chunksize=1)
            tokenize_start = time.perf_counter()
            for chunk_index, result in enumerate(iterator,start=1):
                token_array, accepted_docs, skipped_docs = result
                token_array.tofile(output_file) # Stream directly to .bin

                total_tokens += len( token_array)
                total_accepted_docs += ( accepted_docs)
                total_skipped_docs += ( skipped_docs)
                elapsed = time.perf_counter() - tokenize_start
                tokens_per_sec = ( total_tokens / elapsed if elapsed > 0 else 0)
                chunks_per_sec = ( chunk_index / elapsed if elapsed > 0 else 0 )
                remaining_chunks = ( len(chunks) - chunk_index )
                eta_seconds = (remaining_chunks / chunks_per_sec if chunks_per_sec > 0  else 0)
                print(f"\r"
                    f"Chunks: {chunk_index:,}/{len(chunks):,} | "
                    f"Tokens: {total_tokens:,} | "
                    f"Speed: {tokens_per_sec:,.0f} tok/s | "
                    f"ETA: {eta_seconds / 60:.1f} min | "
                    f"Skipped docs: {total_skipped_docs:,}",
                    end="",
                    flush=True,
                )

    file_size_mb = os.path.getsize(output_path)/ (1024 ** 2)

    print()
    print("=" * 70)
    print( f"Accepted documents: "  f"{total_accepted_docs:,}")
    print( f"Skipped documents:  "f"{total_skipped_docs:,}")
    print( f"Number of tokens:   " f"{total_tokens:,}")
    print( f"Output dtype:        " f"{dtype}" )
    print( f"Output size:         " f"{file_size_mb:.2f} MB" )
    print(f"Saved to:            "f"{output_path}" )
    print("=" * 70)
    print( "Tokenization complete")
    print("=" * 70)


# Windows multiprocessing
if __name__ == "__main__":
    mp.freeze_support()
    tokenize_file(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_path="vocab.json",
        merges_path="merges.txt",
        output_path="TinyStories-train.bin",

        # None:
        #     os.cpu_count() - 1
        # You can benchmark 8 / 12 / 16 manually.
        num_workers=None,
    )
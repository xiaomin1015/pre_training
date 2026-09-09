import os
from typing import BinaryIO
from collections import Counter
import regex
from multiprocessing import Pool

DEFAULT_PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
"""
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
"""
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    # manipulate the file's cursor without loading contents
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args):
    filename, start, end, pattern, split_special_token = args

    with open(filename, "rb") as f:
        f.seek(start)
        raw_chunk = f.read(end - start)

    if not split_special_token:
        raw_documents = [raw_chunk]
    else:
        delimiter = b"|".join(
            regex.escape(
                token.encode("utf-8")
                if isinstance(token, str)
                else token
            )
            for token in split_special_token
        )
        raw_documents = regex.split(delimiter,raw_chunk,)

    counts = Counter()
    skipped_binary_docs = 0
    skipped_utf8_docs = 0
    accepted_docs = 0
    skipped_binary_bytes = 0
    skipped_utf8_bytes = 0
    for raw_document in raw_documents:
        if not raw_document:
            continue
        # --------------------------------------------------------
        # NUL is impossible in normal TinyStories prose.
        # but there's a very strong indicator that this
        # document contains binary contamination.
        # --------------------------------------------------------
        if b"\x00" in raw_document:
            skipped_binary_docs += 1
            skipped_binary_bytes += len(raw_document)
            continue

        # Do NOT use errors="ignore" here because it would silently
        # hide corrupt bytes such as \xab, \x8d, \xff, etc.
        try:
            document = raw_document.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            skipped_utf8_docs += 1
            skipped_utf8_bytes += len(raw_document)
            continue

        # Optional additional control-character validation
        # Normal text may contain: \n = 10 \r = 13  \t = 9
        # Other C0 control characters are suspicious.
        has_bad_control = any(
            ord(ch) < 32
            and ch not in ("\n", "\r", "\t")
            for ch in document
        )
        if has_bad_control:
            skipped_binary_docs += 1
            skipped_binary_bytes += len(raw_document)
            continue

        pre_tokens = regex.findall(pattern, document)
        counts.update(pre_tokens)
        accepted_docs += 1

    if skipped_binary_docs or skipped_utf8_docs:
        print(
            f"[CLEAN] "
            f"chunk={start:,}-{end:,} | "
            f"accepted_docs={accepted_docs:,} | "
            f"binary_docs={skipped_binary_docs:,} | "
            f"invalid_utf8_docs={skipped_utf8_docs:,} | "
            f"binary_bytes={skipped_binary_bytes:,} | "
            f"invalid_utf8_bytes={skipped_utf8_bytes:,}",
            flush=True,
        )
    return counts

def parallel_pretokenize(filename, num_processes, split_special_token, pattern:str|None=None):
    # First find the chunk boundaries
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            split_special_token[0].encode("utf-8"),
        )

    # Convert boundaries into (start, end) pairs
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    if pattern is None:
        pattern = DEFAULT_PRETOKEN_PATTERN
    # Give each process one or more chunks
    args = [
        (filename, start, end, pattern, split_special_token)
        for start, end in chunks
    ]
    # safe with independent file handles and independent file positions.
    # capacity of 'num_processes' process, not necessary receive 'num_processes' task
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, args)

    # dictionary count frenquency of all words
    total_counts = Counter()
    # Merge the counters from all processes
    for counts in results:
        total_counts.update(counts)

    return total_counts


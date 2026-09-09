import os
import time
import json
import logging
import multiprocessing as mp
import tracemalloc
from pathlib import Path

from BPE import train_bpe_from_counts
from pretokenization import parallel_pretokenize


INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_OUTPUT_PATH = "vocab.json"
MERGES_OUTPUT_PATH = "merges.txt"
LOG_PATH = "tokenizer_training.log"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

NUM_PROCESSES = None # None = automatically use almost all available logical CPUs.
CPU_RESERVE = 1 # Leave this many logical CPUs available to Windows.

# tracemalloc is useful for profiling, but it can noticeably slow down Python-heavy workloads
# False = maximum training speed
# True  = measure Python allocation peak memory
ENABLE_TRACEMALLOC = False

logging.basicConfig(
	filename=LOG_PATH,
	level=logging.INFO,
	format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)


def save_vocab(vocab, path):
	"""
	Save BPE vocabulary as:
		token_id -> hexadecimal byte representation
	Example:
		{
			"0": "61",
			"1": "62"
		}
	"""
	data = {
		str(token_id): token_bytes.hex()
		for token_id, token_bytes in vocab.items()
	}

	with open(path,"w",encoding="utf-8",buffering=1024 * 1024,) as f:
		json.dump(data,f,indent=2,)


def save_merges(merges, path):
	"""
	Save BPE merges as pairs of hexadecimal byte strings.
	"""
	with open(path,"w",encoding="utf-8",buffering=1024 * 1024,) as f:
		for token1, token2 in merges:
			f.write(
				f"{token1.hex()} {token2.hex()}\n"
			)


# CPU configuration
def get_worker_count():
	"""
	Determine number of multiprocessing workers.
	On Windows:
		os.cpu_count() normally returns logical CPUs.
	Example:
		i7-12700K -> normally 20 logical CPUs

	With CPU_RESERVE=1:
		workers = 19
	"""

	logical_cpus = os.cpu_count() or 1

	if NUM_PROCESSES is not None:
		workers = int(NUM_PROCESSES)
	else:
		workers = max(1, logical_cpus - CPU_RESERVE)

	return logical_cpus, workers


# ============================================================
# Main training
# ============================================================

def main():
	logical_cpus, num_processes = get_worker_count()

	print("=" * 70)
	print("TinyStories BPE Training")
	print("=" * 70)
	print(f"Input file        : {INPUT_PATH}")
	print(f"Vocabulary size   : {VOCAB_SIZE}")
	print(f"Logical CPUs      : {logical_cpus}")
	print(f"Worker processes  : {num_processes}")
	print(f"tracemalloc       : {ENABLE_TRACEMALLOC}")
	print("=" * 70)
	logger.info("=" * 70)
	logger.info("TinyStories BPE Training")
	logger.info("=" * 70)
	logger.info(f"Input: {INPUT_PATH}")
	logger.info(f"Vocabulary target size: {VOCAB_SIZE}")
	logger.info(f"Special tokens: {SPECIAL_TOKENS}")
	logger.info(f"Logical CPUs: {logical_cpus}")
	logger.info(f"Worker processes: {num_processes}")

	# Optional memory profiling
	if ENABLE_TRACEMALLOC:
		tracemalloc.start()

	total_start = time.perf_counter()

	# 1. Parallel pre-tokenization
	print()
	print(f"Starting pre-tokenization with "f"{num_processes} processes...")
	pretoken_start = time.perf_counter()
	counts = parallel_pretokenize(INPUT_PATH,num_processes,SPECIAL_TOKENS,)
	pretoken_time = (time.perf_counter()- pretoken_start)
	print(f"Pre-tokenization: "f"{pretoken_time:.3f} seconds")
	# Statistics
	num_pretoken_types = len(counts)
	total_token_occurrences = sum(counts.values())
	print(f"Pre-token types: "f"{num_pretoken_types:,}")
	print(f"Token occurrences: "f"{total_token_occurrences:,}")
	print("\nTop 10 pre-tokens:")
	for token, freq in counts.most_common(10):
		print(repr(token), freq)

	# 2. BPE training
	print()
	print("Starting BPE training...")
	bpe_start = time.perf_counter()
	vocab, merges = train_bpe_from_counts(counts,VOCAB_SIZE,SPECIAL_TOKENS)
	bpe_time = (time.perf_counter()- bpe_start)
	print(f"BPE training: "f"{bpe_time:.3f} seconds")
	total_training_time = (time.perf_counter()- total_start)
	print()
	print(f"Total training time: "f"{total_training_time:.3f} seconds")

	# Memory
	if ENABLE_TRACEMALLOC:
		current_memory, peak_memory = (tracemalloc.get_traced_memory())
		tracemalloc.stop()
		peak_memory_gb = (peak_memory/ (1024 ** 3))
		print(f"Peak Python memory: "f"{peak_memory_gb:.3f} GB")
	else:
		peak_memory_gb = None


	# Longest vocabulary token
	longest_token = max(vocab.values(),key=len,)
	longest_token_display = (
		longest_token.decode( "utf-8",errors="replace",)
	)
	print()
	print("Longest token:",repr(longest_token_display),)
	print("Longest token length:",len(longest_token),"bytes")

	# Logging
	logger.info(f"Number of pre-token types: "f"{num_pretoken_types}")
	logger.info(f"Total token occurrences: "f"{total_token_occurrences}")
	logger.info(f"Pre-tokenization time: "f"{pretoken_time:.3f} seconds")
	logger.info(f"BPE training time: "f"{bpe_time:.3f} seconds")
	logger.info(f"Total training time: "f"{total_training_time:.3f} seconds")
	logger.info(f"Vocabulary size: "f"{len(vocab)}")
	logger.info(f"Number of merges: "f"{len(merges)}")
	logger.info(f"Longest token: "f"{longest_token!r}")
	logger.info(f"Longest token length: "f"{len(longest_token)} bytes")
	if peak_memory_gb is not None:
		logger.info(f"Peak Python memory: "f"{peak_memory_gb:.3f} GB")
	else:
		logger.info("Peak memory profiling disabled ""for maximum performance.")

	# 3. Save tokenizer
	print()
	save_start = time.perf_counter()
	save_vocab(vocab,VOCAB_OUTPUT_PATH,)
	save_merges(merges,MERGES_OUTPUT_PATH,)
	save_time = (time.perf_counter()- save_start)
	print(f"Serialization: "f"{save_time:.3f} seconds")
	print()
	print(f"Vocabulary saved to: "f"{Path(VOCAB_OUTPUT_PATH).resolve()}")
	print(f"Merges saved to: "f"{Path(MERGES_OUTPUT_PATH).resolve()}")
	logger.info(f"Vocabulary saved to: "f"{Path(VOCAB_OUTPUT_PATH).resolve()}")
	logger.info(f"Merges saved to: "f"{Path(MERGES_OUTPUT_PATH).resolve()}")
	logger.info(f"Serialization time: "f"{save_time:.3f} seconds")
	logger.info("=" * 70)
	print()
	print("=" * 70)
	print("Training complete.")
	print("=" * 70)



if __name__ == "__main__":
	mp.freeze_support()
	main()
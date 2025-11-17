"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb2_5B".
"""

## Using only 2.5B tokens instead of 10B

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb2_5B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard
total_tokens_target = int(2.5e9) + shard_size # ~2.5B train + 1 shard for val, Change to 10e9 for 10B tokens

try:
    ROOT_DIR = os.path.dirname(__file__)
except NameError:
    ROOT_DIR = os.getcwd()
DATA_CACHE_DIR = os.path.join(ROOT_DIR, local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    total_tokens_written = 0
    progress_bar = None
    try:
      for tokens in pool.imap(tokenize, iter(fw), chunksize=16):
          if total_tokens_written >= total_tokens_target:
              break  # --- stop after ~2.5B tokens total (plus one val shard) ---

          if token_count + len(tokens) < shard_size:
              all_tokens_np[token_count:token_count+len(tokens)] = tokens
              token_count += len(tokens)
              if progress_bar is None:
                  progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
              progress_bar.update(len(tokens))
          else:
              split = "val" if shard_index == 0 else "train"
              filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
              remainder = shard_size - token_count
              progress_bar.update(remainder)
              all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
              write_datafile(filename, all_tokens_np)
              total_tokens_written += shard_size
              shard_index += 1
              progress_bar = None

              all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
              token_count = len(tokens)-remainder

      if token_count != 0 and total_tokens_written < total_tokens_target:
          split = "val" if shard_index == 0 else "train"
          filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
          write_datafile(filename, all_tokens_np[:token_count])
          total_tokens_written += token_count
    finally:
        pool.close()
        pool.terminate()
        pool.join()

print(f"✅ Finished writing ~{total_tokens_written:,} tokens "
      f"({shard_index+1} shards total) to {DATA_CACHE_DIR}")
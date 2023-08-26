# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from llama import Llama

# set RANK to 0
os.environ["RANK"] = "0"
# set WORLD_SIZE to 1
os.environ["WORLD_SIZE"] = "1"
# set MASTER_ADDR to localhost
os.environ["MASTER_ADDR"] = "localhost"
# set MASTER_PORT to 12355
os.environ["MASTER_PORT"] = "12355"



generator = Llama.build(
    ckpt_dir="CodeLlama-7b-Python",
    tokenizer_path="CodeLlama-7b-Python/tokenizer.model",
    max_seq_len=192,
    max_batch_size=4,
 )

prompts = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    """\
import socket

def ping_exponential_backoff(host: str):""",
        """\
import argparse

def main(string: str):
    print(string)
    print(string[::-1])

if __name__ == "__main__":""",]

results = generator.text_completion(
    prompts,
    max_gen_len=1000,
    temperature=0.1,
    top_p=.8,
)

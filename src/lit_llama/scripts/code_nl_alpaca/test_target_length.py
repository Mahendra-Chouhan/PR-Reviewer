import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
print(wd)
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer(Path("checkpoints/lit-llama/tokenizer.model"))
with open("data/code_nl_alpaca/alpaca_gpt4_data.json", "r") as file:
    nl_data = json.load(file)
with open("data/code_nl_alpaca/code_alpaca_20k.json", "r") as file:
    code_data = json.load(file)

nl_data = list(nl_data)
code_data = list(code_data)

gt_x_nl = 0
gt_x_code = 0
gt_256_nl = 0
gt_256_code = 0
gt_512_nl = 0
gt_512_code = 0
total_nl = len(nl_data)
total_code = len(code_data)

x = 1024

for sample in tqdm(nl_data):
    sample_tokens = tokenizer.encode(sample["output"], eos=True)
    tokens_len = len(sample_tokens)
    if tokens_len > 256:
        gt_256_nl += 1
    if tokens_len > 512:
        gt_512_nl += 1
    if tokens_len > x:
        gt_x_nl += 1

for sample in tqdm(code_data):
    sample_tokens = tokenizer.encode(sample["output"], eos=True)
    tokens_len = len(sample_tokens)
    if tokens_len > 256:
        gt_256_code += 1
    if tokens_len > 512:
        gt_512_code += 1
    if tokens_len > x:
        gt_x_code += 1

print(f"NL: {gt_x_nl/total_nl*100:.2f}% of samples ({gt_x_nl}/{total_nl}) are longer than {x} tokens")
print(f"NL: {gt_256_nl/total_nl*100:.2f}% of samples ({gt_256_nl}/{total_nl}) are longer than 256 tokens")
print(f"NL: {gt_512_nl/total_nl*100:.2f}% of samples ({gt_512_nl}/{total_nl}) are longer than 512 tokens")
print(f"Code: {gt_x_code/total_code*100:.2f}% of samples ({gt_x_code}/{total_code}) are longer than {x} tokens")
print(f"Code: {gt_256_code/total_code*100:.2f}% of samples ({gt_256_code}/{total_code}) are longer than 256 tokens")
print(f"Code: {gt_512_code/total_code*100:.2f}% of samples ({gt_512_code}/{total_code}) are longer than 512 tokens")


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
with open("data/crer_data/code_refinement/test.json", "r") as file:
    test_cr = json.load(file)
with open("data/crer_data/comment_generation/test.json", "r") as file:
    test_cg = json.load(file)
with open("data/crer_data/quality_estimation/test.json", "r") as file:
    test_qe = json.load(file)

test_cr = list(test_cr)
test_cg = list(test_cg)
test_qe = list(test_qe)

gt_x_cr = 0
gt_x_cg = 0
gt_256_cr = 0
gt_256_cg = 0
gt_512_cr = 0
gt_512_cg = 0
gt_4_qe = 0
gt_1024_cr = 0
gt_1024_cg = 0
total_cr = len(test_cr)
total_cg = len(test_cg)
total_qe = len(test_qe)
x = 128

for sample in tqdm(test_cr):
    sample_tokens = tokenizer.encode(sample["output"], eos=True)
    tokens_len = len(sample_tokens)
    if tokens_len > 256:
        gt_256_cr += 1
    if tokens_len > 512:
        gt_512_cr += 1
    if tokens_len > 1024:
        gt_1024_cr += 1
    if tokens_len > x:
        gt_x_cr += 1

for sample in tqdm(test_cg):
    sample_tokens = tokenizer.encode(sample["output"], eos=True)
    tokens_len = len(sample_tokens)
    if tokens_len > 256:
        gt_256_cg += 1
    if tokens_len > 512:
        gt_512_cg += 1
    if tokens_len > 1024:
        gt_1024_cg += 1
    if tokens_len > x:
        gt_x_cg += 1

for sample in tqdm(test_qe):
    sample_tokens = tokenizer.encode(sample["output"], eos=True)
    tokens_len = len(sample_tokens)
    if tokens_len > 4:
        gt_4_qe += 1

print(f"CR: {gt_x_cr/total_cr*100:.2f}% of samples ({gt_x_cr}/{total_cr}) are longer than {x} tokens")
print(f"CR: {gt_256_cr/total_cr*100:.2f}% of samples ({gt_256_cr}/{total_cr}) are longer than 256 tokens")
print(f"CR: {gt_512_cr/total_cr*100:.2f}% of samples ({gt_512_cr}/{total_cr}) are longer than 512 tokens")
print(f"CR: {gt_1024_cr/total_cr*100:.2f}% of samples ({gt_1024_cr}/{total_cr}) are longer than 1024 tokens")
print(f"CG: {gt_x_cg/total_cg*100:.2f}% of samples ({gt_x_cg}/{total_cg}) are longer than {x} tokens")
print(f"CG: {gt_256_cg/total_cg*100:.2f}% of samples ({gt_256_cg}/{total_cg}) are longer than 256 tokens")
print(f"CG: {gt_512_cg/total_cg*100:.2f}% of samples ({gt_512_cg}/{total_cg}) are longer than 512 tokens")
print(f"CG: {gt_1024_cg/total_cg*100:.2f}% of samples ({gt_1024_cg}/{total_cg}) are longer than 1024 tokens")
print(f"QE: {gt_4_qe/total_qe*100:.2f}% of samples ({gt_4_qe}/{total_qe}) are longer than 4 tokens")
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

with open("data/tufano_data/comment_generation/train.json", "r") as file:
    train_cg = json.load(file)
with open("data/tufano_data/comment_generation/val.json", "r") as file:
    val_cg = json.load(file)
with open("data/tufano_data/comment_generation/test.json", "r") as file:
    test_cg = json.load(file)

# with open("data/tufano_data/code_refinement/train.json", "r") as file:
#     train_cg = json.load(file)
# with open("data/tufano_data/code_refinement/val.json", "r") as file:
#     val_cg = json.load(file)
# with open("data/tufano_data/code_refinement/test.json", "r") as file:
#     test_cg = json.load(file)

train_cg = list(train_cg)
val_cg = list(val_cg)
test_cg = list(test_cg)
all_cg = train_cg + val_cg + test_cg

gt_x_cg = 0
gt_256_cg = 0
gt_512_cg = 0
gt_1024_cg = 0
total_cg = len(test_cg)
x = 512

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

print(f"Total: {total_cg}")
print(f"GT 256: {gt_256_cg}")
print(f"GT 512: {gt_512_cg}")
print(f"GT 1024: {gt_1024_cg}")
print(f"GT {x}: {gt_x_cg}")
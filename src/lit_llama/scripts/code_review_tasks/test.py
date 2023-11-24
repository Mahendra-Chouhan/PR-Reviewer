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
print(tokenizer.pad_id)
print(tokenizer.eos_id)
print(tokenizer.decode(torch.tensor(1939)))
print(tokenizer.encode("yes"))
print(tokenizer.encode("no"))
# print(len(tokenizer.encode("Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.\n\n### Instruction:\nDetermine whether the provided diff hunk requires a code review. Respond with either 'yes' or 'no'.\n\n\n### Input:\n{example['input']}\n\n", bos=True, eos=False)))
# string = "fuck you\n\n### Response:\n"
# print(string[:-len("\n\n### Response:\n")])
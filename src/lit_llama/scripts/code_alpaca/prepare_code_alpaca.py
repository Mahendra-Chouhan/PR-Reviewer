"""Implementation derived from https://github.com/tloen/alpaca-lora"""
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


DATA_FILE = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
DATA_FILE_NAME = "code_alpaca_20k.json"
IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/alpaca"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_size: int = 1000,
    max_seq_length: int = 2048,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = DATA_FILE_NAME
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / data_file_name
    # download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)
    
    with open(file_path, "r") as file:
        data = json.load(file)

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, validation_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, validation_set = list(train_set), list(validation_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(validation_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    validation_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(validation_set)]
    torch.save(validation_set, file_path.parent / "validation.pt")


def download(file_path: Path):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]

    # Tokenize the prompt and the prompt with response
    # First examine the total token length of the prompt and response
    # encoded_full_prompt_and_response = tokenizer.my_encode_for_prompt_and_output(full_prompt_and_response, bos=True, eos=True, max_length=max_length)
    # masked_tokens_num = 0
    # # if not exceed max_length, then we can use the original tokens
    # if encoded_full_prompt_and_response is not None:
    #     encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    #     masked_tokens_num = len(encoded_full_prompt)
    # else we need to truncate the tokens of the input prompt
    encoded_full_prompt = [tokenizer.bos_id] + tokenizer.processor.encode(full_prompt[:-len("\n\n### Response:\n")])
    encoded_response_start = tokenizer.processor.encode("\n\n### Response:\n")
    encoded_response_context = tokenizer.processor.encode(example["output"]) + [tokenizer.eos_id]
    if len(encoded_full_prompt) + len(encoded_response_start) + len(encoded_response_context) <= max_length:
        encoded_full_prompt = encoded_full_prompt + encoded_response_start
        encoded_full_prompt_and_response = encoded_full_prompt + encoded_response_context
        masked_tokens_num = len(encoded_full_prompt)
    else:
        encoded_response = encoded_response_start + encoded_response_context
        # truncate the input prompt since the total length exceeds max_length
        if len(encoded_response) <= max_length//2:
            encoded_full_prompt = encoded_full_prompt[:max_length - len(encoded_response)]
            encoded_full_prompt_and_response = encoded_full_prompt + encoded_response
        elif len(encoded_full_prompt) <= max_length//2:
            encoded_response = encoded_response[:max_length - len(encoded_full_prompt)]
            encoded_full_prompt_and_response = encoded_full_prompt + encoded_response
        else:
            encoded_full_prompt = encoded_full_prompt[:max_length//2]
            encoded_response = encoded_response[:max_length//2]
            encoded_full_prompt_and_response = encoded_full_prompt + encoded_response
        assert len(encoded_full_prompt_and_response) <= max_length
        masked_tokens_num = len(encoded_full_prompt) + len(encoded_response_start)
    encoded_full_prompt = torch.tensor(encoded_full_prompt, dtype=torch.int, device=None)
    encoded_full_prompt_and_response = torch.tensor(encoded_full_prompt_and_response, dtype=torch.int, device=None)
    # encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    # encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:masked_tokens_num] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}

def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)

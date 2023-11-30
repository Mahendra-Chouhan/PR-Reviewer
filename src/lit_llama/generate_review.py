import sys
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from torchinfo import summary
import json, re

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup, quantization
from scripts.prepare_alpaca import generate_prompt
from utils.smooth_bleu import bleu_fromstr
from lit_llama.adapter_v2 import add_adapter_v2_parameters_to_linear_layers
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
warnings.filterwarnings(
    # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
    "ignore", 
    message="ComplexHalf support is experimental and many operators don't support it yet"
)
warnings.filterwarnings(
    # Triggered in bitsandbytes/autograd/_functions.py:298
    "ignore", 
    message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
)


def main(
    prompt: str = "",
    input: str = "Review the given diff hunk and provide a constructive code review comment.",
    adapter_path: Optional[Path] = None,
    pretrained_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    quantize: Optional[str] = "gptq.int4",
    block_size: int = 2048,
    max_new_tokens: int = 512,
    top_k: int = 1,
    temperature: float = 1,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    if not adapter_path:
        adapter_path = Path("checkpoints/lit-llama/7B/comment_generation/lit-llama-adapter-finetuned.pth")
    if not pretrained_path:
        pretrained_path = Path(f"./checkpoints/lit-llama/7B/llama-gptq.4bit.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    # Get the path of the GitHub workspace
    github_workspace_path = os.getenv("GITHUB_WORKSPACE")

    # Open and read the "difference_hunk.txt" file
    sample = {}
    sample["instruction"] = input
    with open(input_path, "r") as diff_handle:
        diff = diff_handle.read()
    sample["input"] = f"The diff hunk is: {diff}"



    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)
    
    print("Instuction: " + sample["instruction"])
    print("Input: " + sample["input"], end="/n")
    
    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    # pretrained_checkpoint = lazy_load(pretrained_path)
    # adapter_checkpoint = lazy_load(adapter_path)
    # print(pretrained_checkpoint, adapter_checkpoint)
    
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name)
            # add_adapter_v2_parameters_to_linear_layers(model)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)
    
    
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)
    print(summary(model), end="/n")
    tokenizer = Tokenizer(tokenizer_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    
    t0 = time.time()
    
    input_data = {"instruction": sample["instruction"], "input": sample["input"]}
    prompt = generate_prompt(input_data)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device, max_length =2048)
    #print(f"encoded:{encoded}")
    output = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k)

    # output = generate(
    #     model,
    #     idx=encoded,
    #     max_seq_length=block_size,
    #     max_new_tokens=max_new_tokens,
    #     temperature=temperature,
    #     top_k=top_k,
    #     eos_id=tokenizer.eos_id
    # )
    model.reset_cache()
    output = tokenizer.decode(output)
    print("complete response:", output)
    output = output.split("### Response:\n")[1].strip() if len(output.split("### Response:\n")) > 1 else "" 
    
    print("Instuction: " + sample["instruction"])
    print("Input: " + sample["input"])
    print("Respond: " + output)

    t = time.time() - t0
    output = f"### AI Pull Request Review: \n {output} "
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used are: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    # Write the comment to the output file
    with open(output_path, "a") as f:
        f.write(f"{output}")
    return output

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)

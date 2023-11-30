import sys
import os
import time
import warnings
from pathlib import Path
from typing import Optional

from generate_review import main as generate_review
from generate_qaulity import main as generate_qaulity
from generate_refinement import main as generate_refinement

def main(
    prompt: str = "",
    input: str = "",
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
    output = {}
    output["Qualit"] = generate_qaulity(pretrained_path=pretrained_path,input_path=input_path, output_path=output_path,
                                        max_new_tokens=max_new_tokens)
    output["Review"] = generate_review(pretrained_path=pretrained_path, input_path=input_path, output_path=output_path,
                                       max_new_tokens=max_new_tokens)
    output["Refinement"] = generate_refinement(pretrained_path= pretrained_path, input_path=input_path, output_path=output_path,
                                               max_new_tokens=max_new_tokens, 
                                               review=output["Review"])
    
    with open(output_path, "w") as f:
        output = f"""##### AI Code review
        **Review** : {output["Review"]}
        **Refined code**: 
        ```
            {output["Refinement"]}
        ```
        """
        f.write(f"{output}")
    return output

if __name__ == "__main__":
    from jsonargparse import CLI
    import torch

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    print("***********final output**********")
    print(CLI(main))

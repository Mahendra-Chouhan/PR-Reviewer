import os
from pathlib import Path
from typing import Optional

import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class Tokenizer:
    """Tokenizer for LLaMA."""

    def __init__(self, model_path: Path) -> None:
        self.processor = SentencePieceProcessor(model_file=str(model_path))
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            if len(tokens) > max_length:
                print(f"Warning: truncating sequence of {len(tokens)} tokens to {max_length} tokens.")
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int, device=device)
    
    def my_encode_for_generate(self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        tokens_start = self.processor.encode(string[:-len("\n\n### Response:\n")])
        tokens_end = self.processor.encode(string[-len("\n\n### Response:\n"):])
        tokens = tokens_start + tokens_end
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            if len(tokens) > max_length:
                # truncate sequence before "\n\n### Response:\n"
                tokens = tokens_start[:max_length-len(tokens_end)] + tokens_end     
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int, device=device)
    
    def my_encode_for_prompt(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            if len(tokens) > max_length:
                print(f"Warning: truncating sequence of {len(tokens)} tokens to {max_length} tokens.")
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int, device=device)
    
    def my_encode_for_prompt_and_output(
        self,
        string: str,
        bos: bool = True,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        tokens = self.processor.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            if len(tokens) > max_length:
                return None
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.processor.decode(tokens.tolist())

    @staticmethod
    def train(input: str, destination: str, vocab_size=32000) -> None:
        model_prefix = os.path.join(destination, "tokenizer")
        SentencePieceTrainer.Train(input=input, model_prefix=model_prefix, vocab_size=vocab_size)

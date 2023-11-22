import glob
import os
import torch
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
# GPU
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    n_ctx = 1024
    )
repo_path = os.path(os.getenv("GITHUB_WOKRSPACE"))
repo_path = os.path.expandsvars(repo_path)
with open("{repo_path}/.github/workflows/mydiff.txt", "r") as diff_handle:
  diff = diff_handle.read()

prompt = ("Review the code difference and suggest changes: \n" + diff)
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''
    
response=lcpp_llm(prompt=prompt_template, max_tokens=1024, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=150, echo=False)
response = response["choices"][0]["text"]

# Write the comment to the output file
with open("src/files/output.txt", "a") as f:
  f.write(f"{response}\n")
  f.write("\n")
  f.write("\n")

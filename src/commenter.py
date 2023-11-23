import glob
import os
import re

#try and except block for retrieving dependencies from cache.. and stuff.
try:
  import sys
  cache_path = os.path.join(os.getenv("GITHUB_WORKSPACE"), ".cache/pip")
  cache_path = os.path.expandvars(cache_path)
  sys.path.append(cache_path)
  import torch
  from huggingface_hub import hf_hub_download
  from llama_cpp import Llama

  
except:
  import torch
  from huggingface_hub import hf_hub_download
  from llama_cpp import Llama
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

def minify(code):
  """Minify code.

  Args:
    code: The code to minify.

  Returns:
    The minified code.
  """

  code = re.sub(r'#[^\n]*\n', '', code)
  code = re.sub(r'\s+', ' ', code)
  code = re.sub(r'\n\n+', '\n', code)
  code = re.sub(r'\s+$', '', code)

  return code


# Get the path to the src directory
src_path = os.path.join(os.getenv("GITHUB_WORKSPACE"), "src")
src_path = os.path.expandvars(src_path)

# Find all files in the src directory
all_files = glob.glob(os.path.join(src_path, "*"))

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
# GPU
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    n_ctx = 1024
    )

# Loop through all files in the src directory
for file in all_files:

  # Check if the file is a regular file
  if os.path.isfile(file):

    # Print the file name
    print(f"Processing file: {file}")

    # Open the file and read
    with open(file, "r") as f:
      code = f.read()
      code = minify(code)

    prompt = ("Suggest helpful changes to the code: \n" + code)
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

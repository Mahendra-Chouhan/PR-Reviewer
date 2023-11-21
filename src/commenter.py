import glob
import os
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"


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
    n_gpu_layers=32 # Change this value based on your model and your GPU VRAM pool.
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

    prompt = ("Suggest helpful changes to the code: function additionFunction(a, b) {  return a + b; } let num1 = 5; let num2 = 10; let sum = additionFunction(num1, num2); console.log('Sum of given numbers is :', sum);")
    prompt_template=f'''

    USER:{prompt}
    
    CODE-ASSISTANT:
    '''
    
    response=lcpp_llm(prompt=prompt_template, max_tokens=256, temperature=0.5, top_p=0.95, repeat_penalty=1.2, top_k=150, echo=True)
    response = response["choices"][0]["text"]


    # Write the comment to the output file
    with open("src/files/output.txt", "a") as f:
      f.write(f"{response}\n")
      f.write("\n")
      f.write("\n")

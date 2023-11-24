import os
import torch
from huggingface_hub import hf_hub_download
from transformers import BartTokenizer

model_name_or_path = "msinghC/llm-pr-review"
model_basename = "llama-gptq.4bit.pth"

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Get the path of the GitHub workspace
github_workspace_path = os.getenv("GITHUB_WORKSPACE")

# Open and read the "difference_hunk.txt" file
with open(f"{github_workspace_path}/difference_hunk.txt", "r") as diff_handle:
 diff = diff_handle.read()

prompt = ("""you are a code review assistant. Concisely summarize the major code difference in ONE LINE, explaining the difference in a way humans can understand. do it in the format:

CHANGE: Explanation.

Here is the code difference: """ + diff)
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully. 

USER: {prompt}

ASSISTANT:
'''

# Load the model
model = torch.load(model_path)

# Tokenize the prompt and input
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.pad_token = tokenizer.eos_token
encoded_prompt = tokenizer(prompt_template, truncation=True, padding=True, return_tensors="pt")
encoded_input = tokenizer(diff, truncation=True, padding=True, return_tensors="pt")

# Generate the response
with torch.no_grad():
  output = generate(
      input_ids=encoded_input["input_ids"],
      attention_mask=encoded_input["attention_mask"],
      decoder_start_token_id=tokenizer.pad_token_id,
      max_length=1536,
      num_beams=4,
  )
decoded_output = tokenizer.decode(output[0], clean_up_tokenization=True)

# Write the response to the output file
with open("src/files/output.txt", "a") as f:
 f.write(f"{decoded_output}")

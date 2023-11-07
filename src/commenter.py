import glob
import torch
import sys
from transformers import AutoModel, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m-bimodal"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

code = []

files = glob.glob("*.py")
for file in files:
  with open(file, "r") as f:
    code.append(f.read())

# Convert the list of strings to a single string
code_string = "\n".join(code)

input_ids = tokenizer(code_string, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids, max_length=20)
tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

output_handle = open("src/files/output.txt", "w+")
output_handle.write(f'{tokens}')
output_handle.close()

sys.exit(0)


import glob
import os
import torch
import sys
from transformers import AutoModel, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m-bimodal"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# Get the current directory
current_dir = os.getcwd()

# Get a list of all the files in the current directory, except for the current script
files = glob.glob("*.py")
files = [file for file in files if file != os.path.basename(__file__)]
output = ""
# Read each .py file and print its contents
for py_file in files:
    with open(py_file, 'r') as f:
        code = f.read()
        input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)
        generated_ids = model.generate(input_ids, max_length=20)
        xyz= tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(xyz)
        output += xyz
print(output)
output_handle = open("src/files/output.txt", "w+")
output_handle.write(f'{output}')
output_handle.close()
sys.exit(0)


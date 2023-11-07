import glob
import os
import torch
from transformers import AutoModel, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m-bimodal"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# Get the path to the src directory
src_path = os.path.join(os.getenv("GITHUB_WORKSPACE"), "src")
src_path = os.path.expandvars(src_path)
print(src_path)

# Find all files in the src directory
all_files = glob.glob(os.path.join(src_path, "*"))
print(all_files)

# Loop through all files in the src directory
for file in all_files:

  # Check if the file is a regular file
  if os.path.isfile(file):

    # Print the file name
    print(f"Processing file: {file}")

    # Open the file and read the code
    with open(file, "r") as f:
      code = f.read()

    # Encode the code into input IDs
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    # Generate a comment for the code
    generated_ids = model.generate(input_ids, max_length=20)
    tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Print the comment
    print(f"Generated comment: {tokens}")

    # Write the comment to the output file
    with open("src/files/output.txt", "a") as f:
      f.write(f"{tokens}\n")

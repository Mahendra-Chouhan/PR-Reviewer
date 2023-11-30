import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

# Get the path of the GitHub workspace
github_workspace_path = os.getenv("GITHUB_WORKSPACE")

# Open and read the "difference_hunk.txt" file
with open(f"{github_workspace_path}/difference_hunk.txt", "r") as diff_handle:
    diff = diff_handle.read()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaModel.from_pretrained("microsoft/codebert-base") 
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model.to(device)

#tokenize the code
tokens = tokenizer.encode(diff, return_tensors = "pt")

#Run CodeBERT on the tokenized code
with torch.no_grad():
    response = model(tokens) 


# Write the comment to the output file
with open("src/files/output.txt", "a") as f:
  f.write(f"{response}")

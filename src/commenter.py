import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

# Get the path of the GitHub workspace
github_workspace_path = os.getenv("GITHUB_WORKSPACE")

# Open and read the "difference_hunk.txt" file
with open(f"{github_workspace_path}/difference_hunk.txt", "r") as diff_handle:
    diff = diff_handle.read()



# Write the comment to the output file
with open("src/files/output.txt", "a") as f:
  f.write(f"{response}")

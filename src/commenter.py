import transformers
import torch

# Load the quantized Llama 7B model
model = transformers.AutoModelForSequenceClassification.from_pretrained("llm-pr-review/llama-gptq.4bit.pth")

# Get the path of the GitHub workspace
github_workspace_path = os.getenv("GITHUB_WORKSPACE")

# Open and read the "difference_hunk.txt" file
with open(f"{github_workspace_path}/difference_hunk.txt", "r") as diff_handle:
    diff = diff_handle.read()

prompt = ("""you are a code review assistant. Concisely summarize the major code difference in ONE LINE, explaining the difference in a way humans can understand. do it in the format:

CHANGE: Explanation.

Here is the code difference: """ + diff)

# Generate the code review comment
encoded_prompt = model.tokenizer(prompt, return_tensors="pt")
output = model(**encoded_prompt)
logits = output.logits
prediction = logits.argmax(-1).item()

# Format the comment response
response = model.config.id2label[prediction]

# Write the comment to the output file
with open("src/files/output.txt", "a") as f:
    f.write(f"{response}")

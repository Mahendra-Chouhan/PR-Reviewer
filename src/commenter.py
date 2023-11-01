
from transformers import AutoModel, AutoTokenizer
import glob

checkpoint = "Salesforce/codet5p-220m-bimodal"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

files = glob.glob("*.py")
output_handle= open(file, "r") as f
for file in files:
    with output_handle:
        code =f.read()

input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids, max_length=20)
xyz= print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# Convert a string of SVG data to an image.
output_handle.write(xyz)
output_handle.close()

sys.exit(0)


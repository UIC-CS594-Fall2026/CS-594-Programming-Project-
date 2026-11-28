from datasets import load_dataset, Image
from collections import defaultdict
from transformers import CLIPModel, CLIPProcessor
import torch
import hashlib

'''availablity = torch.backends.mps.is_available() 
print(availablity) '''   # True or False
DATASET_NAME = "flaviagiammarino/path-vqa"
ds = load_dataset(DATASET_NAME)
part = ds["train"]

groups = defaultdict(list)
raw = part.cast_column("image", Image(decode=False))
for i in range(len(raw)):
    key = hashlib.md5(raw[i]["image"]["bytes"]).hexdigest()
    groups[key].append((i, raw[i]["question"], raw[i]["answer"]))
print(len(groups))

device = "mps" if torch.backends.mps.is_available() else "cpu"
#print(device)
processor = CLIPProcessor.from_pretrained("vinid/plip")
model = CLIPModel.from_pretrained("vinid/plip")
model = model.to(device)
model.eval()

img = part[0]["image"]
if img.mode != "RGB":
    img = img.convert("RGB")
inputs = processor(images = img, return_tensors="pt")
inputs = inputs.to(device)

#print(inputs["pixel_values"].shape)

with torch.no_grad():
    features = model.get_image_features(**inputs)

emb = features.pooler_output
emb = emb/emb.norm(dim =1, keepdim = True)
print(emb.shape)
print(emb.norm())
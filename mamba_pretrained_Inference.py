# mamba_pretrained_Inference.py
import warnings
warnings.filterwarnings("ignore", message="The fast path is not available because one of")
warnings.filterwarnings("ignore", message="The 'batch_size' argument of MambaCache is deprecated")

import torch
from transformers import MambaForCausalLM, AutoTokenizer

model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")


quesations  = input('--> ')

input_data = tokenizer(
    quesations , 
    return_tensors="pt", 
    padding=True, 
    truncation=True, 
    max_length=50
)

input_ids = input_data["input_ids"]
attention_mask = input_data["attention_mask"]


out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100)


result = tokenizer.batch_decode(out)
print(result)

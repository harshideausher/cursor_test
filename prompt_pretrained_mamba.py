import torch
from transformers import AutoTokenizer
from transformers import  MambaForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

while True:

    user_message = input(">>> ")

    prompt = f"Answer the following question.\n{user_message}"
    print(prompt)


    input_ids = torch.LongTensor([tokenizer.encode(prompt)]).cuda()
    print(input_ids)
    
    out = model.generate(
        input_ids=input_ids,
        max_length=128,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.batch_decode(out)[0]
    print("="*80)

    cleaned = decoded.replace(prompt, "")
    print(cleaned)
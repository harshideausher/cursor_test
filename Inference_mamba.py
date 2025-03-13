# Inference_mamba.py
import torch
from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig

def load_model_and_tokenizer(model_path):

    config = MambaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token.replace("<|endoftext|>", "<|pad|>" if "<|pad|>" not in tokenizer.additional_special_tokens else tokenizer.pad_token)

    model = MambaForCausalLM.from_pretrained(model_path, config=config)
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=100, num_return_sequences=1):
    input_text = f"Q: {question}\nA:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in output_sequences]
    return responses

if __name__ == "__main__":
    model_path = '\output\checkpoint-60'
    model, tokenizer = load_model_and_tokenizer(model_path)

    question = "What is the capital of France?"
    responses = generate_response(model, tokenizer, question)

    for idx, response in enumerate(responses, 1):
        print(f"Response {idx}: {response}")
# finetune_mamba.py
import torch
import argparse
import transformers
import json
import os
import random
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, TrainingArguments
from transformers import Trainer

# Define the model configuration
model_config = MambaConfig(
    vocab_size=50280,
    n_layer=24,
    d_model=768,
    ssm_cfg={},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    pad_vocab_size_multiple=8
)

class FAQ_data(Dataset):
    def __init__(self, data_path, tokenizer):
        super(FAQ_data, self).__init__()
        data = []
        print(f"Reading in data from file: {data_path}")
        with open(data_path, "r") as file:
            try:
                data = json.load(file)
            except Exception as e:
                print("JSON processing exception", e)
                data = []

        print(f"Got {len(data)} examples, preprocess...")

        data_dict = self.preprocess(data, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def preprocess(self, examples, tokenizer):
        all_input_ids = []
        print("Tokenizing dataset...")
        for ex in tqdm(examples):
            questions = ex['questions']
            context = ex['context']
            for answer in ex['answers']:
                text = f"{context}\n\nQ: {questions}\nA: {answer}\n"
                tokenized = tokenizer.encode(text)
                all_input_ids.append(torch.LongTensor(tokenized))

            # Generate a negative example
            random_ex = random.choice(examples)
            random_questions = random_ex['questions']
            random_context = random_ex['questions']
            random_answers = " ; ".join(random_ex['answers'])
            text = f"{random_context} \n\nQ: {random_questions}\nA: I don't know.\n"
            tokenized = tokenizer.encode(text)  
            all_input_ids.append(torch.LongTensor(tokenized))

        random.shuffle(all_input_ids)
        return dict(input_ids=all_input_ids, labels=all_input_ids)

@dataclass
class DataCollatorForFAQ_data(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class SFTDataModule():
    def __init__(self, tokenizer, data_path: str):
        self.dataset = FAQ_data(tokenizer=tokenizer, data_path=data_path)
        self.data_collator = DataCollatorForFAQ_data(tokenizer=tokenizer)

class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    def save_model(self, output_dir, _internal_call=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)

        model_config_dict = model_config.to_dict()

        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(model_config_dict, f, indent=4)

def train(args):
    model = MambaForCausalLM.from_pretrained(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token


    data_module = SFTDataModule(
        tokenizer=tokenizer,
        data_path=args.data_path,
    )

    trainer = MambaTrainer(
        model=model,
        train_dataset=data_module.dataset,
        tokenizer=tokenizer, 
        
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output,
            save_total_limit=2,
            logging_steps=50,
            save_steps=500,
            fp16=True,  
        ),
        data_collator=data_module.data_collator,
    )

    trainer.train()
    trainer.save_model(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--tokenizer", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2)  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4) 
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="data/sample_data.json")
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()


    torch.cuda.empty_cache()

    train(args)


    
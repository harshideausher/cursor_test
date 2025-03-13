# dataset_prep.py
import torch
import json
import random
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse


class FAQ_data(Dataset):
    def __init__(self, data_path, tokenizer, output_dir=None):
        super(FAQ_data, self).__init__()
        self.data_path = data_path
        self.output_dir = output_dir  
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

        if self.output_dir:
            self.save_to_jsonl(self.output_dir)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def preprocess(self, examples, tokenizer):
        all_input_ids = []
        print("Tokenizing dataset...")
        for ex in tqdm(examples):
            questions = ex['questions']
            for answer in ex['answers']:
                text = f"{ex['context']}\n\nQ: {questions}\nA: {answer}\n"
                tokenized = tokenizer.encode(text)
                all_input_ids.append(torch.LongTensor(tokenized))

            # Generate a negative example
            random_ex = random.choice(examples)
            random_questions = random_ex['questions']
            random_answers = " ; ".join(random_ex['answers'])
            text = f"{ex['context']}\n\nQ: {random_questions}\nA: I don't know.\n"
            tokenized = tokenizer.encode(text)
            all_input_ids.append(torch.LongTensor(tokenized))

        random.shuffle(all_input_ids)
        return dict(input_ids=all_input_ids, labels=all_input_ids)

    def save_to_jsonl(self, output_dir):
        output_file = f"{output_dir}/preprocessed_data.jsonl"
        print(f"Saving processed data to {output_file}")
        with open(output_file, "w") as file:
            for input_id, label in zip(self.input_ids, self.labels):
                
                json_record = {
                    "input_ids": input_id.tolist(),
                    "labels": label.tolist()         
                }
                file.write(json.dumps(json_record) + "\n") 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output", help="Directory to save preprocessed data")
    parser.add_argument("--tokenizer", type=str, default="state-spaces/mamba-130m-hf", help="Tokenizer model name")
    parser.add_argument("--data_path", type=str, default="data/sample_data.json", help="Path to the input data file")
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    faq_dataset = FAQ_data(data_path=args.data_path, tokenizer=tokenizer, output_dir=args.output)





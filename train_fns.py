# vim: set textwidth=0:

"""
# FINDSUM methods of summarization

## Only Text
- (in: long_text) summarizer_model = (out: summary) (tgt: text)
- only train summarizer model

## Generate and Combine (GC)
- ((in: long_text) summarizer_model = (out: summary)) + ((in: linearized_table) table2text = (out: t2t_gen)) = (out: summary) (tgt: text)
- train summarizer model and table2text model

## Combine and Generate (CG)
- (in: src_text = linearized_table + long_text) (table2text -> summarizer_model) (out: summary)
- pretrain table2text and finetune to summarization model 

## Generate, Combine and Generate (GCG)
- (in: linearized_table) table2text = (out: t2t_gen)
- (in: t2t_gen + long_text) summarizer_model = (out: summary) (tgt: text)
- train table2text model and embed it inside summarizer_model's pipeline for training

## T&T Summarization with LF
- (in: src_text = linearized_table + long_text) (table2logic -> table2text -> summarizer_model) (out: summary)
- pretrain table2logic and finetune to table2text and to summarization model 
"""

import math
from torch.utils.data import Dataset
from torch import nn
import torch
import json

class FnsDataset(Dataset): 
    def __init__(self, data_file, tokenizer, max_src_len, max_tgt_len, task):
        self.data = self.load_data(data_file)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.task = task
        
    def load_data(self, file):
        # Format: [{'document': doc_text, 'summary': summary_text}, ...]
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        src_text = d['src_text'].strip()
        src_text = ' '.join(src_text.split())

        # if self.task == 'text' and 'sent' in d:
        #     tgt_text = d['sent'].strip()
        # elif self.task == 'logic' and 'logic_str' in d:
        #     tgt_text = d['logic_str'].strip()
        if self.task == 'ot' and 'text' in d:
            tgt_text = d['text'].strip()
        if self.task == 'gc' and 'long_text' in d:
            tgt_text = d['long_text'].strip()
        else:
            raise NotImplementedError
        tgt_text = ' '.join(tgt_text.split())

        source = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_len
        )
        target = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tgt_len
        )

        source_input_ids = torch.LongTensor(source.data["input_ids"])
        target_input_ids = torch.LongTensor(target.data["input_ids"])
        source_mask = torch.LongTensor(source.data["attention_mask"])

        return {
            'source_ids': source_input_ids,
            'source_mask': source_mask,
            'target_ids': target_input_ids
        }

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from tqdm import tqdm
import os

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="facebook/bart-base")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to perform training")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to perform test")

    parser.add_argument('--optimizer', default='Adamw', choices=['Adamw', 'Adafactor'], type=str)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--every', default=50, type=int, help="interval for evaluation")
    parser.add_argument('--interval_type', default='step', type=str, choices=['step', 'epoch'], help="whether to evaluate at intervals based on steps or epochs")
    parser.add_argument('--interval_step', default=1000, type=int, help="interval for evaluation when interval_type = step.")
    parser.add_argument('--load_from', default=None, type=str, help="model checkpoint path")
    parser.add_argument('--max_src_len', default=1024, type=int, help="max length of input sequence")
    parser.add_argument('--max_tgt_len', default=200, type=int, help="target output length")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help="number of steps to accumulate gradients before updating parameters")

    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/Dataset/SciGenMod')
    parser.add_argument('--log_path',type=str, default='/content/drive/MyDrive/Output/logs/d2t/outputs')
    parser.add_argument('--ckpt_path', type=str, default='/content/drive/MyDrive/Output/models/d2t')
    parser.add_argument('--train_file', type=str, required=True, help="Path to training data", default='train-400.json')
    parser.add_argument('--val_file', type=str, required=True, help="Path to validation data", default='valid-50.json')
    parser.add_argument('--test_file', type=str, required=True, help="Path to test data", default='test-50.json')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--affix', type=str, default=None, required=True, help="The experiment name")
    parser.add_argument('--n_gpu', type=str, default=0, help="number of GPU to use")
    parser.add_argument('--task', type=str, default='text', help='task: only text (ot), gc, cg, gcg')
    parser.add_argument('--global_step', default=1, type=int, help="initialize global step counter")
    parser.add_argument('--use_cache', default=False, action="store_true", help="enable caching mechanisms")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(args.device)

    # layer freezing for retaining features
    def freeze_params(model: nn.Module):
        for par in model.parameters():
            par.requires_grad = False

    if args.do_train:
        # freeze parameters
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_tokens)

        # Load dataset
        train_dataset = FnsDataset(args.train_data, tokenizer, args.max_src_len, args.max_tgt_len, args.task)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        model.train()

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        global_step = 0
        total_loss = []
        best_val = 0

        # Training loop
        for epoch in range(1, args.epochs + 1):
            print("[INFO] start training {}th epoch".format(epoch))
            for idx, batch in enumerate(tqdm(train_loader)):
                # Prepare inputs and targets
                labels = batch['target_ids'].to(args.device, dtype=torch.long)
                labels[labels == tokenizer.pad_token_id] = -100
                input_ids = batch['source_ids'].to(args.device, dtype=torch.long)
                attention_mask = batch['source_mask'].to(args.device, dtype=torch.long)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                # gradient accumulation, loss scaling, and backpropagation
                loss = loss / args.gradient_accumulation_steps
                total_loss.append(loss.item())
                loss.backward()

                # optimizer update model parameter using gradients every gradient accumulation step 
                if (idx + 1) % args.gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
                    optimizer.step()
                    model.zero_grad()

                # Perplexity - measure of how well the model is able to predict a sequence every args.every
                if idx % args.every == 0 and idx > 0:
                    perplexity = math.exp(np.mean(total_loss))
                    total_loss = []

            print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader)}")

            # Save model checkpoint
            if args.save_model:
                output_dir = os.path.join(args.save_path, f"epoch_{epoch + 1}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

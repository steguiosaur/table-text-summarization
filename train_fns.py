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
from pyrouge import Rouge155
import os, io, re, subprocess
import logging

def linearize_table_data(data: dict, add_type: bool = False, pre_com: bool = False) -> dict:
    # Extract the json structure
    table_caption = data["table_caption"]
    table_header = data["table_column_names"]
    table_contents = data["table_content_values"]
    # textual_data = data["long_text"]

    # Initialize source text with the caption
    src_text = f"<table> <caption> {str(table_caption)} </caption> "

    # Initialize a list to hold the linearized data
    linearized_data = []

    # Determine the number of rows (assuming all rows are of equal length)
    num_rows = len(table_contents)

    # Create a linearized representation of the table
    for row_idx in range(num_rows):
        for header in table_header:
            # Get the index of the current header
            header_index = table_header.index(header)
            # Get the corresponding cell value, or None if out of bounds
            cell_value = table_contents[row_idx][header_index] if header_index < len(table_contents[row_idx]) else None
            
            # Construct cell string for each cell in the table
            cell_str = f"<cell> {cell_value} <col_header> {header} </col_header> <row_idx> {row_idx} </row_idx> </cell> "
            linearized_data.append(cell_str)

    # Join all cell strings together and add to source text
    src_text += ''.join(linearized_data)

    # Add caption at the end of the table linearization
    src_text += f"</table>"

    # Assign linearized text to 'src_text' field in data
    data['src_text'] = src_text

    return data

def preprocess_summarization(data_file: str):
    with open(data_file, 'r') as f:
        data = json.load(f)

        preprocessed_data = []

        # Process each item in the dataset
        for key, item in tqdm(data.items()):
            # Extract relevant fields
            table_caption = item['table_caption']
            table_column_names = item['table_column_names']
            table_content_values = item['table_content_values']
            textual_data = item['long_text']
            text = item['text']

            # Prepare the data structure
            processed_item = {
                'table_caption': table_caption,
                'table_column_names': table_column_names,
                'table_content_values': table_content_values,
                'long_text': textual_data,
                'text': text,
            }

            # Linearize the table and add src_text
            processed_item = linearize_table_data(processed_item)

            # Add the processed item to the preprocessed data list
            preprocessed_data.append(processed_item)

        return preprocessed_data

class FnsDataset(Dataset): 
    def __init__(self, data_file, tokenizer, max_src_len, max_tgt_len, task):
        self.data = preprocess_summarization(data_file)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.task = task
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        if self.task == 'ot':
            src_text = d['long_text'].strip()
        else:
            src_text = d['src_text'].strip()
        src_text = ' '.join(src_text.split())

        tgt_text = d['text'].strip()
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

def validation_task(val_file, summ_model, summ_tokenizer, tbltxt_model, tbltxt_tokenizer, split, args):
    val_dataset = FnsDataset(val_file, tokenizer, args.max_src_len, args.max_tgt_len, args.task)
    val_loader = DataLoader(val_dataset,
                            num_workers=5,
                            batch_size=args.batch_size,
                            shuffle=False)
    model.eval()
    pred_list = []
    ref_list = []

    # create files for scripts
    gt = open(os.path.join(args.log_path, args.affix, f'references_{split}.txt'), 'w')
    pred = open(os.path.join(args.log_path, args.affix, f'predictions_{split}.txt'), 'w')
    pred_split = os.path.join(args.log_path, args.affix, f'predictions_{split}/')
    ref_split = os.path.join(args.log_path, args.affix, f'references_{split}/')
    if not os.path.exists(pred_split):
        os.makedirs(pred_split)
    if not os.path.exists(ref_split):
        os.makedirs(ref_split)
    k = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            y = batch['target_ids'].to(args.device, dtype=torch.long)
            ids = batch['source_ids'].to(args.device, dtype=torch.long)
            mask = batch['source_mask'].to(args.device, dtype=torch.long)
            samples = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=args.max_tgt_len,
                num_beams=4,
                early_stopping=False
            )


            for reference, s in zip(y, samples):
                with open(ref_split + str(k) + '_reference.txt', 'w') as sr, \
                        open(pred_split + str(k) + '_prediction.txt', 'w') as sw:
                    reference = tokenizer.decode(reference, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
                    gt.write(reference.lower() + '\n')
                    sr.write(reference.lower() + '\n')
                    ref_list.append(reference.lower())
                    text = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_list.append(text.lower())
                    pred.write(text.lower() + '\n')
                    sw.write(text.lower() + '\n')
                    k += 1
        gt.close()
        pred.close()

        if args.task == 'text' or args.task == 'ot':
            gt = os.path.join(args.log_path, args.affix, f'references_{split}.txt')
            pred = os.path.join(args.log_path, args.affix, f'predictions_{split}.txt')
            # bleu4 = bleu_score(gt, pred)
            # print("[INFO] {} BLEU score = {}".format(split, bleu4))
            # log_file.write("[INFO] {} BLEU score = {}\n".format(split, bleu4))

            # ROUGE scripts
            r = Rouge155()
            r.system_dir = os.path.join(args.log_path, args.affix, f'predictions_{split}/')
            r.model_dir = os.path.join(args.log_path, args.affix, f'references_{split}/')
            # define the patterns
            r.system_filename_pattern = '(\d+)_prediction.txt'
            r.model_filename_pattern = '#ID#_reference.txt'
            logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
            results_dict = r.convert_and_evaluate()
            rouge_result = "\n".join(
                [results_dict.split("\n")[3], results_dict.split("\n")[7], results_dict.split("\n")[15],
                 results_dict.split("\n")[19]])
            print("[INFO] Rouge scores: \n", rouge_result)
            # log_file.write(rouge_result + '\n')
            results_dict = results_dict.split("\n")
            rouge_score_list = []

            for i in [3, 7, 15, 19]:
                results = results_dict[i]
                rouge_score = float(results.split()[3])
                rouge_score_list.append(rouge_score * 100)

        val_metric_dict = {}
        if args.task == 'ot' or args.task == 'summ':
            for type, score in zip(['1', '2', '4', 'L'], rouge_score_list):
                val_metric_dict[f'rouge{type}'] = score
        else:
            raise NotImplementedError
        return val_metric_dict

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
    parser.add_argument('--summ_load_from', default=None, type=str, help="model checkpoint path")
    parser.add_argument('--tbx_load_from', default=None, type=str, help="model checkpoint path")
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
    summ_tokenizer = AutoTokenizer.from_pretrained(args.model)
    summ_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    summ_model.to(args.device)

    tbltxt_tokenizer = AutoTokenizer.from_pretrained(args.model)
    markers = ["{", "}", "<table>", "</table>", "<type>", "</type>", "<cell>", "</cell>", "<col_header>", "</col_header>", "<row_idx>", "</row_idx>"]
    tbltxt_tokenizer.add_tokens(markers)
    tbltxt_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tbltxt_model.to(args.device)
    tbltxt_model.resize_token_embeddings(len(tbltxt_tokenizer))

    if args.summ_load_from is not None:
        summ_model.load_state_dict(torch.load(args.summ_load_from))
    if args.tbx_load_from is not None:
        tbltxt_model.load_state_dict(torch.load(args.tbx_load_from))

    # create directories to store logs and models
    if not os.path.exists(os.path.join(args.log_path, args.affix)):
        os.makedirs(os.path.join(args.log_path, args.affix))
    if not os.path.exists(os.path.join(args.ckpt_path, args.affix)):
        os.makedirs(os.path.join(args.ckpt_path, args.affix))

    train_file = os.path.join(args.data_path, args.train_file)
    val_file = os.path.join(args.data_path, args.val_file)
    test_file = os.path.join(args.data_path, args.test_file)

    # just test model, no training
    if args.do_test:
        test_scores = validation_task(test_file, summ_model, summ_tokenizer, tbltxt_model, tbltxt_tokenizer, 'test', args)
        print(test_scores)

import json
from torch.utils.data import Dataset
from tqdm import tqdm

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

def preprocess_summarization(data_file: str, add_type: bool, pre_com: bool):
    # Load the dataset
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Load predictions from the tabletotext folder
    predictions_path = "/content/drive/MyDrive/Output/logs/d2t/outputs/tbl2textnolf_test/predictions_test.txt"
    with open(predictions_path, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]

    preprocessed_data = []
    
    # Process each item and concatenate predictions
    for idx, (key, item) in enumerate(data.items()):
        textual_data = item['long_text']
        text = item['text']
        
        # Concatenate prediction to the long_text field
        prediction = predictions[idx] if idx < len(predictions) else ""
        combined_text = f"{textual_data} {prediction}"
        
        # Prepare the processed data item
        processed_item = {
            'long_text': combined_text,
            'text': text,
        }
        
        # Add to the preprocessed data list
        preprocessed_data.append(processed_item)

    return preprocessed_data

class ScigenDataset(Dataset):
    def __init__(self, file, tokenizer, max_src_len, max_tgt_len, add_type, pre_com):
        self.data = preprocess_summarization(file, add_type, pre_com)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]

        # Get the source text from the linearized data
        src_text = d['long_text'].strip()
        src_text = ' '.join(src_text.split())

        tgt_text = d['text'].strip()
        tgt_text = ' '.join(tgt_text.split())

        # Tokenize source and target texts
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

        # Convert tokenized data to tensors
        source_input_ids = torch.LongTensor(source.data["input_ids"])
        target_input_ids = torch.LongTensor(target.data["input_ids"])
        source_mask = torch.LongTensor(source.data["attention_mask"])

        return {
            'source_ids': source_input_ids,
            'source_mask': source_mask,
            'target_ids': target_input_ids
        }

import os, io, re, subprocess
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer, scoring


def bleu_score(labels_file, predictions_path):
    bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    try:
        with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
            bleu_out = subprocess.check_output(
                [bleu_script, labels_file],
                stdin=predictions_file,
                stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            return float(bleu_score)

    except subprocess.CalledProcessError as error:
        return None


def validation_task(val_file, model, tokenizer, split, args):
    val_dataset = ScigenDataset(val_file, tokenizer, args.max_src_len, args.max_tgt_len, args.add_type, args.pre_com)
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


        gt = os.path.join(args.log_path, args.affix, f'references_{split}.txt')
        pred = os.path.join(args.log_path, args.affix, f'predictions_{split}.txt')

        # ROUGE evaluation using `rouge_score`
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        with open(gt, 'r') as gt_file, open(pred, 'r') as pred_file:
            references = gt_file.readlines()
            predictions = pred_file.readlines()

            for idx, (ref, pred) in enumerate(zip(references, predictions)):
                ref = ref.strip()
                pred = pred.strip()

                # Compute ROUGE scores for each sample
                scores = scorer.score(ref, pred)
                aggregator.add_scores(scores)

                # Log per-sample ROUGE scores
                print(f"Iteration {idx + 1} ROUGE Scores:")
                for rouge_type, score in scores.items():
                    print(f"  {rouge_type.upper()}: Precision: {score.precision:.4f}, Recall: {score.recall:.4f}, F1: {score.fmeasure:.4f}")
                print()

        # Compute and log overall ROUGE scores
        result = aggregator.aggregate()
        overall_scores = {}
        print("\n[INFO] Overall ROUGE Scores:")
        for rouge_type, value in result.items():
            overall_scores[rouge_type] = {
                "Precision": value.mid.precision * 100,
                "Recall": value.mid.recall * 100,
                "F1": value.mid.fmeasure * 100
            }
            print(f"  {rouge_type.upper()}: Precision={overall_scores[rouge_type]['Precision']:.2f}, "
                  f"Recall={overall_scores[rouge_type]['Recall']:.2f}, "
                  f"F1={overall_scores[rouge_type]['F1']:.2f}")

        return overall_scores

import math
import random
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Adafactor, AdamW

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='facebook/bart-base', type=str, help="specify the pretrained model to use")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization and reproducibility")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to perform training")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to perform validation")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to perform testing")
    parser.add_argument('--pretrain', default=False, action="store_true", help="whether to train or test the model")

    parser.add_argument('--optimizer', default='Adamw', choices=['Adamw', 'Adafactor'], type=str)
    parser.add_argument('--epoch', default=1, type=int, help="number of epochs for training")
    parser.add_argument('--batch_size', default=5 , type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--every', default=50, type=int, help="interval for evaluation")
    parser.add_argument('--interval_type', default='step', type=str, choices=['step', 'epoch'], help="whether to evaluate at intervals based on steps or epochs")
    parser.add_argument('--interval_step', default=1000, type=int, help="interval for evaluation when interval_type = step.")
    parser.add_argument('--load_from', default=None, type=str, help="model checkpoint path")
    parser.add_argument('--max_src_len', default=1024, type=int, help="max length of input sequence")
    parser.add_argument('--max_tgt_len', default=200, type=int, help="target output length")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help="number of steps to accumulate gradients before updating parameters")

    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/Dataset/contlog')
    parser.add_argument('--log_path',type=str, default='/content/drive/MyDrive/Output/logs/d2t/outputs')
    parser.add_argument('--ckpt_path', type=str, default='/content/drive/MyDrive/Output/models/d2t')
    parser.add_argument('--affix', type=str, default=None, required=True, help="The experiment name")
    parser.add_argument('--device', type=str, default='cuda', help="specifies the device to use for computations (CUDA only)")
    parser.add_argument('--n_gpu', type=str, default=0, help="number of GPU to use")
    parser.add_argument('--task', type=str, default='text', help='task: text (table2text) or logic (table2logic) or summ (tabletextsumm)')
    parser.add_argument('--add_type', default=False, action="store_true", help="indicate whether to add type information to the input")
    parser.add_argument('--pre_com', default=False, action="store_true", help="whether to do numerical precomputation")
    parser.add_argument('--global_step', default=1, type=int, help="initialize global step counter")
    parser.add_argument('--use_cache', default=False, action="store_true", help="enable caching mechanisms")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    set_seed(args) # for reproducibility

    # add special tokens to tokenizer for linearized table structure
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    markers = ["{", "}", "<table>", "</table>", "<type>", "</type>", "<cell>", "</cell>", "<col_header>", "</col_header>", "<row_idx>", "</row_idx>"]
    if args.pre_com:
        markers += ["<max_rank>", "</max_rank>", "<min_rank>", "</min_rank>", "<sum_cell>", "</sum_cell>", "<avg_cell>", "</avg_cell>"]
    tokenizer.add_tokens(markers)

    # model setup, load pretrained weights, and accomodate special tokens
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    if args.load_from is not None:
        model.load_state_dict(torch.load(args.load_from))

    # layer freezing for retaining features
    def freeze_params(model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    # loss function
    #criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    # create directories to store logs and models
    if not os.path.exists(os.path.join(args.log_path, args.affix)):
        os.makedirs(os.path.join(args.log_path, args.affix))
    if not os.path.exists(os.path.join(args.ckpt_path, args.affix)):
        os.makedirs(os.path.join(args.ckpt_path, args.affix))

    # set pretraining data location
    # if args.task == 'logic':
    #     train_file = os.path.join(args.data_path, 'all_pretrain_train_s.json')
    #     val_file = os.path.join(args.data_path, 'all_pretrain_valid.json')
    #     test_file = os.path.join(args.data_path, 'all_pretrain_test.json')
    # elif args.task == 'text':
    #     train_file = os.path.join(args.data_path, 'train.json')
    #     val_file = os.path.join(args.data_path, 'val.json')
    #     test_file = os.path.join(args.data_path, 'test.json')
    # elif args.task == 'summ':
    train_file = os.path.join(args.data_path, 'train-400.json')
    val_file = os.path.join(args.data_path, 'valid-50.json')
    test_file = os.path.join(args.data_path, 'test-50.json')
    # else:
    #     raise NotImplementedError

    if args.do_train:
        # freeze embedding layers
        if args.model.startswith('t5'):
            for d in [model.encoder]:
                freeze_params(d.embed_tokens)
        else:
            for d in [model.model.encoder, model.model.decoder]:
                # freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

        train_dataset = ScigenDataset(train_file, tokenizer, args.max_src_len, args.max_tgt_len, args.add_type, args.pre_com)
        train_loader = DataLoader(train_dataset, num_workers=5, batch_size=args.batch_size, shuffle=True)
        model.train()

        if args.optimizer == 'Adamw':
            optimizer = AdamW(model.parameters(), args.learning_rate)
        elif args.optimizer == 'Adafactor':
            optimizer = Adafactor(model.parameters(), args.learning_rate, relative_step=False)
        else:
            raise NotImplementedError

        global_step = 0
        total_loss = []
        # best validation score
        best_val = 0
        # best_metric = "bleu4" if args.task == 'text' else "exec_acc"

        for epoch_idx in range(1, args.epoch+1):
            print("[INFO] start training {}th epoch".format(epoch_idx))
            for idx, batch in enumerate(tqdm(train_loader)):
                # data preparation
                lm_labels = batch['target_ids'].to(args.device, dtype=torch.long)
                lm_labels[lm_labels == tokenizer.pad_token_id] = -100
                ids = batch['source_ids'].to(args.device, dtype=torch.long)
                mask = batch['source_mask'].to(args.device, dtype=torch.long)

                # forward pass and loss calculation
                outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                # gradient accumulation, loss scaling, and backpropagation
                loss = loss / args.gradient_accumulation_steps
                total_loss.append(loss.item())
                loss.backward()

                # optimizer update model parameter using gradients every gradient accumulation step 
                if (idx + 1) % args.gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()
                
                # Perplexity - measure of how well the model is able to predict a sequence every args.every
                if idx % args.every == 0 and idx > 0:
                    perplexity = math.exp(np.mean(total_loss))
                    total_loss = []

                # save model every args.interval_type step or epoch
                if (args.interval_type == 'step' and global_step % args.interval_step == 0 and global_step > 0) \
                    or (args.interval_type == 'epoch' and (idx + 1) == len(train_loader)):
                    if args.interval_type == 'step':
                        torch.save(model.state_dict(), '{}/{}/{}_step{}.pt'.format(args.ckpt_path, args.affix, args.model.split('/')[-1], global_step))
                    else:
                        torch.save(model.state_dict(), '{}/{}/{}_ep{}.pt'.format(args.ckpt_path, args.affix, args.model.split('/')[-1], epoch_idx))

                    # validation step
                    val_scores = validation_task(val_file, model, tokenizer, 'valid', args)
                    if args.task == "logic" and val_scores["exec_acc"] > best_val:
                        best_val = val_scores["exec_acc"]
                    test_scores = validation_task(test_file, model, tokenizer, 'test', args)
                global_step += 1

    # just test model, no training
    if args.do_test:
        test_scores = validation_task(test_file, model, tokenizer, 'test', args)
        print(test_scores)

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from pyrouge import Rouge155

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer from a specified path."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(args.device)
    return model, tokenizer

def generate_text(model, tokenizer, input_text, max_len):
    """Generate text given a model, tokenizer, and input."""
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_len
    ).to(args.device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def gc_evaluation(test_data, summarizer_model, summarizer_tokenizer, table2text_model, table2text_tokenizer, args):
    """Generate and Combine (GC) method evaluation on test data."""
    pred_list = []
    ref_list = []
    gt = open(os.path.join(args.log_path, f'gc_references.txt'), 'w')
    pred = open(os.path.join(args.log_path, f'gc_predictions.txt'), 'w')

    for item in tqdm(test_data):
        # Extract table and long text
        table_input = item['table_content_values']
        long_text = item['long_text']
        reference_summary = item['text']

        # Generate table2text output
        t2t_gen = generate_text(table2text_model, table2text_tokenizer, table_input, args.max_src_len)

        # Concatenate table2text output with the long text
        combined_input = f"{t2t_gen} {long_text}"
        
        # Generate final summary from summarizer
        gc_summary = generate_text(summarizer_model, summarizer_tokenizer, combined_input, args.max_tgt_len)

        # Write and collect predictions and references
        pred_list.append(gc_summary.lower())
        ref_list.append(reference_summary.lower())
        pred.write(gc_summary.lower() + '\n')
        gt.write(reference_summary.lower() + '\n')

    gt.close()
    pred.close()

    # Calculate ROUGE score
    r = Rouge155()
    r.system_dir = args.log_path
    r.model_dir = args.log_path
    r.system_filename_pattern = 'gc_predictions.txt'
    r.model_filename_pattern = 'gc_references.txt'
    
    logging.getLogger('global').setLevel(logging.WARNING)  # Silence pyrouge logging
    results_dict = r.convert_and_evaluate()
    print("[INFO] ROUGE scores for Generate and Combine (GC) method:\n", results_dict)

# Usage
if __name__ == "__main__":
    summarizer_model_path = 'path/to/summarizer_model'
    table2text_model_path = 'path/to/table2text_model'

    summarizer_model, summarizer_tokenizer = load_model_and_tokenizer(summarizer_model_path)
    table2text_model, table2text_tokenizer = load_model_and_tokenizer(table2text_model_path)

    # Load the test data as a list of dictionaries, where each dict contains 'table_content_values', 'long_text', and 'text' fields.
    test_data = preprocess_summarization(args.test_file)

    # Run the Generate and Combine evaluation on the test set
    gc_evaluation(test_data, summarizer_model, summarizer_tokenizer, table2text_model, table2text_tokenizer, args)


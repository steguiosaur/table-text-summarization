import logging
import os
from pyrouge import Rouge155

def concatenate_predictions(tabletotext_path, summarizedt_path, gcsummarize_path, split):
    # Define the input and output directories
    tabletotext_pred_path = os.path.join(tabletotext_path, f'predictions_{split}/')
    summarizedt_pred_path = os.path.join(summarizedt_path, f'predictions_{split}/')
    gcsummarize_pred_path = os.path.join(gcsummarize_path, f'predictions_{split}/')

    # Create the output directory if it doesn't exist
    os.makedirs(gcsummarize_pred_path, exist_ok=True)

    # Concatenate predictions from `tabletotext` and `summarizedt` for each file
    for i in range(50):  # Assuming files are named from 0 to 49
        tabletotext_file = os.path.join(tabletotext_pred_path, f"{i}_prediction.txt")
        summarizedt_file = os.path.join(summarizedt_pred_path, f"{i}_prediction.txt")
        gcsummarize_file = os.path.join(gcsummarize_pred_path, f"{i}_prediction.txt")

        # Read the contents of both files and concatenate them
        with open(tabletotext_file, 'r') as f1, open(summarizedt_file, 'r') as f2:
            text1 = f1.read().strip()
            text2 = f2.read().strip()
            concatenated_text = text1 + " " + text2

        # Write the concatenated text to `gcsummarize` predictions directory
        with open(gcsummarize_file, 'w') as f3:
            f3.write(concatenated_text)

def validation_task(split="test"):
    log_path = "/content/drive/MyDrive/Output/logs/d2t/outputs"
    tabletotext = os.path.join(log_path, "tbl2textnolf_test")
    summarizedt = os.path.join(log_path, "only_text")
    gcsummarize = os.path.join(log_path, "gc_test")

    # Ensure output directories exist
    if not os.path.exists(gcsummarize):
        os.makedirs(gcsummarize)

    # Concatenate predictions from `tabletotext` and `summarizedt` into `gcsummarize`
    concatenate_predictions(tabletotext, summarizedt, gcsummarize, split)

    # Paths for the reference and concatenated predictions for ROUGE evaluation
    ref_split = os.path.join(summarizedt, f'references_{split}/')
    pred_split = os.path.join(gcsummarize, f'predictions_{split}/')
    
    # Generate single-file references and predictions for ROUGE scoring
    gt_path = os.path.join(gcsummarize, f'references_{split}.txt')
    pred_path = os.path.join(gcsummarize, f'predictions_{split}.txt')

    # Aggregate reference files into a single reference text file
    with open(gt_path, 'w') as gt_file:
        for i in range(50):  # Assuming reference files range from 0 to 49
            ref_file = os.path.join(ref_split, f"{i}_reference.txt")
            with open(ref_file, 'r') as rf:
                gt_file.write(rf.read().strip() + "\n")

    # Aggregate concatenated prediction files into a single prediction text file
    with open(pred_path, 'w') as pred_file:
        for i in range(50):  # Assuming prediction files range from 0 to 49
            pred_file_path = os.path.join(pred_split, f"{i}_prediction.txt")
            with open(pred_file_path, 'r') as pf:
                pred_file.write(pf.read().strip() + "\n")

    # ROUGE evaluation using `pyrouge`
    r = Rouge155()
    r.system_dir = pred_split
    r.model_dir = ref_split
    r.system_filename_pattern = '(\d+)_prediction.txt'
    r.model_filename_pattern = '#ID#_reference.txt'
    
    logging.getLogger('global').setLevel(logging.WARNING)  # Suppress verbose logging
    results_dict = r.convert_and_evaluate()
    
    # Extracting and formatting ROUGE scores
    results_lines = results_dict.split("\n")
    rouge_score_list = []
    for i in [3, 7, 15, 19]:  # Select specific lines for ROUGE-1, ROUGE-2, ROUGE-4, ROUGE-L
        results = results_lines[i]
        rouge_score = float(results.split()[3])  # Extract score as a float
        rouge_score_list.append(rouge_score * 100)

    val_metric_dict = {}
    for type, score in zip(['1', '2', '4', 'L'], rouge_score_list):
        val_metric_dict[f'rouge{type}'] = score
    
    # Print ROUGE scores for confirmation
    print("[INFO] Rouge scores: ", val_metric_dict)

    return val_metric_dict

# Example usage
validation_task("test")
